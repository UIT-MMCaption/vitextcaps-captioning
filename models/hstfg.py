"""
HSTFG v2: Heterogeneous Scene-Text Fusion Graph for Vietnamese Image Captioning

Architecture:
  1. Node Embedding: V (visual regions), T (OCR tokens) — no L nodes
  2. Learned Spatial Attention: pairwise spatial features → MLP → attention bias per head
  3. MMT Decoder with OcrPtrNet copy mechanism (reused from M4C)

Changes from v1:
  - Removed heuristic graph construction (line grouping, IoU edges, reading order)
  - Replaced HeteroGATLayer with SpatialHeteroAttentionLayer
  - Model learns V-T and T-T connectivity from spatial features end-to-end
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig

from utils.logging_utils import setup_logger
from builders.model_builder import META_ARCHITECTURE
from models.utils import generate_padding_mask, generate_sequential_mask
from models.mmf_m4c import MMT, OcrPtrNet, PrevPredEmbeddings, TextBert, _batch_gather

logger = setup_logger()


# ---------------------------------------------------------------------------
# Spatial Features
# ---------------------------------------------------------------------------

def compute_spatial_features(boxes_a, boxes_b):
    """Compute pairwise spatial features between two sets of boxes.

    Args:
        boxes_a: (B, Na, 4) [x1, y1, x2, y2]
        boxes_b: (B, Nb, 4) [x1, y1, x2, y2]
    Returns: (B, Na, Nb, 4) features [dx, dy, dw, dh]
    """
    cx_a = (boxes_a[..., 0] + boxes_a[..., 2]) / 2
    cy_a = (boxes_a[..., 1] + boxes_a[..., 3]) / 2
    w_a = (boxes_a[..., 2] - boxes_a[..., 0]).clamp(min=1e-6)
    h_a = (boxes_a[..., 3] - boxes_a[..., 1]).clamp(min=1e-6)

    cx_b = (boxes_b[..., 0] + boxes_b[..., 2]) / 2
    cy_b = (boxes_b[..., 1] + boxes_b[..., 3]) / 2
    w_b = (boxes_b[..., 2] - boxes_b[..., 0]).clamp(min=1e-6)
    h_b = (boxes_b[..., 3] - boxes_b[..., 1]).clamp(min=1e-6)

    dx = (cx_b.unsqueeze(1) - cx_a.unsqueeze(2)) / w_a.unsqueeze(2)
    dy = (cy_b.unsqueeze(1) - cy_a.unsqueeze(2)) / h_a.unsqueeze(2)
    dw = torch.log(w_b.unsqueeze(1) / w_a.unsqueeze(2))
    dh = torch.log(h_b.unsqueeze(1) / h_a.unsqueeze(2))

    return torch.stack([dx, dy, dw, dh], dim=-1)


# ---------------------------------------------------------------------------
# Spatial Heterogeneous Attention Layer
# ---------------------------------------------------------------------------

class SpatialHeteroAttentionLayer(nn.Module):
    """Single layer of heterogeneous attention with learned spatial bias.

    3 attention blocks (no L-related edges):
      - V→T: T queries attend to V keys, spatial bias from T-V box pairs
      - T→V: V queries attend to T keys, spatial bias from V-T box pairs
      - T→T: T self-attention, spatial bias from T-T box pairs
    """

    def __init__(self, hidden_size, num_heads, spatial_hidden=32, dropout=0.1,
                 use_spatial_bias=True, edge_types=None, use_confidence_gate=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0

        self.use_spatial_bias = use_spatial_bias
        self.use_confidence_gate = use_confidence_gate
        self.active_edge_types = edge_types if edge_types is not None else ['vt', 'tv', 'tt']

        # Always create all 3 edge types so checkpoint loading works
        all_edge_types = ['vt', 'tv', 'tt']

        # Per-edge-type Q/K/V projections
        self.queries = nn.ModuleDict({
            et: nn.Linear(hidden_size, hidden_size) for et in all_edge_types
        })
        self.keys = nn.ModuleDict({
            et: nn.Linear(hidden_size, hidden_size) for et in all_edge_types
        })
        self.values = nn.ModuleDict({
            et: nn.Linear(hidden_size, hidden_size) for et in all_edge_types
        })

        # Spatial bias MLP per edge type: 4 → spatial_hidden → num_heads
        self.spatial_mlps = nn.ModuleDict({
            et: nn.Sequential(
                nn.Linear(4, spatial_hidden),
                nn.ReLU(),
                nn.Linear(spatial_hidden, num_heads),
            ) for et in all_edge_types
        })

        # Confidence gate per edge type
        self.conf_gates = nn.ModuleDict({
            et: nn.Linear(1, num_heads) for et in all_edge_types
        })

        # Output projections per node type
        self.out_proj_v = nn.Linear(hidden_size, hidden_size)
        self.out_proj_t = nn.Linear(hidden_size, hidden_size)

        # Layer norms
        self.norm_v = nn.LayerNorm(hidden_size)
        self.norm_t = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def _attend(self, query_emb, key_emb, value_emb,
                query_boxes, key_boxes,
                key_mask, conf_scores, edge_type):
        """Compute spatial-bias multi-head attention for one edge type.

        Args:
            query_emb: (B, Nq, D)
            key_emb: (B, Nk, D)
            value_emb: (B, Nk, D)
            query_boxes: (B, Nq, 4) bounding boxes for queries
            key_boxes: (B, Nk, 4) bounding boxes for keys
            key_mask: (B, Nk) True=valid, False=padding
            conf_scores: (B, Nk, 1) source confidence
            edge_type: str key for parameter lookup

        Returns: (B, Nq, D) aggregated messages
        """
        B, Nq, D = query_emb.shape
        Nk = key_emb.shape[1]
        H = self.num_heads

        Q = self.queries[edge_type](query_emb).view(B, Nq, H, self.head_dim).transpose(1, 2)
        K = self.keys[edge_type](key_emb).view(B, Nk, H, self.head_dim).transpose(1, 2)
        V = self.values[edge_type](value_emb).view(B, Nk, H, self.head_dim).transpose(1, 2)

        # Dot-product attention
        attn = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # (B, H, Nq, Nk)

        # Spatial bias (ablation: can be disabled)
        if self.use_spatial_bias:
            spatial_feat = compute_spatial_features(query_boxes, key_boxes)  # (B, Nq, Nk, 4)
            spatial_bias = self.spatial_mlps[edge_type](spatial_feat)  # (B, Nq, Nk, H)
            spatial_bias = spatial_bias.permute(0, 3, 1, 2)  # (B, H, Nq, Nk)
            attn = attn + spatial_bias

        # Mask padding keys
        key_mask_4d = key_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Nk)
        attn = attn.masked_fill(~key_mask_4d, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Confidence gating (ablation: can be disabled)
        if self.use_confidence_gate:
            conf_gate = torch.sigmoid(self.conf_gates[edge_type](conf_scores))  # (B, Nk, H)
            conf_gate = conf_gate.transpose(1, 2).unsqueeze(2)  # (B, H, 1, Nk)
            attn = attn * conf_gate

        out = torch.matmul(attn, V)  # (B, H, Nq, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)
        return out

    def forward(self, v_emb, t_emb,
                v_boxes, t_boxes,
                v_conf, t_conf,
                v_mask, t_mask):
        """
        Args:
            v_emb: (B, Nv, D) visual node embeddings
            t_emb: (B, Nt, D) OCR token node embeddings
            v_boxes: (B, Nv, 4) visual region bounding boxes
            t_boxes: (B, Nt, 4) OCR token bounding boxes
            v_conf: (B, Nv, 1) visual confidence
            t_conf: (B, Nt, 1) OCR confidence
            v_mask: (B, Nv) True=valid visual regions
            t_mask: (B, Nt) True=valid OCR tokens

        Returns: updated (v_emb, t_emb)
        """
        # --- V→T: T queries attend to V keys (ablation: can be disabled) ---
        if 'vt' in self.active_edge_types:
            msg_vt = self._attend(t_emb, v_emb, v_emb,
                                  t_boxes, v_boxes,
                                  v_mask, v_conf, 'vt')
        else:
            msg_vt = torch.zeros_like(t_emb)

        # --- T→V: V queries attend to T keys (ablation: can be disabled) ---
        if 'tv' in self.active_edge_types:
            msg_tv = self._attend(v_emb, t_emb, t_emb,
                                  v_boxes, t_boxes,
                                  t_mask, t_conf, 'tv')
        else:
            msg_tv = torch.zeros_like(v_emb)

        # --- T→T: T self-attention (ablation: can be disabled) ---
        if 'tt' in self.active_edge_types:
            msg_tt = self._attend(t_emb, t_emb, t_emb,
                                  t_boxes, t_boxes,
                                  t_mask, t_conf, 'tt')
        else:
            msg_tt = torch.zeros_like(t_emb)

        # --- Aggregate and update with residual + layer norm ---
        v_new = self.norm_v(v_emb + self.dropout(self.out_proj_v(msg_tv)))
        t_new = self.norm_t(t_emb + self.dropout(self.out_proj_t(msg_vt + msg_tt)))

        return v_new, t_new


# ---------------------------------------------------------------------------
# HSTFG Model
# ---------------------------------------------------------------------------

@META_ARCHITECTURE.register()
class HSTFG(nn.Module):
    """Heterogeneous Scene-Text Fusion Graph for Vietnamese Image Captioning."""

    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.device = config.DEVICE
        self.max_iter = vocab.max_answer_length

        self.mmt_config = BertConfig(
            hidden_size=config.MMT.HIDDEN_SIZE,
            num_hidden_layers=config.MMT.NUM_HIDDEN_LAYERS,
            num_attention_heads=config.MMT.NUM_ATTENTION_HEADS,
        )
        self.d_model = self.mmt_config.hidden_size

        # Graph config (with ablation flag defaults)
        graph_cfg = config.GRAPH
        self.num_gat_layers = graph_cfg.NUM_GAT_LAYERS
        self.spatial_hidden = graph_cfg.SPATIAL_HIDDEN
        self.use_spatial_bias = getattr(graph_cfg, 'USE_SPATIAL_BIAS', True)
        self.use_confidence_gate = getattr(graph_cfg, 'USE_CONFIDENCE_GATE', True)
        edge_types_cfg = getattr(graph_cfg, 'EDGE_TYPES', None)
        self.edge_types = list(edge_types_cfg) if edge_types_cfg is not None else ['vt', 'tv', 'tt']
        self.use_visual = getattr(graph_cfg, 'USE_VISUAL', True)

        self.build()

    def build(self):
        self._build_txt_encoding()
        self._build_node_embeddings()
        self._build_graph_layers()
        self._build_decoder()
        self._build_output()

    # --- Text encoding (same as M4C) ---
    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768
        self.text_bert_config = BertConfig(
            hidden_size=self.config.TEXT_BERT.HIDDEN_SIZE,
            num_hidden_layers=self.config.TEXT_BERT.NUM_HIDDEN_LAYERS,
            num_attention_heads=self.config.MMT.NUM_ATTENTION_HEADS,
        )
        if self.config.TEXT_BERT.LOAD_PRETRAINED:
            self.text_bert = TextBert.from_pretrained(
                self.config.TEXT_BERT.PRETRAINED_NAME, config=self.text_bert_config
            )
        else:
            self.text_bert = TextBert(self.text_bert_config)

        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            self.text_bert_out_linear = nn.Linear(
                self.config.TEXT_BERT.HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    # --- Node embedding layers (V and T only, no L nodes) ---
    def _build_node_embeddings(self):
        D = self.d_model

        # V nodes: region features (2048) + bbox (4)
        self.linear_obj_feat = nn.Linear(self.config.OBJECT_EMBEDDING.D_FEATURE, D)
        self.linear_obj_bbox = nn.Linear(4, D)
        self.obj_feat_norm = nn.LayerNorm(D)
        self.obj_bbox_norm = nn.LayerNorm(D)
        self.obj_drop = nn.Dropout(self.config.OBJECT_EMBEDDING.DROPOUT)

        # T nodes: OCR features (812) + bbox (4)
        self.linear_ocr_feat = nn.Linear(self.config.OCR_EMBEDDING.D_FEATURE, D)
        self.linear_ocr_bbox = nn.Linear(4, D)
        self.ocr_feat_norm = nn.LayerNorm(D)
        self.ocr_bbox_norm = nn.LayerNorm(D)
        self.ocr_drop = nn.Dropout(self.config.OCR_EMBEDDING.DROPOUT)

    # --- Spatial attention layers ---
    def _build_graph_layers(self):
        self.gat_layers = nn.ModuleList([
            SpatialHeteroAttentionLayer(
                self.d_model,
                num_heads=self.config.GRAPH.NUM_HEADS,
                spatial_hidden=self.spatial_hidden,
                dropout=self.config.GRAPH.DROPOUT,
                use_spatial_bias=self.use_spatial_bias,
                edge_types=self.edge_types,
                use_confidence_gate=self.use_confidence_gate,
            )
            for _ in range(self.num_gat_layers)
        ])

    # --- Decoder (reused from M4C) ---
    def _build_decoder(self):
        self.mmt = MMT(self.mmt_config)

    def _build_output(self):
        self.ocr_ptr_net = OcrPtrNet(
            hidden_size=self.config.OCR_PTR_NET.HIDDEN_SIZE,
            query_key_size=self.config.OCR_PTR_NET.QUERY_KEY_SIZE,
        )
        num_choices = len(self.vocab)
        self.classifier = nn.Linear(self.mmt_config.hidden_size, num_choices)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, items):
        fwd = {}
        self._forward_txt_encoding(items, fwd)
        self._forward_graph_encoding(items, fwd)
        self._forward_mmt_and_output(items, fwd)
        return {"scores": fwd["scores"]}

    def _forward_txt_encoding(self, items, fwd):
        fwd["txt_inds"] = items.question_tokens
        mask = generate_padding_mask(items.question_tokens, padding_idx=self.vocab.padding_idx)
        fwd["txt_mask"] = mask

        text_bert_out = self.text_bert(
            txt_inds=fwd["txt_inds"], txt_mask=fwd["txt_mask"]
        )
        fwd["txt_emb"] = self.text_bert_out_linear(text_bert_out)

    def _forward_graph_encoding(self, items, fwd):
        # --- Build V node embeddings ---
        obj_feat = items.region_features
        obj_bbox = items.region_boxes
        v_emb = self.obj_feat_norm(self.linear_obj_feat(obj_feat)) + \
                self.obj_bbox_norm(self.linear_obj_bbox(obj_bbox))
        v_emb = self.obj_drop(v_emb)

        # Ablation: zero out visual features for text-only variant
        if not self.use_visual:
            v_emb = torch.zeros_like(v_emb)

        obj_mask = generate_padding_mask(obj_feat, padding_idx=0)
        obj_mask_bool = (obj_mask.squeeze(1).squeeze(1) == 0)  # (B, Nv) True=valid

        # V confidence: fixed 1.0
        v_conf = torch.ones(v_emb.shape[0], v_emb.shape[1], 1, device=v_emb.device)

        # --- Build T node embeddings ---
        ocr_fasttext = F.normalize(items.ocr_fasttext_features, dim=-1)
        ocr_phoc = F.normalize(items.ocr_rec_features, dim=-1)
        ocr_fc = F.normalize(items.ocr_det_features, dim=-1)
        ocr_fasttext = ocr_fasttext[:, :ocr_phoc.size(1), :]
        ocr_feat = torch.cat([ocr_fasttext, ocr_phoc, ocr_fc], dim=-1)

        ocr_bbox = items.ocr_boxes
        t_emb = self.ocr_feat_norm(self.linear_ocr_feat(ocr_feat)) + \
                self.ocr_bbox_norm(self.linear_ocr_bbox(ocr_bbox))
        t_emb = self.ocr_drop(t_emb)

        ocr_mask = generate_padding_mask(ocr_feat, padding_idx=0)
        ocr_mask_bool = (ocr_mask.squeeze(1).squeeze(1) == 0)  # (B, Nt) True=valid

        # T confidence: from SwinTextSpotter scores — ensure (B, Nt, 1)
        t_conf = items.ocr_scores
        if t_conf.dim() == 2:
            t_conf = t_conf.unsqueeze(-1)
        elif t_conf.dim() > 3:
            t_conf = t_conf.view(t_conf.shape[0], t_conf.shape[1], -1)[:, :, :1]
        # t_conf is now (B, Nt, 1)

        # --- Run spatial attention layers (no loops, no adjacency construction) ---
        v_boxes = items.region_boxes
        t_boxes = ocr_bbox

        for layer in self.gat_layers:
            v_emb, t_emb = layer(
                v_emb, t_emb,
                v_boxes, t_boxes,
                v_conf, t_conf,
                obj_mask_bool, ocr_mask_bool,
            )

        # Store fused embeddings for decoder
        fwd["obj_mmt_in"] = v_emb
        fwd["obj_mask"] = obj_mask
        fwd["ocr_mmt_in"] = t_emb
        fwd["ocr_mask"] = ocr_mask

    # --- Decoder (same as M4C) ---

    def _forward_mmt(self, items, fwd):
        mmt_results = self.mmt(
            txt_emb=fwd["txt_emb"],
            txt_mask=fwd["txt_mask"],
            obj_emb=fwd["obj_mmt_in"],
            obj_mask=fwd["obj_mask"],
            ocr_emb=fwd["ocr_mmt_in"],
            ocr_mask=fwd["ocr_mask"],
            fixed_ans_emb=self.classifier.weight,
            prev_inds=fwd["prev_inds"],
        )
        fwd.update(mmt_results)

    def _forward_output(self, items, fwd):
        mmt_dec_output = fwd["mmt_dec_output"]
        mmt_ocr_output = fwd["mmt_ocr_output"]
        ocr_mask = fwd["ocr_mask"]

        fixed_scores = self.classifier(mmt_dec_output)
        dynamic_ocr_scores = self.ocr_ptr_net(mmt_dec_output, mmt_ocr_output, ocr_mask)
        fwd["scores"] = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)

    def _forward_mmt_and_output(self, items, fwd):
        if self.training:
            fwd["prev_inds"] = items.answer_tokens.clone()
            self._forward_mmt(items, fwd)
            self._forward_output(items, fwd)
        else:
            fwd["prev_inds"] = torch.zeros(
                (items.batch_size, self.max_iter), dtype=torch.long, device=self.device
            )
            fwd["prev_inds"][:, 0] = self.vocab.bos_idx

            last_ids = torch.zeros((items.batch_size,), device=self.device)
            for ith in range(self.max_iter):
                self._forward_mmt(items, fwd)
                self._forward_output(items, fwd)

                argmax_inds = fwd["scores"].argmax(dim=-1)
                fwd["prev_inds"][:, 1:] = argmax_inds[:, :-1]

                last_ids = torch.where(
                    last_ids.float() == self.vocab.eos_idx,
                    last_ids.float(),
                    argmax_inds[:, ith].float(),
                )
                if last_ids.mean() == self.vocab.eos_idx:
                    break
