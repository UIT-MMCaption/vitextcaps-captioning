import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPreTrainedModel,
)
from models.utils import generate_padding_mask, generate_sequential_mask
from utils.logging_utils import setup_logger
from builders.model_builder import META_ARCHITECTURE

logger = setup_logger()

@META_ARCHITECTURE.register()
class CRN_MODEL(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config

        self.vocab = vocab

        self.device = config.DEVICE
        self.max_iter = vocab.max_answer_length

        self.PAM_config = BertConfig(hidden_size=self.config.PAM.HIDDEN_SIZE,
                                     num_hidden_layers=self.config.PAM.NUM_HIDDEN_LAYERS)  # Progressive Attention Module

        self.MRG_config = BertConfig(hidden_size=self.config.MRG.HIDDEN_SIZE) # Multimodal Reasoning Graph

        self.mmt_config = BertConfig(hidden_size=self.config.MMT.HIDDEN_SIZE,
                                     num_hidden_layers=self.config.MMT.NUM_HIDDEN_LAYERS,
                                     num_attention_heads=self.config.MMT.NUM_ATTENTION_HEADS) # Multimodal Transformer (answering module)
        self.d_model = self.mmt_config.hidden_size

        self.build()

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        # self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_model()
        self._build_output()

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768

        self.text_bert_config = BertConfig(hidden_size=self.config.TEXT_BERT.HIDDEN_SIZE,
                                            num_hidden_layers=self.config.TEXT_BERT.NUM_HIDDEN_LAYERS,
                                            num_attention_heads=self.config.MMT.NUM_ATTENTION_HEADS)
        if self.config.TEXT_BERT.LOAD_PRETRAINED:
            self.text_bert = TextBert.from_pretrained(
                self.config.TEXT_BERT.PRETRAINED_NAME, config=self.text_bert_config
            )
        else:
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            logger.info(
                f"Projecting text_bert output to {self.mmt_config.hidden_size} dim"
            )

            self.text_bert_out_linear = nn.Linear(
                self.config.TEXT_BERT.HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    def _build_obj_encoding(self):
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.OBJECT_EMBEDDING.D_FEATURE, self.mmt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        self.obj_feat_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.obj_bbox_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.OBJECT_EMBEDDING.DROPOUT)

    def _build_ocr_encoding(self):
        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.OCR_EMBEDDING.D_FEATURE, self.mmt_config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        # OCR word embedding features
        # self.ocr_word_embedding = build_word_embedding(self.config.OCR_TEXT_EMBEDDING)

        self.ocr_feat_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_text_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.OCR_EMBEDDING.DROPOUT)

    def _build_model(self):
        self.PAM_1 = Q(self.PAM_config)
        self.PAM_2 = QT(self.PAM_config)
        self.PAM_3 = QTV(self.PAM_config)
        self.MRG = MRG_Graph(self.MRG_config)
        self.mmt = MMT(self.mmt_config)

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(hidden_size=self.config.OCR_PTR_NET.HIDDEN_SIZE,
                                     query_key_size=self.config.OCR_PTR_NET.QUERY_KEY_SIZE)

        # fixed answer vocabulary scores
        num_choices = len(self.vocab)
        # remove the OCR copying dimensions in LoRRA's classifier output
        # (OCR copying will be handled separately)
        self.classifier = nn.Linear(self.mmt_config.hidden_size, num_choices)


    def forward(self, items):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_txt_encoding(items, fwd_results)
        self._forward_obj_encoding(items, fwd_results)
        self._forward_ocr_encoding(items, fwd_results)

        if len(fwd_results['txt_mask'].size()) == 4:
            fwd_results['txt_mask'] = fwd_results['txt_mask'].squeeze()
        self._forward_mmt_and_output(items, fwd_results)

        # only keep scores in the forward pass results

        results = {"scores": fwd_results["scores"]}
        return results

    def _forward_txt_encoding(self, items, fwd_results):
        fwd_results["txt_inds"] = items.question_tokens

        # binary mask of valid text (question words) vs padding
        # mask = generate_padding_mask(
        #     items.question_tokens,
        #     padding_idx=self.vocab.padding_idx
        # ) == 0.
        # mask = mask.float()
        mask = generate_padding_mask(
            items.question_tokens,
            padding_idx=self.vocab.padding_idx
        )
        fwd_results["txt_mask"] = mask
        text_bert_out = self.text_bert(
            txt_inds=fwd_results["txt_inds"], txt_mask=fwd_results["txt_mask"]
        )
        fwd_results["txt_emb"] = self.text_bert_out_linear(text_bert_out)

    def _forward_obj_encoding(self, items, fwd_results):
        # object appearance feature
        obj_feat = items.region_features
        obj_bbox = items.region_boxes
        obj_mmt_in = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_feat)
        ) + self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(obj_bbox))
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results["obj_mmt_in"] = obj_mmt_in

        # binary mask of valid object vs padding
        # mask = generate_padding_mask(
        #     obj_feat,
        #     padding_idx=0
        # ) == 0.
        # mask = mask.float()
        mask = generate_padding_mask(
            obj_feat,
            padding_idx=0
        )
        fwd_results["obj_mask"] = mask

    def _forward_ocr_encoding(self, items, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = items.ocr_fasttext_features
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR rec feature (256-dim), replace the OCR PHOC features, extracted from swintextspotter
        ocr_phoc = items.ocr_rec_features
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 256

        # OCR appearance feature, extracted from swintextspotter
        ocr_fc = items.ocr_det_features
        ocr_fc = F.normalize(ocr_fc, dim=-1)

        assert ocr_fc.size(1) == ocr_phoc.size(1), "Second dimensions must match!"

        ocr_fasttext = ocr_fasttext[:, :ocr_phoc.size(1), :]
        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_fc], dim=-1
        )
        ocr_bbox = items.ocr_boxes
        ocr_mmt_in = self.ocr_feat_layer_norm(
            self.linear_ocr_feat_to_mmt_in(ocr_feat)
        ) + self.ocr_bbox_layer_norm(self.linear_ocr_bbox_to_mmt_in(ocr_bbox))
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results["ocr_mmt_in"] = ocr_mmt_in

        # binary mask of valid OCR vs padding
        # mask = generate_padding_mask(
        #     ocr_feat,
        #     padding_idx=0
        # ) == 0.
        # mask = mask.float()
        mask = generate_padding_mask(
            ocr_feat,
            padding_idx=0
        )
        fwd_results["ocr_mask"] = mask


    def _forward_pam_graph(self, items, fwd_results):
        
        # self.PAM_1(fwd_results)
        
        self.PAM_2(fwd_results)
        
        self.PAM_3(fwd_results)

        self.MRG(items, fwd_results)

    def _forward_mmt(self, items, fwd_results):

        mmt_results = self.mmt(
            txt_emb=fwd_results["txt_emb"],
            txt_mask=fwd_results["txt_mask"],
            obj_emb=fwd_results["obj_mmt_in"],
            obj_mask=fwd_results["obj_mask"],
            ocr_emb=fwd_results["ocr_mmt_in"],
            ocr_mask=fwd_results["ocr_mask"],
            fixed_ans_emb=self.classifier.weight,
            prev_inds=fwd_results["prev_inds"],
        )
        fwd_results.update(mmt_results)

    def _forward_output(self, items, fwd_results):
        mmt_dec_output = fwd_results["mmt_dec_output"]
        mmt_ocr_output = fwd_results["mmt_ocr_output"]

        ocr_mask = fwd_results["ocr_mask"]

        fixed_scores = self.classifier(mmt_dec_output)
        dynamic_ocr_scores = self.ocr_ptr_net(mmt_dec_output, mmt_ocr_output, ocr_mask)

        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)

        fwd_results["scores"] = scores

    def _forward_mmt_and_output(self, items, fwd_results):
        self._forward_pam_graph(items, fwd_results)

        if self.training:
            fwd_results['prev_inds'] = items.answer_tokens.clone()
            self._forward_mmt(items, fwd_results)
            self._forward_output(items, fwd_results)
        else:
            # self.train()
            dec_step_num = items.answer_tokens.size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            fwd_results['prev_inds'] = torch.zeros_like(
                items.answer_tokens
            )
            fwd_results['prev_inds'][:, 0] = self.vocab.bos_idx

            # greedy decoding at test time
            for t in range(dec_step_num):
                self._forward_mmt(items, fwd_results)
                self._forward_output(items, fwd_results)

                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                argmax_inds = fwd_results["scores"].argmax(dim=-1)
                fwd_results['prev_inds'][:, 1:] = argmax_inds[:, :-1]
            # self.eval()


class Q(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, fwd_results):
        txt_emb = fwd_results['txt_emb']
        txt_mask = fwd_results['txt_mask']
        encoder_inputs = txt_emb
        attention_mask = txt_mask

        txt_max_num = txt_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        # flip the mask, so that invalid attention pairs have -10000.
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        # mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        fwd_results['txt_emb'] = fwd_results['txt_emb'] + torch.tanh(mmt_seq_output)


class QT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, fwd_results):
        txt_emb = fwd_results['txt_emb'].squeeze()
        txt_mask = fwd_results['txt_mask'].squeeze()
        obj_emb = fwd_results['ocr_mmt_in'].squeeze()
        obj_mask = fwd_results['ocr_mask'].squeeze()

        # Correct the shape
        if len(txt_emb.size()) == 2:
            txt_emb = txt_emb.unsqueeze(0)
        if len(txt_mask.size()) == 1:
            txt_mask = txt_mask.unsqueeze(0)
        if len(obj_emb.size()) == 2:
            obj_emb = obj_emb.unsqueeze(0)
        if len(obj_mask.size()) == 1:
            obj_mask = obj_mask.unsqueeze(0)
        
        encoder_inputs = torch.cat(
            [txt_emb, obj_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [txt_mask, obj_mask],
            dim=1
        )

        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        # flip the mask, so that invalid attention pairs have -10000.
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        fwd_results['txt_emb'] = fwd_results['txt_emb'] + torch.tanh(mmt_seq_output[:, txt_begin:txt_end])
        fwd_results['ocr_mmt_in'] = fwd_results['ocr_mmt_in'] + torch.tanh(mmt_seq_output[:, txt_end:])


class QTV(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, fwd_results):
        txt_emb = fwd_results['txt_emb'].squeeze()
        txt_mask = fwd_results['txt_mask'].squeeze()
        obj_emb = fwd_results['obj_mmt_in'].squeeze()
        obj_mask = fwd_results['obj_mask'].squeeze()
        ocr_emb = fwd_results['ocr_mmt_in'].squeeze()
        ocr_mask = fwd_results['ocr_mask'].squeeze()

        # Correct the shape
        if len(txt_emb.size()) == 2:
            txt_emb = txt_emb.unsqueeze(0)
        if len(txt_mask.size()) == 1:
            txt_mask = txt_mask.unsqueeze(0)

        if len(obj_emb.size()) == 2:
            obj_emb = obj_emb.unsqueeze(0)
        if len(obj_mask.size()) == 1:
            obj_mask = obj_mask.unsqueeze(0)
        
        if len(ocr_emb.size()) == 2:
            ocr_emb = ocr_emb.unsqueeze(0)
        if len(ocr_mask.size()) == 1:
            ocr_mask = ocr_mask.unsqueeze(0)

        encoder_inputs = torch.cat(
            [txt_emb, obj_emb, ocr_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [txt_mask, obj_mask, ocr_mask],
            dim=1
        )

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        ocr_begin = txt_max_num + obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        # flip the mask, so that invalid attention pairs have -10000.
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        fwd_results['txt_emb'] = fwd_results['txt_emb'] + torch.tanh(mmt_seq_output[:, txt_begin:txt_end])
        fwd_results['obj_mmt_in'] = fwd_results['obj_mmt_in'] + torch.tanh(mmt_seq_output[:, txt_end:ocr_begin])
        fwd_results['ocr_mmt_in'] = fwd_results['ocr_mmt_in'] + torch.tanh(mmt_seq_output[:, ocr_begin:ocr_end])


class MRG_Graph(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Only build visual-text interaction layer
        self._build_common_layer(module_name='vt', hidden_size=config.hidden_size)
        
    def _build_common_layer(self, module_name, hidden_size, edge_dim=5):
        # Removed question-related attention
        transform_edge = nn.Sequential(
            nn.Linear(edge_dim, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        embeded = nn.Linear(hidden_size, hidden_size)
        edge_attn = nn.Linear(hidden_size, 1)
        
        setattr(self, '{}_transform_edge'.format(module_name), transform_edge)
        setattr(self, '{}_embeded'.format(module_name), embeded)
        setattr(self, '{}_edge_attn'.format(module_name), edge_attn)
        
        # Feature processing layers
        feat_layer_1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        feat_layer_2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        feat_layer_3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        input_drop = nn.Dropout(0.1)
        
        setattr(self, '{}_feat_layer_1'.format(module_name), feat_layer_1)
        setattr(self, '{}_feat_layer_2'.format(module_name), feat_layer_2)
        setattr(self, '{}_feat_layer_3'.format(module_name), feat_layer_3)
        setattr(self, '{}_drop'.format(module_name), input_drop)

    def _build_compute_graph(self, edge_feat, visual_context, input_mask, module_name):
        batch, num_obj, num_subobj = edge_feat.size()[:3]
        
        # Transform edge features
        edge_feat = getattr(self, module_name+'_transform_edge')(edge_feat)
        
        # Use visual context instead of question features
        visual_context_emb = getattr(self, module_name+'_embeded')(visual_context)
        visual_context_emb = visual_context_emb.unsqueeze(1).expand(-1, num_obj, num_subobj, -1)
        
        # Compute attention weights
        edge_attn = getattr(self, module_name+'_edge_attn')(
            torch.tanh(visual_context_emb + edge_feat)
        ).squeeze(-1)
        A_edge_attn = F.softmax(edge_attn, -1)
        
        # Apply mask
        target_size = edge_feat.size(1)
        s = input_mask.size(-1)
        input_mask = torch.nn.functional.pad(input_mask, (0, target_size-s), value=0)
        
        A_edge_attn = A_edge_attn * input_mask.unsqueeze(-1)
        A_edge_attn = A_edge_attn / (A_edge_attn.sum(dim=-1, keepdim=True) + 1e-12)
        
        # Update edge features
        updated_edge_feat = edge_feat * A_edge_attn.unsqueeze(-1)
        updated_edge_feat = updated_edge_feat.sum(2)
        
        return A_edge_attn, updated_edge_feat, input_mask.squeeze()

    def forward(self, item, fwd_results):
        v_feat = fwd_results['obj_mmt_in']
        t_feat = fwd_results['ocr_mmt_in']
        v2t_edge = item['obj_ocr_edge_feat']
        t2v_edge = item['ocr_obj_edge_feat']
        
        v_mask = fwd_results['obj_mask'].squeeze()
        t_mask = fwd_results['ocr_mask'].squeeze()

        if len(v_mask.size()) == 1:
            v_mask = v_mask.unsqueeze(0)
            t_mask = t_mask.unsqueeze(0)
            
        v_mask = (v_mask == 0).float()
        t_mask = (t_mask == 0).float()

        # Use mean pooled visual features as context instead of question
        visual_context = v_feat.mean(dim=1, keepdim=True)
        
        # Compute visual-text interactions
        v2t_attn, v2t_feat, v_mask = self._build_compute_graph(
            v2t_edge, visual_context, v_mask, module_name='vt'
        )
        t2v_attn, t2v_feat, t_mask = self._build_compute_graph(
            t2v_edge, visual_context, t_mask, module_name='vt'
        )
        if len(v_mask.size()) == 1:
            v_mask = v_mask.unsqueeze(0)
            t_mask = t_mask.unsqueeze(0)
        # Compute masks for interactions
        v2t_mask = torch.bmm(v_mask.unsqueeze(-1), t_mask.unsqueeze(1))
        t2v_mask = v2t_mask.transpose(1, 2)
        v2t_attn = v2t_attn * v2t_mask
        t2v_attn = t2v_attn * t2v_mask
        
        # Pad features if necessary
        v_s = v_feat.size(1)
        v_feat = torch.nn.functional.pad(v_feat, (0, 0, 0, 100-v_s), value=0)
        t_s = t_feat.size(1)
        t_feat = torch.nn.functional.pad(t_feat, (0, 0, 0, 50-t_s), value=0)
        
        # Update features through cross-attention
        new_t_feat = torch.bmm(v2t_attn.transpose(1, 2), v_feat)
        new_v_feat = torch.bmm(t2v_attn.transpose(1, 2), t_feat)
        
        # Final feature update
        v_feat = self.vt_feat_layer_1(v_feat) + self.vt_feat_layer_2(new_v_feat) + self.vt_feat_layer_3(v2t_feat)
        t_feat = self.vt_feat_layer_1(t_feat) + self.vt_feat_layer_2(new_t_feat) + self.vt_feat_layer_3(t2v_feat)
        
        # Update forward results
        fwd_results['obj_mmt_in'] = self.vt_drop(v_feat)
        fwd_results['ocr_mmt_in'] = self.vt_drop(t_feat)
        
        return fwd_results



def pad_or_truncate_embedding(embedding, target_length, pad_value=0):
    """
    Pad or truncate embedding tensor to target length along sequence dimension
    
    Args:
        embedding (torch.Tensor): Input embedding of shape [batch_size, seq_len, hidden_dim]
        target_length (int): Desired sequence length
        pad_value (float): Value to use for padding
    
    Returns:
        torch.Tensor: Padded/truncated embedding of shape [batch_size, target_length, hidden_dim]
    """
    batch_size, curr_length, hidden_dim = embedding.shape
    
    if curr_length > target_length:
        # Truncate
        return embedding[:, :target_length, :]
    elif curr_length < target_length:
        # Pad
        padding = torch.full((batch_size, target_length - curr_length, hidden_dim),
                              pad_value,
                              dtype=embedding.dtype,
                              device=embedding.device)
        return torch.cat([embedding, padding], dim=1)
    return embedding


class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(
        self,
        txt_emb,
        txt_mask,
        obj_emb,
        obj_mask,
        ocr_emb,
        ocr_mask,
        fixed_ans_emb,
        prev_inds,
    ):
        
        # Get target lengths from masks
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        
        # Pad or truncate embeddings to match mask lengths
        txt_emb = pad_or_truncate_embedding(txt_emb, txt_max_num)
        obj_emb = pad_or_truncate_embedding(obj_emb, obj_max_num)
        ocr_emb = pad_or_truncate_embedding(ocr_emb, ocr_max_num)

        if len(txt_mask.size()) == 1:
            txt_mask = txt_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        if len(txt_mask.size()) == 2:
            txt_mask = txt_mask.unsqueeze(1).unsqueeze(1)
        # Get decoder embeddings
        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)
        
        # Create decoder mask
        dec_mask = torch.zeros(
            dec_emb.size(0), dec_emb.size(1), dtype=torch.float32, device=dec_emb.device
        ).unsqueeze(1).unsqueeze(2)
        
        # Concatenate embeddings and masks
        encoder_inputs = torch.cat([txt_emb, obj_emb, ocr_emb, dec_emb], dim=1)
        attention_mask = torch.cat([txt_mask, obj_mask, ocr_mask, dec_mask], dim=-1)

        # Calculate offsets
        dec_max_num = dec_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        ocr_begin = txt_max_num + obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        # Create extended attention mask
        to_seq_length = attention_mask.size(-1)
        from_seq_length = to_seq_length
        extended_attention_mask = attention_mask.repeat(1, 1, from_seq_length, 1)
        
        # Add sequential mask for decoder
        mask = generate_sequential_mask(dec_max_num)
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = mask

        # Forward pass through encoder
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers
        
        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )

        # Extract outputs
        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        return {
            "mmt_seq_output": mmt_seq_output,
            "mmt_txt_output": mmt_txt_output,
            "mmt_ocr_output": mmt_ocr_output,
            "mmt_dec_output": mmt_dec_output,
        }

class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = attention_mask.squeeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores

class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output

class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 410
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, ocr_emb, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2

        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)
        ans_num = ans_emb.size(0)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)
        assert ans_emb.size(-1) == ocr_emb.size(-1)
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)

        # Add position and type embedding for previous predictions
        position_ids = torch.arange(seq_length, dtype=torch.long, device=ocr_emb.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings

        return dec_emb


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size * length, dim)

    # Move batch_offsets to the same device as inds
    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results