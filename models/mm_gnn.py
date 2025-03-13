import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.weight_norm import weight_norm
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPreTrainedModel,
)
from models.utils import generate_padding_mask, generate_sequential_mask
from utils.logging_utils import setup_logger
from builders.model_builder import META_ARCHITECTURE


@META_ARCHITECTURE.register()
class MM_GNN_MODEL(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config

        self.vocab = vocab

        self.device = config.DEVICE
        self.max_iter = vocab.max_answer_length

        self.f_engineer = config.F_ENGINEER
        self.si_k_valve = config.SI_GNN.K_VALUE
        self.si_it = config.SI_GNN.ITERATION
        self.s_it = config.S_GNN.ITERATION
        self.si_penalty = config.SI_GNN.PENALTY
        self.s_penalty = config.S_GNN.PENALTY
        self.si_inter_dim = config.SI_GNN.INTER_DIM
        self.s_inter_dim = config.S_GNN.INTER_DIM
        self.K = config.SI_GNN.K

        self.bb_dim = config.BB_DIM
        self.fsd = config.FSD
        self.fvd = config.FVD

        self.dropout = config.TEXT_EMBEDDING.DROPOUT

        self.mmt_config = BertConfig(hidden_size=self.config.MMT.HIDDEN_SIZE,
                                     num_hidden_layers=self.config.MMT.NUM_HIDDEN_LAYERS,
                                     num_attention_heads=self.config.MMT.NUM_ATTENTION_HEADS) # Multimodal Transformer (answering module)
        self.d_model = self.mmt_config.hidden_size
        self.l_dim = 768
        self.build()

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        # self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_mmt_input()
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
            self.config.OBJECT_EMBEDDING.D_FEATURE, 1024
        )

        # object location feature: relative bounding box coordinates (4-dim)
        # self.linear_obj_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        self.obj_feat_layer_norm = nn.LayerNorm(1024)
        # self.obj_bbox_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
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

    def _build_mmt_input(self):
        self.linear_obj_mmt_in = nn.Linear(1024, self.mmt_config.hidden_size)
        self.linear_txt_emb_mmt_in = nn.Linear(600, self.mmt_config.hidden_size)
        self.linear_ocr_emb_mmt_in = nn.Linear(1200, self.mmt_config.hidden_size)

    def _build_model(self):
        self.si_gnn = SI_GNN(f_engineer=self.f_engineer,
                             bb_dim=self.bb_dim,
                             fvd=self.fvd,
                             fsd=self.fsd,
                             l_dim=self.l_dim,
                             inter_dim=self.si_inter_dim,
                             K=self.K,
                             dropout=self.dropout)
        self.s_gnn = S_GNN(self.f_engineer, self.bb_dim, 2 * self.fsd, self.l_dim, self.s_inter_dim, self.dropout)
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

        # fwd_results['txt_mask'] = fwd_results['txt_mask']
        # if len(fwd_results['txt_mask'].size()) == 4:
        #     fwd_results['txt_mask'] = fwd_results['txt_mask'].squeeze()

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
        # obj_bbox = items.region_boxes
        # obj_mmt_in = self.obj_feat_layer_norm(
        #     self.linear_obj_feat_to_mmt_in(obj_feat)
        # ) + self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(obj_bbox))
        obj_mmt_in = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_feat)
        )
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results["obj_mmt_in"] = obj_mmt_in

        # binary mask of valid object vs padding
        # mask = generate_padding_mask(
        #     obj_feat,
        #     padding_idx=0
        # ) == 0.
        # mask = mask.float()
        mask1 = generate_padding_mask(
            obj_feat,
            padding_idx=0
        )

        mask2 = generate_padding_mask(
            items.grid_features,
            padding_idx=0
        )

        mask = torch.cat([mask1, mask2], dim=-1)
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


    def _forward_gnn(self, items, fwd_results):
        s, i, si_adj, loss1 = self.si_gnn(l=fwd_results["txt_emb"],
                                          s=items['ocr_fasttext_features'],
                                          ps=items['ocr_boxes'],
                                          mask_s=(items['ocr_fasttext_features'] != 0).sum(dim=1).max(dim=1)[0].clamp_(min=1, max=100),
                                          v_ori=torch.cat([fwd_results["obj_mmt_in"], items['grid_features']], dim=1,),
                                          pv=torch.cat([items['region_boxes'].squeeze(), items['grid_boxes'].squeeze()], dim=1),
                                          mask_v=None,)
        fwd_results['obj_mmt_in'] = self.linear_obj_mmt_in(i)
        fwd_results['txt_emb'] = self.linear_txt_emb_mmt_in(s)
        fwd_results['txt_mask'] = fwd_results['ocr_mask']

        s, gnn_adj, loss2 = self.s_gnn(l=fwd_results["txt_emb"],
                                       s=s,
                                       ps=items['ocr_boxes'],
                                       mask_s=(items['ocr_fasttext_features'] != 0).sum(dim=1).max(dim=1)[0].clamp_(min=1, max=100))
        fwd_results['ocr_mmt_in'] = self.linear_ocr_emb_mmt_in(s)


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

        self._forward_gnn(items, fwd_results)
        for k, v in fwd_results.items():
            print(f"{k}: {v.shape}")
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


class SI_GNN(nn.Module):
    def __init__(self, f_engineer, bb_dim, fvd, fsd, l_dim, inter_dim, K, dropout):
        super(SI_GNN, self).__init__()
        self.f_engineer = f_engineer
        self.bb_dim = bb_dim
        self.fvd = fvd
        self.fsd = fsd
        self.l_dim = l_dim
        self.inter_dim = inter_dim
        self.K = K  # attention heads
        self.dropout = dropout

        self.bb_proj = LinearTransform(10, self.bb_dim)
        self.W1 = Parameter(torch.Tensor(self.K, self.bb_dim, self.inter_dim), requires_grad=True)
        self.W2 = Parameter(torch.Tensor(self.K, self.bb_dim, self.inter_dim), requires_grad=True)
        # self.fs_fa1 = LinearTransform(self.fsd + self.bb_dim, self.fvd - self.fsd)
        # self.fs_fa2 = LinearTransform(self.fsd + self.bb_dim, self.fvd - self.fsd)
        # self.fs_fa3 = LinearTransform(self.fvd + self.bb_dim, self.fvd + self.bb_dim)
        self.fs_fa4 = LinearTransform(self.fsd + self.bb_dim, self.fvd + self.bb_dim)
        # self.l_proj2 = LinearTransform(self.l_dim, self.fvd + self.bb_dim)
        # self.l_proj3 = LinearTransform(self.l_dim, self.fvd + self.bb_dim)
        # self.fv_fa1 = LinearTransform(self.fvd + self.bb_dim, self.fvd + self.bb_dim)
        self.fv_fa2 = LinearTransform(self.fvd + self.bb_dim, self.fvd + self.bb_dim)
        self.output_proj1 = ReLUWithWeightNormFC(self.bb_dim + self.fvd, self.fsd)
        self.output_proj2 = ReLUWithWeightNormFC(self.bb_dim + self.fvd, self.fvd)
        self.epsilon = Parameter(torch.Tensor(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W1)
        glorot(self.W2)
        nn.init.normal_(self.epsilon)

    def bb_process(self, bb):
        """
        :param bb: [B, num, 4], left, down, upper, right
        :return: [B, num(50 or 100), bb_dim]
        """
        bb_size = (bb[:, :, 2:] - bb[:, :, :2])  # 2
        bb_centre = bb[:, :, :2] + 0.5 * bb_size  # 2
        bb_area = (bb_size[:, :, 0] * bb_size[:, :, 1]).unsqueeze(2)  # 1
        bb_shape = (bb_size[:, :, 0] / (bb_size[:, :, 1] + 1e-14)).unsqueeze(2)  # 1
        return self.bb_proj(torch.cat([bb, bb_size, bb_centre, bb_area, bb_shape], dim=-1))

    def att_loss(self, adj, mask_s):
        """
        :param adj: [B, 50, 100]
        :param mask_s: [B]
        :return: loss induced from this attention, the more average, the more penalty
        """
        fro_norm = torch.sum(torch.norm(adj, dim=-1), dim=-1) / mask_s.to(torch.float)
        return -torch.sum(fro_norm)

    def forward(self, l, s, ps, mask_s, v_ori, pv, mask_v, k_valve=4, it=1, penalty_ratio=10):
        """
        # all below should be batched
        :param l: [2048], to guide edge strengths, by attention
        :param s: [50, 300]
        :param ps: [50, 4], same as above
        :param mask_s: int, <num_tokens> <= 50
        :param v_ori: [loc, vfd]
        :param pv: [100, 4]
        :param mask_v: [1]
        :param k_valve: the k in top_k, to control flow from v to s
        :param it: iterations for GNN
        :param penalty_ratio: the ratio need to be shrunk by penalty loss
        :return: updated i and s with identical shape
        """
        loc = v_ori.size(1) #mask_v[0]  # number of image features
        s_bb = self.bb_process(ps)  # [B, 50, bb_dim]
        v_bb = self.bb_process(pv)  # [B, 100, bb_dim]
        # s = torch.cat([s, s_bb], dim=2)  # [B, 50, bb_dim + fsd]
        v = torch.cat([v_ori, v_bb], dim=2)  # [B, 50, bb_dim + fvd]
        # l = l.unsqueeze(1)  # [B, 1, l_dim]

        inf_tmp = torch.ones(ps.size(0), ps.size(1), loc).to(l.device) * float('-inf')
        mask1 = (torch.arange(ps.size(1)).to(mask_s.device)[None, :] < mask_s[:, None]).unsqueeze(2).repeat(1, 1, loc)
        inf_tmp[mask1] = 0

        output_mask = (torch.arange(ps.size(1)).to(mask_s.device)[None, :] < mask_s[:, None]).unsqueeze(2).to(s.dtype)

        for _ in range(it):
            s_bb_formul = torch.matmul(s_bb.unsqueeze(1), self.W1.unsqueeze(0))  # [B, K, 50, inter_dim]
            v_bb_formul = torch.matmul(v_bb.unsqueeze(1), self.W2.unsqueeze(0))  # [B, K, 100, inter_dim]
            adj = torch.matmul(s_bb_formul, v_bb_formul.transpose(2, 3))  # [B, K, 50, 100]
            adj = torch.mean(adj, dim=1)  # [B, 50, 100]
            # index_mask = torch.topk(adj, loc - k_valve, dim=-1, largest=False, sorted=False)[-1]
            # adj.scatter_(-1, index_mask, float("-inf"))

            adj = F.softmax(adj, dim=2)  # [B, 50, 100]
            # adj = self.cooling(adj, temperature=0.25) * output_mask

            # prepared_s_source = self.output_proj2(
            #     self.fs_fa4(torch.cat([s, s_bb], dim=-1)) * self.l_proj3(l))  # [B, 50, fvd]
            prepared_s_source = self.output_proj2(self.fs_fa4(torch.cat([s, s_bb], dim=-1)))  # [B, 50, fvd]
            prepared_s_source = F.dropout(prepared_s_source, self.dropout)

            # prepared_v_source = self.output_proj1(self.fv_fa2(v) * F.softmax(self.l_proj2(l), dim=-1))  # [B, 100, fsd]
            prepared_v_source = self.output_proj1(self.fv_fa2(v))  # [B, 100, fsd]

            prepared_v_source = F.dropout(prepared_v_source, self.dropout)

            new_ele = torch.matmul(adj.transpose(1, 2), prepared_s_source)
            v = self.epsilon * new_ele + v_ori  # [B, loc, fvd]
            s = torch.cat([s, torch.matmul(adj, prepared_v_source)], dim=2)  # [B, 50, 2 * fsd]

        return s * output_mask, v, adj + inf_tmp, self.att_loss(adj, mask_s) / penalty_ratio

    def cooling(self, adj, temperature=0.5):
        """
        :param adj: [B, 50, 50], with adj value in 0 to 1, usually after softmax
        :return: cooled adj of the same shape
        """
        if self.training:
            adj = adj + (torch.randn(adj.shape) / 592).to(adj.device)
        adj = torch.pow(F.relu(adj), 1 / temperature)
        adj = adj / torch.sum(adj, dim=-1, keepdim=True)
        return adj


class S_GNN(nn.Module):
    def __init__(self, f_engineer, bb_dim, feature_dim, l_dim, inter_dim, dropout):
        super(S_GNN, self).__init__()
        self.f_engineer = f_engineer
        self.bb_dim = bb_dim
        self.feature_dim = feature_dim
        self.l_dim = l_dim
        self.inter_dim = inter_dim
        self.dropout = dropout

        self.bb_proj = ReLUWithWeightNormFC(10, self.bb_dim)
        self.fea_fa1 = ReLUWithWeightNormFC(self.bb_dim + self.feature_dim, self.bb_dim + self.feature_dim)
        self.fea_fa2 = ReLUWithWeightNormFC(self.bb_dim + self.feature_dim, self.bb_dim + self.feature_dim)
        self.fea_fa3 = ReLUWithWeightNormFC(2 * (self.bb_dim + self.feature_dim), 2 * (self.bb_dim + self.feature_dim))
        self.fea_fa4 = ReLUWithWeightNormFC(2 * (self.bb_dim + self.feature_dim), 2 * (self.bb_dim + self.feature_dim))
        self.fea_fa5 = ReLUWithWeightNormFC(2 * (self.bb_dim + self.feature_dim), 2 * (self.bb_dim + self.feature_dim))
        self.l_proj1 = ReLUWithWeightNormFC(self.l_dim, 2 * (self.bb_dim + self.feature_dim))
        self.l_proj2 = ReLUWithWeightNormFC(self.l_dim, 2 * (self.bb_dim + self.feature_dim))
        self.output_proj = ReLUWithWeightNormFC(2 * (self.bb_dim + self.feature_dim), self.feature_dim)

    def reset_parameters(self):
        pass

    def bb_process(self, bb):
        """
        :param bb: [B, num, 4], left, down, upper, right
        :return: [B, num(50 or 100), bb_dim]
        """
        bb_size = (bb[:, :, 2:] - bb[:, :, :2])  # 2
        bb_centre = bb[:, :, :2] + 0.5 * bb_size  # 2
        bb_area = (bb_size[:, :, 0] * bb_size[:, :, 1]).unsqueeze(2)  # 1
        bb_shape = (bb_size[:, :, 0] / (bb_size[:, :, 1] + 1e-14)).unsqueeze(2)  # 1
        return self.bb_proj(torch.cat([bb, bb_size, bb_centre, bb_area, bb_shape], dim=-1))

    def att_loss(self, adj, mask_s):
        """
        :param adj: [B, 50, 50]
        :param mask_s: [B]
        :return: loss induced from this attention, the more average, the more penalty
        """
        fro_norm = torch.sum(torch.norm(adj, dim=-1), dim=-1) / mask_s.to(torch.float)
        return -torch.sum(fro_norm)

    def forward(self, l, s, ps, mask_s, it=1, penalty_ratio=10):
        """
        # all below should be batched
        :param l: [2048], to guide edge strengths, by attention
        :param s: [50, 300]
        :param ps: [50, 4], same as above
        :param mask_s: int, <num_tokens> <= 50
        :param it: iterations for GNN
        :param penalty_ratio: need tobe shrunk by att loss
        :return: updated s with identical shape
        """
        bb = self.bb_process(ps)  # [B, 50, bb_dim]
        s_with_bb = torch.cat([s, bb], dim=2)  # [B, 50, bb_dim + feature_dim]
        # l = l.unsqueeze(1).repeat(1, 50, 1)  # [B,50, l_dim]

        inf_tmp = torch.ones(bb.size(0), ps.size(1), ps.size(1)).to(l.device) * float('-inf')
        mask1 = torch.max(torch.arange(ps.size(1))[None, :], torch.arange(ps.size(1))[:, None])
        mask1 = mask1[None, :, :].to(mask_s.device) < mask_s[:, None, None]
        mask2 = torch.arange(ps.size(1)).unsqueeze(1).expand(-1, ps.size(1)).to(mask_s.device)[None, :, :] >= mask_s[:, None, None]
        inf_tmp[mask1] = 0
        inf_tmp[mask2] = 0
        inf_tmp[torch.eye(ps.size(1)).bool().unsqueeze(0).repeat(bb.size(0), 1, 1)] = float('-inf')
        mask3 = mask_s == 1
        inf_tmp[:, 0, 0][mask3] = 0

        zero_tmp = torch.zeros(bb.size(0), ps.size(1), ps.size(1)).to(l.device)
        zero_tmp[mask1] = 1
        zero_tmp[mask2] = 1
        zero_tmp[torch.eye(ps.size(1)).bool().unsqueeze(0).repeat(bb.size(0), 1, 1)] = 0
        zero_tmp[:, 0, 0][mask3] = 1

        output_mask = (torch.arange(ps.size(1)).to(mask_s.device)[None, :] < mask_s[:, None]).unsqueeze(2).to(s.dtype)

        for _ in range(it):
            combined_fea = torch.cat(
                [s_with_bb, F.dropout(self.fea_fa1(s_with_bb) * self.fea_fa2(s_with_bb), self.dropout)],
                dim=2)  # [B, 50, 2*(bb_dim + feature_dim)]
            # l_masked_source = self.fea_fa3(combined_fea) * self.l_proj1(l)  # [B, 50, 2*(bb_dim + feature_dim)]
            l_masked_source = self.fea_fa3(combined_fea)

            l_masked_source = F.dropout(l_masked_source, self.dropout)
            fea_fa4 = F.dropout(self.fea_fa4(combined_fea), self.dropout)
            adj = torch.matmul(fea_fa4, l_masked_source.transpose(1, 2))  # [B, 50, 50]
            adj = F.softmax(adj + inf_tmp, dim=2)  # [B, 50, 50]
            adj = self.cooling(adj, temperature=0.1)  # [B, 50, 50]
            # prepared_source = self.fea_fa5(combined_fea) * F.softmax(self.l_proj2(l),
            #                                                          dim=-1)  # [B, 50, 2*(bb_dim + feature_dim)]
            prepared_source = self.fea_fa5(combined_fea)
            messages = self.output_proj(torch.matmul(adj, prepared_source))  # [B, 50, feature_dim]
            s = torch.cat([s, messages], dim=2)  # [B, 50, 2 * feature_dim]

        return s * output_mask, adj * output_mask, self.att_loss(adj * output_mask, mask_s) / penalty_ratio

    def cooling(self, adj, temperature=0.5):
        """
        :param adj: [B, 50, 50], with adj value in 0 to 1, usually after softmax
        :return: cooled adj of the same shape
        """
        if self.training:
            adj = adj + (torch.randn(adj.shape) / 100).to(adj.device)
        adj = torch.pow(adj, 1 / temperature)
        adj = adj / torch.sum(adj, dim=-1, keepdim=True)
        return adj


class ReLUWithWeightNormFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReLUWithWeightNormFC, self).__init__()

        layers = []
        layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LinearTransform(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearTransform, self).__init__()
        self.lc = weight_norm(
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=False), dim=None
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.lc(x)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


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

        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        dec_mask = torch.zeros(
            dec_emb.size(0), dec_emb.size(1), dtype=torch.float32, device=dec_emb.device
        ).unsqueeze(1).unsqueeze(2)
        encoder_inputs = torch.cat([txt_emb, obj_emb, ocr_emb, dec_emb], dim=1)
        attention_mask = torch.cat([txt_mask, obj_mask, ocr_mask, dec_mask], dim=-1)

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        dec_max_num = dec_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        ocr_begin = txt_max_num + obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(-1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
        # decoding step elements can attend to themselves in a causal manner
        # mask = generate_sequential_mask(dec_max_num) == 0
        # mask = mask.float()
        mask = generate_sequential_mask(dec_max_num)
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = mask

        # flip the mask, so that invalid attention pairs have -10000.
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        results = {
            "mmt_seq_output": mmt_seq_output,
            "mmt_txt_output": mmt_txt_output,
            "mmt_ocr_output": mmt_ocr_output,
            "mmt_dec_output": mmt_dec_output,
        }
        return results


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

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results