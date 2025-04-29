import math

import torch
from torch import nn
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPreTrainedModel,
)

from utils.logging_utils import setup_logger
from builders.model_builder import META_ARCHITECTURE
from models.utils import generate_padding_mask, generate_sequential_mask

logger = setup_logger()

@META_ARCHITECTURE.register()
class VIWORD_MODEL(nn.Module):
    '''
        This is the original version of M4C method copied directly from https://github.com/ronghanghu/mmf
    '''
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.mmt_config = BertConfig(hidden_size=self.config.MMT.HIDDEN_SIZE,
                                        num_hidden_layers=self.config.MMT.NUM_HIDDEN_LAYERS,
                                        num_attention_heads=self.config.MMT.NUM_ATTENTION_HEADS)
        self.vocab = vocab
        self.d_model = self.mmt_config.hidden_size
        self.device = config.DEVICE
        self.max_iter = vocab.max_answer_length
        self.max_iter = 410
        print('vocab.max_answer_length', vocab.max_answer_length)

        self.build()

    def build(self):
        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_mmt()
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

    def _build_mmt(self):
        self.mmt = MMT(self.mmt_config, self.vocab)


    def _build_output(self):
        # fixed answer vocabulary scores

        self.mtp_heads = nn.ModuleList([
            OutputHead(self.vocab, self.config, self.mmt_config) 
            for _ in range(self.config.CLASSIFIER.N_FUTURE_TOKENS)
        ])


    def forward(self, items):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_txt_encoding(items, fwd_results)
        self._forward_obj_encoding(items, fwd_results)
        self._forward_ocr_encoding(items, fwd_results)
        self._forward_mmt_and_output(items, fwd_results)

        # only keep scores in the forward pass results
        # results = {"scores": fwd_results["scores"]}
        return fwd_results

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

    def _forward_mmt(self, items, fwd_results):
        # first forward the text BERT layers
        text_bert_out = self.text_bert(
            txt_inds=fwd_results["txt_inds"], txt_mask=fwd_results["txt_mask"]
        )
        fwd_results["txt_emb"] = self.text_bert_out_linear(text_bert_out)

        mmt_results = self.mmt(
            txt_emb=fwd_results["txt_emb"],
            txt_mask=fwd_results["txt_mask"],
            obj_emb=fwd_results["obj_mmt_in"],
            obj_mask=fwd_results["obj_mask"],
            ocr_emb=fwd_results["ocr_mmt_in"],
            ocr_mask=fwd_results["ocr_mask"],
            prev_inds=fwd_results["prev_inds"],
        )
        fwd_results.update(mmt_results)
        
        

    def _forward_output(self, items, fwd_results):
        mmt_dec_output = fwd_results["mmt_dec_output"]
        batch_size = mmt_dec_output.size(0)
        
        if len(self.mtp_heads) == 1:
            preds = self.mtp_heads[0](mmt_dec_output)
        else:
            if self.training:
                # Shape: [n_future_tokens, batch_size, seq_len, 4, vocab_size]
                preds = torch.stack([
                    classifier_head(mmt_dec_output)
                    for classifier_head in self.mtp_heads
                ], dim=0)
            else:
                # Shape: [batch_size, seq_len, 4, vocab_size]
                preds = self.mtp_heads[0](mmt_dec_output)
        
        fwd_results["scores"] = preds
    
    def _forward_mmt_and_output(self, items, fwd_results):
        if self.training:
            answer_tokens = items.answer_tokens.clone()
            fwd_results["prev_inds"] = torch.stack([
                F.pad(answer_tokens[i], (0, 0, 0, self.max_iter - answer_tokens.shape[1])) 
                for i in range(answer_tokens.size(0))
            ])
            self._forward_mmt(items, fwd_results)
            
            self._forward_output(items, fwd_results)
        else:
            active_mask = torch.ones(items.batch_size, dtype=torch.bool).to(self.device)
            # fill prev_inds with bos_idx at index 0, and zeros elsewhere
            fwd_results["prev_inds"] = torch.zeros((items.batch_size, self.max_iter, 4)).long().to(self.device)
            fwd_results['prev_inds'][:, 0, 0]  = self.vocab.bos_idx

            # greedy decoding at test time
            last_ids = torch.zeros((items.batch_size, 4)).to(self.device)
            for ith in range(self.max_iter):
                if not active_mask.any():
                    break  # All sequences finished
                self._forward_mmt(items, fwd_results)
                self._forward_output(items, fwd_results)

                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                argmax_inds = fwd_results["scores"].argmax(dim=-1)
                fwd_results["prev_inds"][:, 1:] = argmax_inds[:, :-1]

                # whether or not to interrupt the decoding process
                last_ids = torch.where(last_ids.float() == self.vocab.eos_idx, last_ids.float(), argmax_inds[:, ith].float())
                newly_finished = (last_ids[:, 0] == self.vocab.eos_idx)
                # print(newly_finished.shape)
                # print(active_mask.shape)
                active_mask[active_mask.clone()] = ~newly_finished[active_mask]

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


class MMT(BertPreTrainedModel):
    def __init__(self, config, vocab):
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config, vocab)
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
        prev_inds,
    ):

        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        dec_emb = self.prev_pred_embeddings(prev_inds)

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
        obj_begin = txt_end
        obj_end = obj_begin + obj_max_num
        ocr_begin = obj_end
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
        mmt_obj_output = mmt_seq_output[:, obj_begin:obj_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        results = {
            "mmt_seq_output": mmt_seq_output,
            "mmt_txt_output": mmt_txt_output,
            "mmt_ocr_output": mmt_ocr_output,
            "mmt_dec_output": mmt_dec_output,
        }
        return results

class WordRepresentation(nn.Module):
    def __init__(self, vocab, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), hidden_size)
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        embed = self.embedding(x) # (batch_size, num_tokens, 4, hidden_size)
        batch_size, num_tokens, seq_len, feature_dim = embed.size()
        embed = embed.view(batch_size*num_tokens, seq_len, feature_dim) # (batch_size*num_tokens, 4, hidden_size)
        out, h_n = self.gru(embed)
        word_representations = h_n.squeeze(0)
        word_representations = word_representations.view(batch_size, num_tokens, self.hidden_size)

        return word_representations

class PrevPredEmbeddings(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        MAX_DEC_LENGTH = 410
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.caption_embeddings = WordRepresentation(vocab, hidden_size)
        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)

        self.emb_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, prev_inds):

        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)

        token_emb = self.caption_embeddings(prev_inds.type(torch.long))

        position_ids = torch.arange(seq_length, dtype=torch.long, device=prev_inds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = token_emb + position_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        return embeddings


class OutputHead(nn.Module):
    def __init__(self, vocab, config, mmt_config):
        super().__init__()
        self.dense = nn.Linear(mmt_config.hidden_size, mmt_config.hidden_size)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(mmt_config.hidden_size)
        num_choices = len(vocab)

        self.classifier_head = nn.ModuleList([
            nn.Linear(mmt_config.hidden_size, len(vocab))
            for _ in range(4)
        ])

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        x = self.norm(x)
        preds = torch.stack([
                linear(x)
                for linear in self.classifier_head
            ], dim=2)
        return preds
