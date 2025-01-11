import torch
from torch import nn
import numpy as np
from builders.model_builder import META_ARCHITECTURE
import torch.nn.functional as F
import math


class MultiHeadedDotAttention(nn.Module):
    def __init__(self, num_heads, features_size, dropout=0.1):
        super(MultiHeadedDotAttention, self).__init__()
        self.d_model = features_size
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads

        # Create linear projections
        self.query_linear = nn.Linear(features_size, features_size)
        self.key_linear = nn.Linear(features_size, features_size)
        self.value_linear = nn.Linear(features_size, features_size)

        self.aoa_layer = nn.Sequential(
            nn.Linear(features_size * 2, features_size * 2),
            nn.GLU()
        )
        self.output_linear = nn.Linear(features_size, features_size)

        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, dropout, att_mask = None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if att_mask is not None:
            scores = scores.masked_fill(att_mask[:, None, None, :] == 1, float('-inf'))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value)

    def forward(self, query, key, value, use_aoa = False, att_mask = None):
        batch_size = query.size(0)

        query_ = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key_ = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value_ = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attended = self.attention(query_, key_, value_, self.dropout, att_mask)

        # Concat using view
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, -1, self.d_model)

        # Attention on Attention
        if use_aoa:
            aoa_output = self.aoa_layer(torch.cat([attended, query], dim = 2))
            output = self.output_linear(aoa_output)
        else:
            output = self.output_linear(attended)
        return output


class ResidualConnection(nn.Module):
    def __init__(self, _size, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(_size)

    def forward(self, x, att_features):
        return x + self.dropout(self.norm(att_features))


class AoA_Refiner_Layer(nn.Module):
    def __init__(self, features_size, num_heads, dropout=0.1):
        super(AoA_Refiner_Layer, self).__init__()
        self.attn = MultiHeadedDotAttention(num_heads, features_size)
        self.res_connection = ResidualConnection(features_size)

    def forward(self, x):
        att_features = self.attn(x, x, x, use_aoa=True)
        refined_features = self.res_connection(x, att_features)

        return refined_features


class AoA_Refiner_Core(nn.Module):
    def __init__(self, num_heads, stack_layers, features_size, out_size):
        super(AoA_Refiner_Core, self).__init__()

        self.layers = nn.ModuleList([AoA_Refiner_Layer(features_size, num_heads) for _ in range(stack_layers)])
        self.linear = nn.Linear(features_size, out_size)

        self.norm = nn.LayerNorm(out_size)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        x = self.linear(x)

        return self.norm(x)


class AoA_Decoder_Core(nn.Module):
    def __init__(self, embedding_layer, num_heads, features_size, embedding_size, vocab_size):
        super(AoA_Decoder_Core, self).__init__()
        self.feature_size = features_size
        self.out_dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embedding_size*2)

        self.resize_features = nn.Linear(features_size, embedding_size)

        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.att_lstm = nn.LSTM(embedding_size*2,
                                embedding_size,
                                num_layers=2)
        self.multi_head = MultiHeadedDotAttention(num_heads, embedding_size)

        self.aoa_layer = nn.Sequential(
            nn.Linear(features_size * 2, features_size * 2),
            nn.GLU()
        )

        self.residual_connect = ResidualConnection(features_size)
        self.out_linear = nn.Linear(features_size, vocab_size)

    def forward(self, features, captions_ids, captions_mask=None):
        batch_size = features.size(0)
        sequence_length = captions_ids.size(1)

        # Prepare Img Features
        features = self.resize_features(features) # batch_size, img_size, embedding_size
        features_ = torch.mean(features, dim=1).unsqueeze(dim=1).expand(batch_size, sequence_length, self.feature_size) # batch_size, sequence_length, embedding_size

        # Embedding Captions
        embedded_captions = self.embedding_layer(captions_ids) # batch_size, sequence_length, embedding_size

        # Prepare Inputs
        input_concat = self.norm(torch.cat([features_, embedded_captions], dim = 2)) # batch_size, sequence_length, embedding_size * 2

        # LSTM
        output, (h_att, c_att) = self.att_lstm(input_concat) # batch_size, sequence_length, embedding_size

        # Calculate Attention
        att = self.multi_head(output, features, features, use_aoa = False) # batch_size, sequence_length, embedding_size

        # Applying AoA
        ctx_input = torch.cat([att, output], dim=2) # batch_size, sequence_length, embedding_size * 2
        output_ = self.aoa_layer(ctx_input) # batch_size, sequence_length, embedding_size

        # Add Residual Connect
        residual_aoa = self.residual_connect(output_, output) # batch_size, sequence_length, embedding_size

        # Output
        return self.out_linear(self.out_dropout(residual_aoa)) # batch_size, sequence_length, vocab_size


@META_ARCHITECTURE.register()
class AoA_Model(nn.Module):
    def __init__(self, config, vocab):
        super(AoA_Model, self).__init__()
        self.vocab = vocab
        self.refiner_layer = AoA_Refiner_Core(config.REFINE_LAYER.NUM_HEADS,
                                              config.REFINE_LAYER.STACK_LAYERS,
                                              config.REFINE_LAYER.FEATURE_SIZE,
                                              config.REFINE_LAYER.OUT_SIZE)
        self.decoder_layer = AoA_Decoder_Core(config.DECODER.EMBEDDING_LAYERS,
                                              config.DECODER.NUM_HEADS,
                                              config.DECODER.FEATURE_SIZE,
                                              config.DECODER.EMBEDDING_SIZE,
                                              config.DECODER.VOCAB_SIZE)

        self.max_len = 50
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, sample):
        refined_features = self.refiner_layer(sample['region_features']) # batch_size, img_size, features_size
        print(refined_features.shape)
        if self.training:
            decoded_outputs = self.decoder_layer(refined_features, 
                                                 sample['answer_tokens'].type(torch.long).squeeze(), 
                                                 sample['answer_masks'].type(torch.long).squeeze())
        else:
            input_ids = torch.zeros_like(sample['answer_tokens'].squeeze(), dtype=torch.long)
            input_ids[:, 0] = 1
            decoded_outputs = self.decoder_layer(refined_features, 
                                                 input_ids, 
                                                 sample['answer_masks'].type(torch.long).squeeze())
            
        return decoded_outputs
    