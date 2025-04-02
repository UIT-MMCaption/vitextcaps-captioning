import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from builders.model_builder import META_ARCHITECTURE
from utils.instance import InstanceList

### This model is based on the MMF framework and BUTD model for original version.

class LanguageDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout, fc_bias_init, **kwargs):
        super().__init__()
        self.language_lstm = nn.LSTMCell(in_dim + hidden_dim, hidden_dim, bias=True)
        self.fc = weight_norm(nn.Linear(hidden_dim, out_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights(fc_bias_init)

    def init_weights(self, fc_bias_init):
        self.fc.bias.data.fill_(fc_bias_init)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, weighted_attn, state):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        weighted_attn = weighted_attn.to(device)
        h1, c1 = state["td_hidden"]
        h2, c2 = state["lm_hidden"]
        h1, c1 = h1.to(device), c1.to(device)
        h2, c2 = h2.to(device), c2.to(device)
        
        
        cat_input = torch.cat([weighted_attn, h1], dim=1).to(device)
        h2, c2 = self.language_lstm(cat_input, (h2, c2))
        predictions = self.fc(self.dropout(h2))
        state["lm_hidden"] = (h2, c2)
        return predictions, state


class ClassifierLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.module = LanguageDecoder(
            in_dim,
            out_dim,
            hidden_dim=kwargs["hidden_dim"],
            dropout=kwargs["dropout"],
            fc_bias_init=kwargs["fc_bias_init"]
        )

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

@META_ARCHITECTURE.register()
class MMF_BUTD(nn.Module):
    '''
        Reimplementation of BUTD method.
    '''
    def __init__(self, config, vocab):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  
        self.vocab = vocab
        self.max_len = vocab.max_answer_length
        self.vocab_size = len(vocab)
        self.eos_idx = vocab.eos_idx
        self.hidden_dim = config.classifier.params.hidden_dim
        self.d_model = 2048

        self.build(config)

    def build(self, config):
        self._build_word_embedding(config)
        self._init_classifier(config)
    
    def _build_word_embedding(self, config):
        self.word_embedding = nn.Embedding(self.vocab_size, self.d_model).to(self.device)
        self.text_embeddings_out_dim = self.d_model

    def _init_classifier(self, config):
        self.classifier = ClassifierLayer(
            in_dim=config["classifier"]["params"]["feature_dim"],
            out_dim=self.vocab_size,
            **config["classifier"]["params"]
        ).to(self.device)

    def init_hidden_state(self, features):
        h = torch.zeros((features.size(0), self.hidden_dim), dtype=torch.float, device=self.device) # (bs, hidden_dim)
        c = torch.zeros((features.size(0), self.hidden_dim), dtype=torch.float, device=self.device) # (bs, hidden_dim)
        return h, c

    def get_data_t(self, t, data, batch_size_t, prev_output):
        batch_size_t = sum([l > t for l in data["decode_lengths"]])
        data["texts"] = data["texts"][:batch_size_t].to(self.device)  # (bs, max_len)
        if "state" in data:
            h1 = data["state"]["td_hidden"][0][:batch_size_t].to(self.device)  # (bs, hidden_dim)
            c1 = data["state"]["td_hidden"][1][:batch_size_t].to(self.device)  # (bs, hidden_dim)
            h2 = data["state"]["lm_hidden"][0][:batch_size_t].to(self.device)  # (bs, hidden_dim)
            c2 = data["state"]["lm_hidden"][1][:batch_size_t].to(self.device)  # (bs, hidden_dim)
        else:
            h1, c1 = self.init_hidden_state(data["texts"]) # attention feature for img_f & lm_f
            h2, c2 = self.init_hidden_state(data["texts"]) # top_down + embed => predictions for next
        data["state"] = {"td_hidden": (h1, c1), "lm_hidden": (h2, c2)}
        return data, batch_size_t

    def prepare_data(self, sample_list, batch_size):
        self.teacher_forcing = "answer_tokens" in sample_list
        data = {}
        lengths = (torch.tensor(sample_list["answer_tokens"]) != 0).sum(dim=1)
        data["decode_lengths"] = (lengths - 1).tolist()  # Bỏ token <SOS>
        data["texts"] = torch.tensor(sample_list["answer_tokens"]).to(self.device)  # (bs, max_len)
        timesteps = max(data["decode_lengths"])
        sample_list["targets"] = torch.tensor(sample_list["answer_tokens"]).to(self.device)[:, 1:]  # (bs, max_len-1)
        return data, sample_list, timesteps

    def process_feature_embedding(self, sample_list, embedding, batch_size_t):
        image_features = sample_list["region_features"][:batch_size_t].to(self.device)  # (batch_size_t, num_features, feature_dim)
        scores = torch.bmm(image_features, embedding.squeeze(1).unsqueeze(2)).squeeze(2)  # (batch_size_t, num_features)
        attn_weights = F.softmax(scores, dim=1)  # (batch_size_t, num_features)
        attention_feature = torch.bmm(attn_weights.unsqueeze(1), image_features).squeeze(1)  # (batch_size_t, feature_dim)
        return attention_feature, attn_weights
    
    def forward(self, sample_list):
        batch_size = len(sample_list["answers"])  
        scores = torch.ones((batch_size, self.max_len, self.vocab_size), dtype=torch.float, device=self.device)  # (bs, max_len, vocab_size)

        data, sample_list, timesteps = self.prepare_data(sample_list, batch_size)
        output = None
        batch_size_t = batch_size
        for t in range(timesteps):
            data, batch_size_t = self.get_data_t(t, data, batch_size_t, output)
            pi_t = data["texts"][:, t].unsqueeze(-1)  # (word_ids at t, 1)
            embedding = self.word_embedding(pi_t)  # (batch_size_t, d_model)
            attention_feature, _ = self.process_feature_embedding(sample_list, embedding[:, 0, :], batch_size_t=batch_size_t)
            output, updated_state = self.classifier(attention_feature, data["state"])  # (batch_size_t, vocab_size)
            data["state"] = updated_state 
            scores[:batch_size_t, t] = output

        model_output = {"scores": scores}
        return model_output
