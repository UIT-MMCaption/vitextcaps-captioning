from torch import nn
from torch.nn import functional as F

from data_utils.vocabs import Vocab

class ViWordEmbedder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.embed_dim = config.embedder.embed_dim
        self.model_type = config.embedder.model_type
        self.dropout_prob = config.embedder.dropout
        self.num_layer = config.embedder.num_layer
        self.device = config.model.device
        self.pad_idx = vocab.pad_idx
        self.total_tokens = vocab.total_tokens

        self.embedding = nn.Embedding(
            num_embeddings=self.total_tokens,
            embedding_dim=self.embed_dim,
            padding_idx=self.pad_idx,
        )
        self.proj = nn.Linear(
            in_features=self.embed_dim,
            out_features=self.embed_dim
        )

        if self.model_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.embed_dim,
                hidden_size=self.embed_dim,
                num_layers=self.num_layer,
                bidirectional=False,
                batch_first=True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )
        elif self.model_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.embed_dim,
                hidden_size=self.embed_dim,
                num_layers=self.num_layer,
                bidirectional=False,
                batch_first=True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )

    def forward(self, x):
        """
        (bs, seq_len, 5)
        """

        embedded = self.embedding(x)  # (bs, seq_len, 5, d_model)
        embedded = F.gelu(embedded)

        # turn the tensor into (bs*seq_len, 5, d_model)
        bs, seq_len, dim_1, dim_2 = embedded.shape
        embedded = embedded.reshape((-1, dim_1, dim_2))

        if self.model_type == "LSTM":
            _, (embedded, _) = self.rnn(embedded)
        else:
            _, embedded = self.rnn(embedded)
        
        embedded = embedded[-1]

        # turn the tensor back to (bs, seq_len, d_model)
        embedded = embedded.reshape((bs, seq_len, -1))

        return embedded
