class LanguageDecoder(nn.Module): # Languague LSTM: v_hat (diminish dim) + hidden state -> next word 
    def __init__(self, in_dim, out_dim, hidden_dim, dropout, fc_bias_init, vis_feat_dim):
        super().__init__()
        self.vis_proj = nn.Linear(vis_feat_dim, in_dim) # 2048 -> 500
        self.language_lstm = nn.LSTMCell(in_dim+hidden_dim, hidden_dim) 
        self.fc = weight_norm(nn.Linear(hidden_dim, out_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights(fc_bias_init)

    def init_weights(self, fc_bias_init):
        self.fc.bias.data.fill_(fc_bias_init)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, v_hat, h1, h2, c2):
        v_hat_proj = self.vis_proj(v_hat) # (active, 500)
        x = torch.cat([v_hat_proj, h1], dim=1) # (active,500) (active,500)
        h2, c2 = self.language_lstm(x, (h2, c2))
        predictions = self.fc(self.dropout(h2))
        return predictions, h2, c2

class MMF_BUTD(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hidden_dim = 500
        self.d_model = 500
        self.visual_feat_dim = 2048
        self.proj_dim = 1000
        self.dropout = 0.1
        self.fc_bias_init = 1
        self.attn_dim = 512
        
        self.build()

    def build(self):
        self.word_embedding =  nn.Embedding(self.vocab_size, self.d_model)
        self.vis_proj = nn.Linear(self.visual_feat_dim, self.proj_dim)
        self.att_lstm = nn.LSTMCell(self.hidden_dim + self.proj_dim + self.d_model, self.hidden_dim)

        # Attention for region features
        self.att_mlp_v = nn.Linear(self.visual_feat_dim, self.attn_dim) # 2048 -> 512
        self.att_mlp_h = nn.Linear(self.hidden_dim, self.attn_dim) # 500 -> 512
        self.att_mlp_score = nn.Linear(self.attn_dim, 1) # 512 -> 1

        # LanguageDecoder 
        self.language_decoder = LanguageDecoder(
            in_dim=self.d_model,  
            out_dim=self.vocab_size, 
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            fc_bias_init=self.fc_bias_init,
            vis_feat_dim=self.visual_feat_dim
        )

    def init_hidden(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_dim)
        c = torch.zeros(batch_size, self.hidden_dim)
        return h, c

    def forward(self, items):
        region_features = items.region_features
        answer_tokens = items.answer_tokens
        answer_tokens = answer_tokens.clone()
        answer_mask = items.answer_mask
        decode_lengths = (answer_mask != 0).sum(dim=1)

        # convert <unk> 
        invalid_mask = (answer_tokens >= self.vocab_size) | (answer_tokens < 0)
        if invalid_mask.any():
            answer_tokens[invalid_mask] = 3
        
        batch_size, num_regions, _ = region_features.size()
        max_len = answer_tokens.size(1)

        embeddings = self.word_embedding(answer_tokens)
        scores = torch.zeros(batch_size, max_len, self.vocab_size)
        h1, c1 = self.init_hidden(batch_size) # (bs, hd)
        h2, c2 = self.init_hidden(batch_size) # (bs, hd)

        for t in range(max_len):
            mask = decode_lengths > t
            active_indices = torch.nonzero(mask).squeeze(1) # indexing
            if active_indices.numel() == 0:
                break
            v = region_features[active_indices] # (active, num_regions, 2048)
            mean_feat = v.mean(dim=1)   # (active, 2048)
            mean_feat_proj = self.vis_proj(mean_feat) # (active, 1000)
            current_word_emb = embeddings[active_indices, t, :] # (active, 500)
            h2_active = h2[active_indices]
            attn_lstm_input = torch.cat([h2_active, mean_feat_proj, current_word_emb], dim=1) # [(500), (1000), (500)]
            h1_active = h1[active_indices]
            c1_active = c1[active_indices]
            h1_t, c1_t = self.att_lstm(attn_lstm_input, (h1_active, c1_active))

            # attention of region features
            proj_v = self.att_mlp_v(v)  # (active, num_regions, 512)
            proj_h = self.att_mlp_h(h1_t).unsqueeze(1) # (active, 1, 512)
            scores_att = self.att_mlp_score(torch.tanh(proj_v + proj_h)).squeeze(2)  # (active, num_regions)
            alpha = F.softmax(scores_att, dim=1)  # (active, num_regions)
            v_hat = torch.bmm(alpha.unsqueeze(1), v).squeeze(1)

            h2_active_old = h2[active_indices]
            c2_active_old = c2[active_indices]
            predictions, h2_new, c2_new = self.language_decoder(
                v_hat,
                h1_t,
                h2_active_old,
                c2_active_old
            )

            # Update
            h1[active_indices] = h1_t
            c1[active_indices] = c1_t
            h2[active_indices] = h2_new
            c2[active_indices] = c2_new

            scores[active_indices, t, :] = predictions


        return {"scores": scores}
