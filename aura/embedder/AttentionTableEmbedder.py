from transformers import AutoModel
from torch import reshape, cat, Tensor, mean, stack, save
from torch.nn import Module, Linear, MultiheadAttention
from torch.nn.functional import pad, softmax


class AttentionTableEmbedder(Module):

    def __init__(self, d: int = 1024, num_heads: int = 8, dropout: float = 0.1, ff_hidden_dim: int = 1024):
        super(AttentionTableEmbedder, self).__init__()

        self.d = d
        self.num_heads = num_heads
        self.dropout = dropout
        self.ff_hidden_dim = ff_hidden_dim

        self.row_q_proj = Linear(ff_hidden_dim, d)
        self.row_k_proj = Linear(ff_hidden_dim, d)
        self.row_v_proj = Linear(ff_hidden_dim, d)

        self.row_attention = MultiheadAttention(embed_dim = d, num_heads = num_heads, dropout = dropout, batch_first = True)
        self.row_output = Linear(d, d)

        self.col_q_proj = Linear(ff_hidden_dim, d)
        self.col_k_proj = Linear(ff_hidden_dim, d)
        self.col_v_proj = Linear(ff_hidden_dim, d)

        self.col_attention = MultiheadAttention(embed_dim = d, num_heads = num_heads, dropout = dropout, batch_first = True)
        self.col_output = Linear(d, d)

        self.tab_q_proj = Linear(d, d)
        self.tab_k_proj = Linear(d, d)
        self.tab_v_proj = Linear(d, d)

        self.tab_attention = MultiheadAttention(embed_dim = d, num_heads = num_heads, dropout = dropout, batch_first = True)
        self.tab_output = Linear(d, d)

    def forward(self, rows: list[Tensor], cols: list[Tensor]) -> Tensor:
        embedded_rows = []
        embedded_cols = []

        for row in rows:
            attention_output, _ = self.row_attention(
               self.row_q_proj(row),
               self.row_k_proj(row),
               self.row_v_proj(row)
            )
            row_output = self.row_output(attention_output)

            print(row_output.shape)

            pooled = mean(row_output, dim = 0)  # TODO: apply NN

            embedded_rows.append(pooled)

        for col in cols:
            attention_output, _ = self.row_attention(
               self.row_q_proj(col),
               self.row_k_proj(col),
               self.row_v_proj(col)
            )
            col_output = self.col_output(attention_output)
            pooled = mean(col_output, dim = 0)  # TODO: apply NN
            embedded_cols.append(pooled)

        combined_vectors = stack(embedded_rows + embedded_cols, dim = 0)
        attention_output, _ = self.tab_attention(
            self.tab_q_proj(combined_vectors),
            self.tab_k_proj(combined_vectors),
            self.tab_v_proj(combined_vectors)
        )
        tab_output = self.tab_output(attention_output)
        tab_embedding = mean(tab_output, dim = 0)

        return tab_embedding

    def save(self, filepath: str):
        save({
            'model_state_dict': self.state_dict(),
            'd': self.d,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'ff_hidden_dim': self.ff_hidden_dim
        }, filepath)
