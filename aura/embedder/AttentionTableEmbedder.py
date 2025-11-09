from transformers import AutoModel
from torch import reshape, cat, Tensor, mean, stack, save
from torch.nn import Module, Linear, MultiheadAttention, LayerNorm, Dropout, GELU, Sequential, init
from torch.nn.functional import pad, softmax


class AttentionTableEmbedder(Module):

    def __init__(self, input_dim: int = 1024, d: int = 1024, num_heads: int = 8, dropout: float = 0.1, ff_hidden_dim: int = 2048):
        super(AttentionTableEmbedder, self).__init__()

        self.input_dim = input_dim
        self.d = d
        self.num_heads = num_heads
        self.dropout = dropout
        self.ff_hidden_dim = ff_hidden_dim

        self.row_skip_proj = Linear(input_dim, d)
        self.row_q_proj = Linear(input_dim, d)
        self.row_k_proj = Linear(input_dim, d)
        self.row_v_proj = Linear(input_dim, d)

        self.row_attention = MultiheadAttention(embed_dim = d, num_heads = num_heads, dropout = dropout, batch_first = True)
        self.row_norm1 = LayerNorm(d)
        self.row_ff = Sequential(
            Linear(d, ff_hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(ff_hidden_dim, d)
        )
        self.row_norm2 = LayerNorm(d)

        self.col_skip_proj = Linear(input_dim, d)
        self.col_q_proj = Linear(input_dim, d)
        self.col_k_proj = Linear(input_dim, d)
        self.col_v_proj = Linear(input_dim, d)

        self.col_attention = MultiheadAttention(embed_dim = d, num_heads = num_heads, dropout = dropout, batch_first = True)
        self.col_norm1 = LayerNorm(d)
        self.col_ff = Sequential(
            Linear(d, ff_hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(ff_hidden_dim, d)
        )
        self.col_norm2 = LayerNorm(d)

        self.row_pooler = Sequential(
            Linear(d, ff_hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(ff_hidden_dim, d)
        )
        self.col_pooler = Sequential(
            Linear(d, ff_hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(ff_hidden_dim, d)
        )

        self.tab_q_proj = Linear(d, d)
        self.tab_k_proj = Linear(d, d)
        self.tab_v_proj = Linear(d, d)

        self.tab_attention = MultiheadAttention(embed_dim = d, num_heads = num_heads, dropout = dropout, batch_first = True)
        self.tab_norm1 = LayerNorm(d)
        self.tab_ff = Sequential(
            Linear(d, ff_hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(ff_hidden_dim, d)
        )
        self.tab_norm2 = LayerNorm(d)

        self.tab_pooler = Sequential(
            Linear(d, ff_hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(ff_hidden_dim, d)
        )

        self.tab_output = Linear(d, d)

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, rows: list[Tensor], cols: list[Tensor]) -> Tensor:
        embedded_rows = []
        embedded_cols = []

        for row in rows:
            attention_output, _ = self.row_attention(
               self.row_q_proj(row),
               self.row_k_proj(row),
               self.row_v_proj(row)
            )
            row_output = self.row_norm1(self.row_skip_proj(row) + attention_output)
            ff_output = self.row_ff(row_output)
            row_output = self.row_norm2(row_output + ff_output)
            pooled = self.row_pooler(row_output.mean(dim = 0, keepdim = True)).squeeze(0)
            embedded_rows.append(pooled)

        for col in cols:
            attention_output, _ = self.col_attention(
               self.col_q_proj(col),
               self.col_k_proj(col),
               self.col_v_proj(col)
            )
            col_output = self.col_norm1(self.col_skip_proj(col) + attention_output)
            ff_output = self.col_ff(col_output)
            col_output = self.col_norm2(col_output + ff_output)
            pooled = self.col_pooler(col_output.mean(dim = 0, keepdim = True)).squeeze(0)
            embedded_cols.append(pooled)

        combined_vectors = stack(embedded_rows + embedded_cols, dim = 0)
        attention_output, _ = self.tab_attention(
            self.tab_q_proj(combined_vectors),
            self.tab_k_proj(combined_vectors),
            self.tab_v_proj(combined_vectors)
        )
        tab_output = self.tab_norm1(combined_vectors + attention_output)
        ff_output = self.tab_ff(tab_output)
        tab_output = self.tab_norm2(tab_output + ff_output)

        tab_embedding = self.tab_pooler(tab_output.mean(dim = 0, keepdim = True)).squeeze(0)
        tab_embedding = self.tab_output(tab_embedding)

        return tab_embedding

    def save(self, filepath: str):
        save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'd': self.d,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'ff_hidden_dim': self.ff_hidden_dim
        }, filepath)

    @classmethod
    def load(cls, filepath: str):
        checkpoint = torch.load(filepath, map_location = 'cpu')
        model = cls(
            input_dim = checkpoint['input_dim'],
            d = checkpoint['d'],
            num_heads = checkpoint['num_heads'],
            dropout = checkpoint['dropout'],
            ff_hidden_dim = checkpoint['ff_hidden_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        return model
