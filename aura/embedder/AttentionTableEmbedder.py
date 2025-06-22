from transformers import AutoModel
from torch import reshape, cat
from torch.nn import Module, Linear
from torch.nn.functional import pad, softmax


class AttentionTableEmbedder(Module):

    def __init__(self, model: str, cuda: bool = True):
        super().__init__()

        self.model_name = model
        self.model = model = AutoModel.from_pretrained(model)

        self.hidden_size = hidden_size = model.config.hidden_size

        self.cell_wise_attention = cell_wise_attention = Linear(hidden_size, 1)
        self.row_wise_attention = row_wise_attention = Linear(hidden_size, 1)
        self.column_wise_attention = column_wise_attention = Linear(hidden_size, 1)

        self.table_wise_attention = table_wise_attention = Linear(hidden_size, 1)

        if cuda:
            model.to('cuda')

            row_wise_attention.to('cuda')
            column_wise_attention.to('cuda')
            table_wise_attention.to('cuda')

    def forward(self, *args, n_levels: int, **kwargs):
        # embedding

        embeddings = self.model(*args, **kwargs).last_hidden_state
        embeddings = embeddings.masked_fill(~kwargs['attention_mask'][..., None].bool(), 0.0)

        # cell-wise attention

        x = embeddings
        a, b, c = x.shape

        attn_scores = self.cell_wise_attention(x)
        attn_weights = softmax(attn_scores, dim = 1)
        attn_weights = attn_weights.expand(-1, -1, c)
        x_weighted = (x * attn_weights).sum(dim = 1)

        cell_embeddings = x_weighted

        new_shape = (n_levels, cell_embeddings.shape[0] // n_levels) + cell_embeddings.shape[1:]
        embeddings = reshape(cell_embeddings, new_shape)

        # row-wise attention

        x = embeddings

        attn_scores = self.row_wise_attention(x)
        attn_weights = softmax(attn_scores, dim = 1)
        attn_weights = attn_weights.expand(-1, -1, c)
        x_weighted = (x * attn_weights).sum(dim = 1)

        row_embeddings = x_weighted

        # column-wise attention

        x_ = x.permute(1, 0, 2)

        attn_scores = self.column_wise_attention(x_)
        attn_weights = softmax(attn_scores, dim = 1)
        x_weighted = (x_ * attn_weights).sum(dim = 1)

        column_embeddings = x_weighted

        # global attention

        x = cat((row_embeddings, column_embeddings), dim = 0)

        attn_scores = self.table_wise_attention(x)
        attn_weights = softmax(attn_scores, dim = 0)
        x_weighted = (x * attn_weights).sum(dim = 0)

        return x_weighted
