import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_size, heads, embed_size):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.tokeys = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
        self.toqueries = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
        self.tovalues = nn.Linear(self.input_size, self.emb_size * heads, bias = False)

    def forward(self, x):
        b, t, hin = x.size()
        assert hin == self.input_size, f'Input size {{hin}} should match {{self.input_size}}'
        
        h = self.heads 
        e = self.emb_size
        
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention
        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b*h, t, t)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return out