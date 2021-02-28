import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_c: int, head=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.sqrt_d = math.sqrt(in_c)

        self.Q = nn.Linear(in_c, in_c)
        self.K = nn.Linear(in_c, in_c)
        self.V = nn.Linear(in_c, in_c)

        self.O = nn.Linear(in_c, in_c)

    def forward(self, x):
        """
        x: (b, n, f)
        """
        b, n, f = x.size()
        q = self.Q(x).view(b, n, self.head, f//self.head)
        k = self.K(x).view(b, n, self.head, f//self.head)
        v = self.V(x).view(b, n, self.head, f//self.head)

        score = torch.einsum("bihf, bjhf->bhij",q, k)
        attn = F.softmax(score/self.sqrt_d, dim=-1) # (b, h, n, n)
        o = torch.einsum("bhij,bjhf->bihf", attn, v)  # (b, n, h, f//self.head)
        o = self.O(o.flatten(2))
        return o

class TransformerEncoder(nn.Module):
    def __init__(self, in_c, head=8):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(in_c)
        self.msa = MultiHeadSelfAttention(in_c, head)
        self.la2 = nn.LayerNorm(in_c)
        self.mlp = nn.Linear(in_c, in_c)

    def forward(self, x):
        out = self.msa(self.la1(x))+x
        out = F.gelu(self.mlp(self.la2(out)))+out
        return out

class UpScale(nn.Module):
    def __init__(self, upscale_factor=2):
        super(UpScale,self).__init__()
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, x):
        """
        x: (b, n, f)
        """
        b, n, f = x.size()
        n_sqrt = int(math.sqrt(n))
        out = x.view(b, n_sqrt, n_sqrt, f).permute(0,3,1,2)
        out = self.pixelshuffle(out) # (b, f//4, n_sqrt*2, n_sqrt*2)
        out = out.permute(0,2,3,1).contiguous().view(b,-1,f//4)
        return out 

if __name__=="__main__":
    b, c, h, w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)
    p = h//8
    out = x.unfold(2, p, p).unfold(3, p, p).permute(0,2,3,4,5,1).contiguous()
    out = out.view(b,64,-1)
    # sa = MultiHeadSelfAttention(p*p*c)
    enc = TransformerEncoder(in_c=p*p*c)
    out = enc(out)
    up = UpScale(2)
    out = up(out)
    print(out.shape)







