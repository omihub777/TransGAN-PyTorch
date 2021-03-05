import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import einops

from layers import TransformerEncoder, UpScale

CONFIG = {
    "s":{
        "layers":[5,2,2],
        "emb":384            
    },
    "m":{
        "layers":[5,2,2],
        "emb":512            
    },
    "l":{
        "layers":[5,2,2],
        "emb":768           
    },
    "xl":{
        "layers":[5,4,2],
        "emb":1024       
    }

}


class Generator(nn.Module):
    def __init__(self,img_c:int=3, img_size:int=32, model_size:str="s"):
        super(Generator, self).__init__()
        self.img_size, self.img_c = img_size, img_c
        self.cfg = CONFIG[model_size]
        self.C, layers = self.cfg["emb"], self.cfg["layers"]
        init_img = img_size//4
        in_feats = init_img * init_img * self.C

        self.pos_emb = nn.Parameter(torch.randn(1, init_img**2, 1))
        self.linear1 = nn.Linear(self.C, in_feats)
        stage1 = [TransformerEncoder(self.C) for _ in range(layers[0])]
        stage2 = [TransformerEncoder(self.C//4) for _ in range(layers[1])]
        stage3 = [TransformerEncoder(self.C//16) for _ in range(layers[2])]
        self.stage1 = nn.Sequential(*stage1)
        self.up1 = UpScale()
        self.stage2 = nn.Sequential(*stage2)
        self.up2 = UpScale()
        self.stage3 = nn.Sequential(*stage3)
        
        self.linear2 = nn.Linear(self.C//16, img_c)
    
    def forward(self, x):
        out = self.linear1(x).view(x.size(0), -1,self.C)
        out = out + self.pos_emb
        out = self.up1(self.stage1(out))
        out = self.up2(self.stage2(out))
        out = self.stage3(out)
        out = self.linear2(out).view(x.size(0), self.img_size, self.img_size, self.img_c).permute(0,3,1,2)
        return out


class Discriminator(nn.Module):
    def __init__(self, img_c:int=3, img_size:int=32, patch:int=8):
        super(Discriminator, self).__init__()
        hidden=384

        self.patch = patch # number of patches in one row(or col)
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.pos_emb = nn.Parameter(torch.randn(1, (self.patch**2)+1, hidden))
        self.enc = nn.Sequential(
            TransformerEncoder(hidden),
            TransformerEncoder(hidden),
            TransformerEncoder(hidden),
            TransformerEncoder(hidden),
            TransformerEncoder(hidden),
            TransformerEncoder(hidden),
            TransformerEncoder(hidden)
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1) # for cls_token
        )


    def forward(self, x):
        out = self._to_words(x)
        out = torch.cat([self.cls_token.repeat(out.size(0),1,1), self.emb(out)],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        out = out[:,0]
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        # out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1).contiguous()
        # out = out.view(x.size(0), self.patch**2 ,-1)
        out = einops.rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        return out



if __name__=="__main__":
    model_size="s"
    emb = CONFIG[model_size]["emb"]
    z = torch.randn(4, emb)
    g = Generator(img_size=32,model_size=model_size)
    d = Discriminator(img_c=3, img_size=32, patch=8)
    out = d(g(z))
    torchsummary.summary(g, (emb,))
    torchsummary.summary(d, (3,32,32))
    print(out.shape)
