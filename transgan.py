import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

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
    def __init__(self, img_size=32, img_c = 3, model_size="s",):
        super(Generator, self).__init__()
        self.img_size, self.img_c = img_size, img_c
        self.cfg = CONFIG[model_size]
        self.C, layers = self.cfg["emb"], self.cfg["layers"]
        in_feats = img_size//4 * img_size//4 * self.C
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
        out = self.up1(self.stage1(out))
        out = self.up2(self.stage2(out))
        out = self.stage3(out)
        out = self.linear2(out).view(x.size(0), self.img_size, self.img_size, self.img_c).permute(0,3,1,2)
        return out

if __name__=="__main__":
    model_size="s"
    emb = CONFIG[model_size]["emb"]
    z = torch.randn(4, emb)
    g = Generator(img_size=32,model_size=model_size)
    out = g(z)
    torchsummary.summary(g, (emb,))
    print(out.shape)
