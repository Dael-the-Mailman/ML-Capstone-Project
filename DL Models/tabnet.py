import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from dotenv import dotenv_values

import pytorch_lightning as pl
import torch.nn.functional as F
import dask.dataframe as dd

"""
Derived from
https://towardsdatascience.com/implementing-tabnet-in-pytorch-fc977c383279
https://github.com/dreamquark-ai/tabnet/
"""

class Sparsemax(nn.Module):
    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)
        
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, device=device,step=1, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]
        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)
        self.output = torch.max(torch.zeros_like(input), input - taus)
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)
        return output

    def backward(self, grad_output):
        dim = 1
        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))
        return self.grad_input

def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return

def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return

class GBN(nn.Module):
    def __init__(self, input_dim, vbs=128, momentum=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.bn = nn.BatchNorm1d(input_dim,momentum=momentum)
        self.vbs = vbs
    
    def forward(self, x):
        chunk = torch.chunk(x, max(1, x.size(0)//self.vbs), 0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res, 0)

class GLU(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 fc=None, 
                 vbs=128, 
                 momentum=0.02):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, output_dim*2, bias=False)
        initialize_glu(self.fc, input_dim, 2*output_dim)
        self.bn = GBN(output_dim*2, vbs=vbs, momentum=momentum)
        self.output_dim = output_dim
    
    def forward(self, x):
        x = self.bn(self.fc(x))
        return torch.mul(x[:,:self.output_dim], torch.sigmoid(x[:,self.output_dim:]))

class GLU_BLock(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_glu=2,
                 first=False,
                 shared_layers=None,
                 vbs=128,
                 momentum=0.02
                 ):
        super().__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = nn.ModuleList()
        
        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLU(input_dim, output_dim, fc=fc, vbs=vbs, momentum=momentum))
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU(output_dim, output_dim, fc=fc, vbs=vbs, momentum=momentum))
        
    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5])).to(x.device)
        if self.first:
            x = self.glu_layers[0](x)
            layers_left = range(1,self.n_glu)
        else:
            layers_left = range(self.n_glu)
        
        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x

class FeatureTransformer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 shared_layers, 
                 n_glu_independent, 
                 vbs=128,
                 momentum=0.02):
        super().__init__()
        
        if shared_layers is None:
            self.shared = nn.Identity()
            is_first=True
        else:
            self.shared = GLU_BLock(
                input_dim,
                output_dim,
                first=True,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                vbs=vbs,
                momentum=momentum
            )
            is_first = False
        
        if n_glu_independent == 0:
            self.specifics = nn.Identity()
        else:
            spec_input_dim = input_dim if is_first else output_dim
            self.specifics = GLU_BLock(
                spec_input_dim, 
                output_dim, 
                first=is_first, 
                n_glu=n_glu_independent,
                vbs=vbs,
                momentum=momentum
            )
    
    def forward(self, x):
        return self.specifics(self.shared(x))

class AttentionTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 gamma=1.3,
                 vbs=128,
                 momentum=0.02):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, output_dim)
        self.bn = GBN(
            output_dim,
            vbs=vbs,
            momentum=momentum
        )
        self.gamma = gamma
    
    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = torch.sigmoid(a*priors)
        priors = priors*(self.gamma - mask)
        return mask

class DecisionStep(nn.Module):
    def __init__(self,
                 input_dim,
                 n_d=8,
                 n_a=8,
                 shared_layers=None,
                 n_independent=2,
                 gamma=1.3,
                 vbs=128,
                 momentum=0.02):
        super().__init__()
        self.feat = FeatureTransformer(
            input_dim,
            n_d+n_a,
            shared_layers,
            n_independent,
            vbs,
            momentum
        )

        self.attn = AttentionTransformer(
            n_a,
            input_dim,
            gamma,
            vbs,
            momentum
        )

class TabNet(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_d=8,
                 n_a=8,
                 n_shared=2,
                 n_independent=2,
                 n_steps=3,
                 gamma=1.3,
                 vbs=128,
                 momentum=0.02):
        super().__init__()
        if n_shared>0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(input_dim,2*(n_d+n_a)))
            for _ in range(n_shared-1):
                self.shared.append(nn.Linear(n_d+n_a,2*(n_d+n_a)))
        else:
            self.shared = None
        self.first_step = FeatureTransformer(
            input_dim, 
            n_d+n_a,
            self.shared,
            n_independent,
            vbs,
            momentum
        )
        self.steps = nn.ModuleList()
        for _ in range(n_steps-1):
            self.steps.append(DecisionStep(
                input_dim,
                n_d,
                n_a,
                self.shared,
                n_independent,
                gamma,
                vbs,
                momentum
            ))
        self.fc = nn.Linear(n_d, output_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.n_d = n_d

    def forward(self, x):
        x = self.bn
        x_a = self.first_step(x)[:,self.n_d:]
        loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te, l = step(x, x_a, priors)
            out += F.relu(x_te[:,:self.n_d])
            x_a = x_te[:,self.n_d:]
            loss += l
        return self.fc(out), loss

class TabNetWithEmbed(pl.LightningModule):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 n_d=8,
                 n_a=8,
                 n_shared=2,
                 n_independent=2,
                 n_steps=3,
                 gamma=1.3,
                 vbs=128,
                 momentum=0.02):
        self.tabnet = TabNet(
            input_dim,
            output_dim,
            n_d,
            n_a,
            n_shared,
            n_independent,
            n_steps,
            gamma,
            vbs,
            momentum
        )
        self.lr=1e-3
    
    def forward(self, x):
        x, l = self.tabnet(x)
        return torch.sigmoid(x), l
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out, _ = self.tabnet(x)
        loss = F.mse_loss(out, y)
        return loss

if __name__ == '__main__':
    config = dotenv_values("../.env")
    df = dd.read_csv(config["WRANGLED_DATA"] + "scaled_train/train-*.csv.part")
    df = df.compute()
    print(df.head())
    
    