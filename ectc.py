import torch
import torch.nn as nn


class ContextEnhance(nn.Module): 
    
    def __init__(self, 
                 encoder, #an encoder model  
                 pooling, # a pooling funciton to aggregate encoder outputs
                 proj: nn.Module 
                 ): 

        self.encoder = encoder 
        self.pooling = pooling 
        self.proj = proj 
    
    def forward(self,
                src: torch.tensor, 
                tgt: torch.tensor): 
        
       #src.shape = (batch_size, seq_len)
       #tgt.shape = (batch_size, seq_len)
       batch_size = src.shape[0]

       ws = self.encoder(src) 
       wt = self.encoder(tgt)
       #ws.shape = wt.shape = (batch_size, seq_len, emb_dim)

       sigmas = torch.sum(ws, dim=1)
       sigmat = torch.sum(wt, dim=1)
       #sigmas.shape = sigmat.shape = (batch_size, emb_dim)

       zs = self.proj(sigmas)
       zt = self.proj(sigmat)
       #zs.shape = zt.shape = (batch_size, emb_dim2)
        
       c = zs.T @ zt 
       c = c.div_(batch_size)
       torch.distributed.all_reduce(c)
       #C.shape = (emb_dim2, emb_dim2)
       
       return c 
       