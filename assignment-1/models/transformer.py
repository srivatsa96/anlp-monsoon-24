import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    ''' One head of Self Attention '''

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.key = nn.Linear(self.config.n_embed, self.config.head_size, bias=False)
        self.query = nn.Linear(self.config.n_embed, self.config.head_size, bias=False)
        self.value = nn.Linear(self.config.n_embed, self.config.head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(self.config.block_size, self.config.block_size)))
        self.dropout = nn.Dropout(self.config.dropout)
    
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        #print(x.shape, k.shape, q.shape)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    ''' Multiple heads od self attention in parallel '''

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([Head(self.config) for _ in range(self.config.num_heads)])
        self.proj = nn.Linear(self.config.n_embed,self.config.n_embed)
        self.dropout = nn.Dropout(self.config.dropout)
    
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        #print('Out shape',out.shape)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    ''' A simple linear layer followed by non linearity '''
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(self.config.n_embed, 4*self.config.n_embed),
            nn.ReLU(),
            nn.Linear(4*self.config.n_embed,self.config.n_embed),
            nn.Dropout(self.config.dropout)
        )
    
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    ''' Transformer block: communication followed by computation ''' 

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.sa = MultiHeadAttention(self.config)
        self.ffwd = FeedForward(self.config)
        self.ln1 = nn.LayerNorm(self.config.n_embed)
        self.ln2 = nn.LayerNorm(self.config.n_embed)
    
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TSGLangauageModel(nn.Module):

    def __init__(self,config,device):
        super().__init__()
        self.config = config
        self.device = device
        self.token_embedding_table = nn.Embedding(self.config.vocab_size, self.config.n_embed)
        self.position_embedding_table = nn.Embedding(self.config.block_size, self.config.n_embed)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embed)
        self.lm_head = nn.Linear(self.config.n_embed, self.config.vocab_size)
    
    def forward(self, idx, target=None):
        B,T = idx.shape 
        tok_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(T,device=self.device))
        x = tok_embed + pos_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None 
        else:
            B,T,C = logits.shape 
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        
        return logits, loss 
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
