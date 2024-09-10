import torch
import torch.nn as nn
from torch.nn import functional as F

class NGramLanguageModel(nn.Module):

    def __init__(self,config,device):
        super().__init__()
        self.config = config
        self.device = device
        self.token_embedding_table = nn.Embedding(self.config.vocab_size, self.config.n_embed)
        #self.ln2 = nn.LayerNorm(self.config.hidden)
        self.fc1 = nn.Sequential(
            nn.Linear(self.config.block_size*self.config.n_embed, self.config.hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )
        self.fc2 = nn.Linear(self.config.hidden, self.config.vocab_size)
    
    def forward(self,x,target=None):
        x = self.token_embedding_table(x)
        B,T,C = x.shape
        x = self._get_concatenated_input(x,self.config.block_size)
        x = self.fc1(x)
        logits = self.fc2(x)
        if target is None:
            loss = None 
        else:
            B,T,C = logits.shape 
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits,loss
    
    def _get_concatenated_input(self,x,window_size):
        B, T, C = x.shape
        padded_tensor = torch.nn.functional.pad(x, (0,0,window_size-1, 0))
        padded_tensor = padded_tensor.view(B, (T + (window_size-1)) *C)
        unfolded = padded_tensor.unfold(1, window_size*C, C)
        return unfolded
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
