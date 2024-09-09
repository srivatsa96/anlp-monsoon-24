import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):

    def __init__(self,config,device):
        super().__init__()
        self.config = config
        self.device = device
        self.token_embedding_table = nn.Embedding(
                self.config.vocab_size, 
                self.config.n_embed
        )
        self.lstm = nn.LSTM(
            self.config.n_embed,
            self.config.hidden,
            self.config.num_lstm_layer,
            batch_first=True,
            dropout=self.config.dropout
        )

        self.vocab_proj =  nn.Linear(
                self.config.num_lstm_layer*self.config.hidden,
                self.config.vocab_size
        )
    
    def forward(self,x,target=None):
        x = self.token_embedding_table(x)
        B,T,C = x.shape
        x,(h_n,c_n) = self.lstm(x)
        logits = self.vocab_proj(x)
        if target is None:
            loss = None 
        else:
            B,T,C = logits.shape 
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits,loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx