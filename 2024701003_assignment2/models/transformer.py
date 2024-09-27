import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    ''' One head of Self Attention '''

    def __init__(self,config,mode='causal'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.key = nn.Linear(self.config.n_embed, self.config.head_size, bias=False)
        self.query = nn.Linear(self.config.n_embed, self.config.head_size, bias=False)
        self.value = nn.Linear(self.config.n_embed, self.config.head_size, bias=False)
        self.dropout = nn.Dropout(self.config.dropout)
        if self.mode == 'causal':
            self.register_buffer('tril',torch.tril(torch.ones(self.config.dec_block_size, self.config.dec_block_size)))
    
    def forward(self,q_vec,k_vec,v_vec,mask=None):
        B,T,C = q_vec.shape
        k = self.key(k_vec)
        q = self.query(q_vec)
        attn_scores = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        if mask is not None:
            mask = self._expand_along_time(mask,attn_scores.shape[1])
            if self.mode == 'causal':
                mask = mask * self.tril[:T,:T]  ## (B T T) * (T T) --> Causal Attention during decoding
            attn_scores = attn_scores.masked_fill(mask==0,float('-inf'))
        attn_weight = F.softmax(attn_scores,dim=-1)
        attn_weight = self.dropout(attn_weight)
        v = self.value(v_vec)
        out = attn_weight @ v
        return out
    
    def _expand_along_time(self,mask,time):
        B,T = mask.shape
        return mask.repeat(1, time).view(B,time,T)


class MultiHeadAttention(nn.Module):
    ''' Multiple heads of self attention in parallel '''

    def __init__(self,config,mode='causal'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.heads = nn.ModuleList([Head(self.config,self.mode) for _ in range(self.config.num_heads)])
        self.proj = nn.Linear(self.config.n_embed,self.config.n_embed)
        self.dropout = nn.Dropout(self.config.dropout)
        self.mode = mode
    
    def forward(self,q_vec,k_vec,v_vec,mask=None):
        out = torch.cat([h(q_vec,k_vec,v_vec,mask) for h in self.heads],dim=-1)
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

'''
In "encoder-decoder attention" layers, the queries come from the previous decoder layer,
and the memory keys and values come from the output of the encoder. This allows every
position in the decoder to attend over all positions in the input sequence. This mimics the
typical encoder-decoder attention mechanisms in sequence-to-sequence models such as
[31, 2, 8].
'''
class Block(nn.Module):
    ''' Transformer block: communication followed by computation ''' 

    def __init__(self,config,cross_attention=True,block_type='decoder'):
        super().__init__()
        self.config = config
        self.cross_attention=cross_attention

        # In decoder we apply masked self attention while in encoder we do full attention
        if block_type == 'decoder':
            self.sa = MultiHeadAttention(self.config,'causal')
        else:
            self.sa = MultiHeadAttention(self.config,'full')

        self.ca = MultiHeadAttention(self.config,'full')
        self.ffwd = FeedForward(self.config)
        self.ln1 = nn.LayerNorm(self.config.n_embed)

        # Should cross attention be added (decoder can be w/o one)
        if self.cross_attention:
            self.ln2 = nn.LayerNorm(self.config.n_embed)
        
        self.ln3 = nn.LayerNorm(self.config.n_embed)
    
    def forward(self,input):
        x,x_mask,mem,mem_mask = input
        x = self.ln1(x + self.sa(q_vec=x,k_vec=x,v_vec=x,mask=x_mask))
        if self.cross_attention:
            if mem == None:
                raise ValueError('Encoder Input Required for Cross Attention')
            x = self.ln2(x + self.ca(q_vec=x,k_vec=mem,v_vec=mem,mask=mem_mask))
        x = self.ln3(x + self.ffwd(x))
        
        return (x,x_mask,mem,mem_mask)

class EncoderDecoderTransformer(nn.Module):

    def __init__(self,config,device):
        super().__init__()
        self.config = config
        self.device = device

        ## Define Embedding Tables (Both have same dimension as using the ss)
        self.input_vocab_embedding_table = nn.Embedding(self.config.vocab_size + 10, self.config.n_embed)
        if not self.config.share_embedding:
            self.output_vocab_embedding_table = nn.Embedding(self.config.vocab_size + 10, self.config.n_embed)
        

        ## Define Positional Embedding
        if self.config.pe_mode == 'sinusoid':
            self._create_pe_cache()
            self.encoder_position_embedding_table = self.pe
            self.decoder_position_embedding_table = self.pe
        elif self.config.pe_mode == 'learnable':
            self.encoder_position_embedding_table = nn.Embedding(self.config.enc_block_size, self.config.n_embed)
            self.decoder_position_embedding_table = nn.Embedding(self.config.dec_block_size, self.config.n_embed)
        else:
            raise ValueError('Unsupported PE')


        self.encoder_blocks = nn.Sequential(*[Block(config,cross_attention=False,block_type='encoder') for _ in range(self.config.n_layer)])
        self.decoder_blocks = nn.Sequential(*[Block(config,cross_attention=True,block_type='decoder') for _ in range(self.config.n_layer)])
        
        self.ln_f = nn.LayerNorm(self.config.n_embed)
        self.lm_head = nn.Linear(self.config.n_embed, self.config.vocab_size + 10)

        self._create_pe_cache()
    
    def forward(self, enc_idx, dec_idx, target=None, enc_mask=None, dec_mask=None,target_mask=None):
        B_enc,T_enc = enc_idx.shape 

        encoder_input = self.input_vocab_embedding_table(enc_idx) + self.pe[:,:T_enc] # Input + Input PE
        B_dec,T_dec = dec_idx.shape

        if B_enc != B_dec:
            raise ValueError('Encoder Decoder Input Should have same batch size')

        ## If Embedding Layer is shared between input and output layer.
        if self.config.share_embedding:
            decoder_input = self.input_vocab_embedding_table(target) + self.pe[:,:T_dec]
        else:
            decoder_input = self.output_vocab_embedding_table(target) + self.pe[:,:T_dec]
        
        mem,_,_,_ = self.encoder_blocks((encoder_input, enc_mask, None, None))
        output,_,_,_ = self.decoder_blocks((decoder_input,dec_mask,mem,enc_mask))

        logits = self.lm_head(output)

        if target is None:
            loss = None 
        else:
            B,T,C = logits.shape 
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            target_mask = target_mask.view(B*T)
            
            valid_logits = logits[target_mask.bool()]
            valid_targets = target[target_mask.bool()]
            loss = F.cross_entropy(valid_logits, valid_targets)
            print(type(loss))
        return logits, loss 
    
    ## Not supporting batch prediction as inputs are not padded.
    def generate(self, enc_idx, dec_idx, max_new_tokens):

        if(enc_idx.shape[0]!=1):
            raise ValueError('Only Single Sample Prediction Supported')
        if(enc_idx.shape[1]>self.config.enc_block_size):
            raise ValueError('Input Sentence longer than what model is trained on')

        for _ in range(max_new_tokens):
            idx_cond = dec_idx[:,-self.config.dec_block_size:]
            logits, loss = self(enc_idx,idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            dec_idx = torch.cat((dec_idx, idx_next), dim=1)
        return dec_idx
    
    def _create_pe_cache(self):
        max_len = max(self.config.enc_block_size,self.config.dec_block_size)
        pe = torch.zeros(max_len, self.config.n_embed)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.config.n_embed, 2) *
                             -(torch.log(torch.tensor(10000.0)) / self.config.n_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)


