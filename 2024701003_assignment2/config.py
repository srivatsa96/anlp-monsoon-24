'''
As per Attention is all you need
'''
class Config:
    def __init__(self,tokeniser):
        self.n_embed = 512  # Embedding size
        self.num_heads = 8   # Number of attention heads
        self.dropout = 0.1    # Dropout rate
        self.vocab_size = tokeniser.vocab_size  # Vocabulary size
        self.pe_mode = 'sinusoid'  # Positional encoding method
        self.share_embedding = False  # Whether to share embeddings
        self.n_layer = 6  # Number of transformer layers
        self.enc_block_size = 50  # Encoder block size
        self.dec_block_size = 50  # Decoder block size
        self.head_size = self.n_embed // self.num_heads # Dimension of each head

class TrainConfig:
    def __init__(self, accelerator):
        self.batch_size = 40
        self.epochs = 10  # No of Pass over train data
        self.device = accelerator.device #'cuda' if torch.cuda.is_available() else 'cpu'
        self.learning_rate = 1e-4 # Static Learning rate. TODO: Experiment with ScheduleLR
