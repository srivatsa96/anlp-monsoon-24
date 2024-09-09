class Config:
    def __init__(self,block_size,tokenizer):
        ## Hyperparameters 
        self.batch_size = 128 
        self.block_size = block_size # Context window for for look back
        self.n_embed = 510 # Vocabulory embedding dimensions
        self.n_head = 6 # No of Heads
        self.num_heads = 6  # No of Heads
        self.n_layer = 6 # No of Transfomer Layers
        self.dropout = 0.4 # Dropout probablity
        self.vocab_size = tokenizer.vocab_size # No of tokens in the embedding table
        self.head_size = self.n_embed // self.n_head # Output Dimension of one head. Make Sure n_embed if perfect multiple of number of heads
        self.hidden = 600 # No of Hidden Dimension
        self.num_lstm_layer = 1 # No of LSTM Layers
        # --------------

class TrainConfig:
    def __init__(self, accelerator):
        self.epochs = 10  # No of Pass over train data
        self.device = accelerator.device #'cuda' if torch.cuda.is_available() else 'cpu'
        self.learning_rate = 1e-5 # Static Learning rate. TODO: Experiment with ScheduleLR