import torch
class Config():
    def __init__(self):
        self.n_vocab = 1002
        self.embed_size = 256
        self.hidden_size = 256
        self.num_layers = 5
        self.dropout = 0.8
        self.num_classes = 2
        self.pad_size = 32
        self.batch_size = 256
        self.is_shuffle = True
        self.learn_rate = 0.001
        self.num_epochs = 10
        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
