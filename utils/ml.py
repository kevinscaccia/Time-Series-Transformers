import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np

def create_chunks(serie, chunk_size):
    # sliding window
    chunks = []
    i = 0
    while (i+chunk_size) <= len(serie):
        chunks.append(serie[i:i+chunk_size])
        i += 1
    return np.asarray(chunks)

def make_batches(serie, input_len, forecast_horizon):
    chunk_size = input_len + forecast_horizon
    data = create_chunks(serie, chunk_size)
    data = [prepare_input(sample, input_len, forecast_horizon) for sample in data]
    enc_x = torch.tensor(np.asarray([sample[0] for sample in data]),dtype=torch.float32)
    dec_x = torch.tensor(np.asarray([sample[1] for sample in data]),dtype=torch.float32)
    tgt_y = torch.tensor(np.asarray([sample[2] for sample in data]),dtype=torch.float32)

    return enc_x, dec_x, tgt_y
def prepare_input(serie: torch.Tensor, encoder_len: int, target_len: int):
    #
    assert(len(serie) == encoder_len+target_len)
    serie = torch.tensor(serie, dtype=torch.float32)
    # encoder input
    encoder_input = serie[:encoder_len]
    # decoder input (It must have the same dimension as the target sequence. It must contain the last value of src, and all values of target_y except the last (i.e. it must be shifted right by 1))
    decoder_input = serie[encoder_len-1:-1]
    # target output (decoder output target)
    decoder_target = serie[-target_len:]
    return encoder_input.unsqueeze(-1), decoder_input.unsqueeze(-1), decoder_target.unsqueeze(-1)


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    
class TransformerDataset(Dataset):
    def __init__(self, dec_x, enc_x, tgt_y):
        self.dec_x = dec_x
        self.enc_x = enc_x
        self.tgt_y = tgt_y

    def __len__(self):
        return len(self.dec_x)
    
    def __getitem__(self,idx):
        return self.dec_x[idx], self.enc_x[idx], self.tgt_y[idx]

class DecoderDataset(Dataset):
    def __init__(self, x, y, mask):
        self.x = x
        self.y = y
        self.mask = mask

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx], self.mask[idx]

class EarlyStopperPercent():
    def __init__(self, patience=1, min_percent=0.1, verbose=False):
        self.patience = patience
        self.min_percent = min_percent
        self.counter = 0
        self.last_loss = 1e21
        #
        self.verbose = verbose
        self.epoch = 0

    def step(self, loss):
        self.epoch += 1
        # o loss atual é 10% menor do que o anterior?
        if (self.last_loss - loss)/self.last_loss >= self.min_percent:
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose: print(f'Loss dont decrease {(self.min_percent*100):.4f}% {self.counter}/{self.patience}')
            if self.counter >= self.patience: # if dont decrease n times..
                if self.verbose:  print(f'    Early Stopping on epoch {self.epoch}')
                return True # stop training
        self.last_loss = loss
        return False


class EarlyStopper():
    def __init__(self, patience=1, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')
        self.verbose = verbose
        self.epoch = 0
        self.last_loss = float('inf')

    def step(self, loss):
        self.epoch += 1
        # if loss decrease, set min loss and reset counter(dont stop training)
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        # 
        elif loss > (self.min_loss + self.min_delta): # if loss dont decrease by a delta 
            self.counter += 1
            if self.verbose: print(f'Loss dont decrease {self.counter}/{self.patience}')
            if self.counter >= self.patience: # if dont decrease n times..
                if self.verbose:  print(f'    Early Stopping on epoch {self.epoch}')
                return True # stop training
        self.last_loss = loss
        return False

def print_num_weights(model):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Num of weights: {n}')