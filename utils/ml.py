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