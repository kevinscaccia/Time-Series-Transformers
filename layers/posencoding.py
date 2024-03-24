import torch
from torch import nn
import math
#
# PatchTST
# #
# class PositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEmbedding, self).__init__()
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False

#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return self.pe[:, :x.size(1)]


# #
# # https://github.com/AIStream-Peelout/flow-forecast/blob/master/flood_forecast/transformer_xl/transformer_basic.py
# #
# class SimplePositionalEncoding(torch.nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(SimplePositionalEncoding, self).__init__()
#         self.dropout = torch.nn.Dropout(p=dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Creates a basic positional encoding"""
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

# class PositionalEncoder(nn.Module):
#     """
#     The authors of the original transformer paper describe very succinctly what 
#     the positional encoding layer does and why it is needed:
    
#     "Since our model contains no recurrence and no convolution, in order for the 
#     model to make use of the order of the sequence, we must inject some 
#     information about the relative or absolute position of the tokens in the 
#     sequence." (Vaswani et al, 2017)
#     Adapted from: 
#     https://pytorch.org/tutorials/beginner/transformer_tutorial.html
#     """

#     def __init__(
#         self, 
#         d_model: int=512,
#         dropout: float=0.1, 
#         max_seq_len: int=5000, 
#         batch_first: bool=True
#         ):

#         """
#         Parameters:
#             dropout: the dropout rate
#             max_seq_len: the maximum length of the input sequences
#             d_model: The dimension of the output of sub-layers in the model 
#                      (Vaswani et al, 2017)
#         """

#         super().__init__()

#         self.d_model = d_model
        
#         self.dropout = nn.Dropout(p=dropout)

#         self.batch_first = batch_first

#         self.x_dim = 1 if batch_first else 0

#         # copy pasted from PyTorch tutorial
#         position = torch.arange(max_seq_len).unsqueeze(1)
        
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
#         pe = torch.zeros(max_seq_len, 1, d_model)
        
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
        
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
        
#         self.register_buffer('pe', pe)
        
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
#                [enc_seq_len, batch_size, dim_val]
#         """

#         x = x + self.pe[:x.size(self.x_dim)]

#         return self.dropout(x)




__all__ = ['PositionalEncoding', 'SinCosPosEncoding', 'positional_encoding']

# Cell

import torch
from torch import nn
import math

# Cell
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)