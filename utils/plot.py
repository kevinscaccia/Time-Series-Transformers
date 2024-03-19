import numpy as np
import matplotlib.pyplot as plt
import math

import torch
from torch import nn, Tensor


def print_losses(losses, offset=0):
    last_l = losses[-1]
    min_l = np.min(losses)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(offset, len(losses)), losses[offset:], label=f'Min = {min_l:.6} | Last = {last_l:.6}', color='red')
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()



def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Source:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


#
# https://github.com/AIStream-Peelout/flow-forecast/blob/master/flood_forecast/transformer_xl/transformer_basic.py
#
class SimplePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(SimplePositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a basic positional encoding"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what 
    the positional encoding layer does and why it is needed:
    
    "Since our model contains no recurrence and no convolution, in order for the 
    model to make use of the order of the sequence, we must inject some 
    information about the relative or absolute position of the tokens in the 
    sequence." (Vaswani et al, 2017)
    Adapted from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self, 
        d_model: int=512,
        dropout: float=0.1, 
        max_seq_len: int=5000, 
        batch_first: bool=True
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if batch_first else 0

        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_seq_len, 1, d_model)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """

        x = x + self.pe[:x.size(self.x_dim)]

        return self.dropout(x)



import plotly.graph_objects as go
THEME = 'plotly_dark'
def plot_predictions(history, real_serie, pred_y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0, len(history)), y=history, mode='lines', 
                              name=f'History', line=dict(color='#0099ff'), hovertemplate='$%{y:.2f}')) # past values
    
    fig.add_trace(go.Scatter(x=np.arange(len(history), len(history)+len(real_serie)), y=real_serie, mode='lines', 
                              name=f'Future', line=dict(color='yellow'), hovertemplate='$%{y:.2f}')) # real future
    
    fig.add_trace(go.Scatter(x=np.arange(len(history), len(history)+len(pred_y)), y=pred_y, mode='lines', 
                            name='Forecasted', line=dict(color='red'), hovertemplate='$%{y:.2f}')) # predicted 
    
    # Layout and configurations
    config = {
        'template':THEME,
        'hovermode':'x unified',
        'xaxis_rangeselector_font_color':'black',
        'legend':dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=0.9),
        }
    fig.update_layout(config)
    fig.layout.yaxis.fixedrange = True # block vertical zoom
    fig.show()


# def plot_predictions(real_serie, pred_y):
#     plt.figure(figsize=(18,5))
#     plt.plot(real_serie, 'g', label='Real')
#     plt.plot(pred_y, 'r', label='Prediction')
#     plt.legend()
#     plt.show()

def plot_serie(serie):
    plt.figure(figsize=(10, 6))
    plt.plot(serie, label='Time Serie', color='green')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_train_history(train_l, val_l=None, offset=0,):
    plt.plot(np.arange(offset+1,len(train_l)+1),train_l[offset:],'r', label='Train')
    if val_l is not None:
        plt.plot(np.arange(offset+1,len(train_l)+1),val_l[offset:],'b', label='Validation')
    plt.legend()
    plt.show()

    