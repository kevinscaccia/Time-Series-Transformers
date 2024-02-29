import numpy as np
import torch
from torch import nn
from utils.plot import generate_square_subsequent_mask
#
from layers import Time2Vec, PositionalEncoding

class TimeSeriesTransformer(nn.Module):
    def __init__(self, model_params: dict):
        super(TimeSeriesTransformer, self).__init__()
        # Set model vars
        expected_vars = ['in_features','d_model','input_len',
                         'encoder_nheads','encoder_nlayers','encoder_dropout',
                         'decoder_nheads','decoder_nlayers','decoder_dropout',
                         'feedforward_dim','forecast_horizon','seed','mapping_dim']
        for v in expected_vars:
            assert v in model_params.keys(), f'Key "{v}" is missing on params dict'
            vars(self)[v] = model_params[v]
        #
        # Pre-configuration (to produce same result in inference/predict)
        #
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        #
        # Encoder layers
        #
        self.encoder_input_layer = nn.Linear(in_features=self.in_features, out_features=self.d_model)
        self.time2vec = PositionalEncoding(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=self.d_model,
                            nhead=self.encoder_nheads,
                            dim_feedforward=self.feedforward_dim,
                            dropout=self.encoder_dropout,
                            batch_first=True)
        self.encoder_block = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.encoder_nlayers, norm=None)
        #
        # Decoder layers
        #
        self.decoder_input_layer = nn.Linear(in_features=self.in_features, out_features=self.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
                            d_model=self.d_model, 
                            nhead=self.decoder_nheads,
                            dim_feedforward=self.feedforward_dim,
                            dropout=self.decoder_dropout,
                            batch_first=True
                            )
        self.decoder_block = nn.TransformerDecoder(decoder_layer=decoder_layer,num_layers=self.decoder_nlayers, norm=None)
        # self.decoder_dense_mapping = nn.Linear(self.d_model, self.mapping_dim)        
        self.decoder_dense_mapping = nn.Linear(self.d_model, self.in_features)
        # self.decoder_dense_mapping2 = nn.Linear(self.mapping_dim, self.in_features)
        # self.init_weights()
    
    def get_train_masks(self,):
        # which will mask the encoder output
        memory_mask = generate_square_subsequent_mask(self.forecast_horizon, self.input_len)
        # which will mask the decoder input
        tgt_mask = generate_square_subsequent_mask(self.forecast_horizon, self.forecast_horizon)
        return memory_mask, tgt_mask

    def init_weights(self):
        initrange = 0.1    
        # self.decoder_dense_mapping.bias.data.zero_()
        self.decoder_dense_mapping.weight.data.uniform_(-initrange, initrange)
        # self.decoder_dense_mapping2.weight.data.uniform_(-initrange, initrange)

    def encode(self, x, verbose):
        if verbose: print(f'#1) Encoder input shape = {x.shape}')
        
        # linear transformation (embedding)
        y = self.encoder_input_layer(x)
        
        if verbose: print(f'#2) Encoder embedding layer output: {y.shape}')
        
        # positional encoding 
        y = self.time2vec(y)
        if verbose: print(f'#3) Positional encoding output: {y.shape}')

        # encoder block 
        y = self.encoder_block(y)
        if verbose: print(f'#4) Encoder block output: {y.shape}\n')
        return y
    
    def decode(self, x, enc_y,  memory_mask, tgt_mask, verbose):
        if verbose: print(f'#5) Decoder input shape = {x.shape}')
        y = self.decoder_input_layer(x)
        if verbose: print(f'#6) Decoder input layer output: {y.shape}')
        
        # positional encoding 
        y = self.time2vec(y)
        if verbose: print(f'#7) Positional encoding output: {y.shape}')
        
        # Pass decoder input through decoder input layer
        y = self.decoder_block(
            tgt=y,
            memory=enc_y,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        if verbose: print(f'#8) Decoder block output: {y.shape}')

        # mapping (dense)
        y = self.decoder_dense_mapping(y)
        # y = self.decoder_dense_mapping2(y)
        if verbose: print(f'#9) Decoder mapping(dense) output: {y.shape}')

        return y

    def forward(self, enc_x, dec_x, memory_mask=None, tgt_mask=None, verbose=False):
        enc_y = self.encode(enc_x, verbose)
        dec_y = self.decode(dec_x, enc_y, memory_mask, tgt_mask, verbose)
        return dec_y

    