import numpy as np
import torch
from torch import nn
import math
import time
from utils.plot import generate_square_subsequent_mask
from torch.utils.data import DataLoader
#
# from layers import PositionalEncoding

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
        self.train_loss_history = []
        self.validation_loss_history = []
        #
        # Encoder layers
        #
        self.encoder_input_layer = nn.Linear(in_features=self.in_features, out_features=self.d_model)# embedding 
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=512)
        

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
        
        # # positional encoding 
        y = self.positional_encoding(y)
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
        y = self.positional_encoding(y)
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
    
    def is_transformer(self,):
        return True
    
    def predict(self, src, forecast_horizon):
        self.eval()
        with torch.no_grad():
            output = torch.zeros(1, forecast_horizon + 1, 1).to('cuda')
            output[0, 0, 0] = src[0, -1] # first value
            for i in range(forecast_horizon):
                dim_a = output.shape[1]
                tgt_mask = generate_square_subsequent_mask(dim_a, dim_a).to('cuda')
                y = self(src, output, None, tgt_mask)[0,i,0]
                output[0,i+1,0] = y
        return output[:,1:,:] # remove first value (copy from last history step)

    def validate(self, data):
        val_loader = DataLoader(data, batch_size=1024, shuffle=False)
        loss_fn = nn.MSELoss()
        val_loss = 0.0
        #
        memory_mask, tgt_mask = self.get_train_masks()
        memory_mask, tgt_mask = memory_mask.to('cuda'), tgt_mask.to('cuda')
        self.eval()
        with torch.no_grad():
            for enc_x, dec_x, tgt_y in val_loader:
                pred_y = self(enc_x, dec_x, memory_mask, tgt_mask)
                val_loss += loss_fn(pred_y, tgt_y)
        val_loss = val_loss/len(val_loader)
        return val_loss

    




# Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html,
# only modified to account for "batch first".
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the given tensor.

        Args:
            x: tensor to add PE to [bs, seq_len, embed_dim]

        Returns:
            torch.Tensor: tensor with PE [bs, seq_len, embed_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

import numpy as np
import torch
from torch import nn
import math, time, tqdm
from utils.plot import generate_square_subsequent_mask
from torch.utils.data import DataLoader

class DecoderOnlyTransformer(torch.nn.Module):
    def __init__(self, model_params: dict):
        super(DecoderOnlyTransformer, self).__init__()
        # Set model vars
        expected_vars = ['d_model','block_size','num_heads','num_layers','dim_feedforward','device','pad_token']
        for v in expected_vars:
            assert v in model_params.keys(), f'Key "{v}" is missing on params dict'
            vars(self)[v] = model_params[v]
        
        self.train_loss_history = []
        self.validation_loss_history = []

        self.pos_emb = nn.Embedding(num_embeddings=self.block_size, embedding_dim=self.d_model)
        self.decoder_embedding = torch.nn.Linear(in_features=1, out_features=self.d_model)
        self.output_layer = torch.nn.Linear(in_features=self.d_model, out_features=1)
        # self.scaler = nn.BatchNorm1d(num_features=self.block_size)

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model, 
                nhead=self.num_heads, 
                dim_feedforward=self.dim_feedforward,
                dropout=0.1, batch_first=True, device=self.device), 
        self.num_layers, norm=None)
    
    def is_transformer(self, ): return True

    def forward(self, src: torch.Tensor, mask=None, pad_mask=None) -> torch.Tensor:
        src = self.decoder_embedding(src) + self.pos_emb(torch.arange(src.shape[1]).to(self.device))# (B, T) --> (B, T, Emb) 
        pred = self.decoder(src, mask=mask, src_key_padding_mask=pad_mask)
        pred = self.output_layer(pred)
        return pred
    
    @torch.no_grad()
    def validate(self, data):
        self.eval()
        val_loader = DataLoader(data, batch_size=1024, shuffle=False)
        loss_fn = nn.MSELoss()
        val_loss = 0.0
        #
        with torch.no_grad():
            for enc_x, dec_x, tgt_y in val_loader:
                pred_y = self(enc_x, dec_x)
                val_loss += loss_fn(pred_y, tgt_y)
        val_loss = val_loss/len(val_loader)
        self.train()
        return val_loss
    
    @torch.no_grad()
    def predict(self, src, forecast_horizon):
        self.eval()
        src = src[:, -self.block_size:, :].clone()
        h_len = src.shape[1]
        
        for i in range(forecast_horizon):
            x = src[:, -self.block_size:, :]
            y = self(x)[:, -1:, :]
            src = torch.concat((src, y), dim=1)
        return src[:, h_len:, :]
    
    def fit(self, conf):
        self.train()
        expected_vars = ['epochs','lr','batch_size','train_dataset',]
        for v in expected_vars:
            assert v in conf.keys(), f'Key "{v}" is missing on params dict'
        #
        epochs = conf['epochs']
        verbose = conf['verbose']
        batch_size = conf['batch_size']
        train_dataset = conf['train_dataset']
        val_dataset = conf.get('train_dataset',None)
        validate_freq = conf.get('validate_freq',100)
        early_stop = conf.get('early_stop', None)
        epoch_offset = conf.get('epoch_offset', 0)
        #
        optimizer = torch.optim.AdamW(self.parameters(), lr=conf['lr'])
        loss_fn = nn.MSELoss(reduction='none')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if verbose: print(f'Starting train. {len(train_loader)} batches {len(train_dataset)}/{batch_size}')
        for epoch_i in range(epochs):
            epoch_i += epoch_offset
            timr = time.time()
            epoch_loss = .0
            val_loss = -1
            for enc_x, tgt_y, mask_x in train_loader:#tqdm.tqdm(train_loader):
                enc_x, tgt_y, mask_x  = enc_x.to(self.device), tgt_y.to(self.device), mask_x.to(self.device)
                optimizer.zero_grad() # current batch zero-out the loss
                pred_y = self(enc_x, pad_mask=mask_x) # mask x is very very important!!!
                loss = loss_fn(pred_y, tgt_y) # loss with padding
                loss = loss.where((tgt_y != self.pad_token), torch.tensor(0.0)).mean() # mask pading in the loss!
                loss.backward()
                optimizer.step()
                epoch_loss += loss        
            # end epoch
            epoch_loss = epoch_loss/len(train_loader)
            self.train_loss_history.append(epoch_loss.to('cpu').detach().numpy())
            if early_stop is not None:
                if early_stop.step(epoch_loss): break
            torch.save(self, f'decoder_only_weekly_{epoch_i}.model')
            to_validate = False#(val_dataset is not None) and (epoch_i % validate_freq == 0)
            if verbose:
                if to_validate: 
                    val_loss = self.validate(val_dataset)
                    self.validation_loss_history.append(val_loss.to('cpu').detach().numpy())
                    timr = time.time() - timr
                    # torch.save(self, f'former_{epoch_i}.model')
                    print(f'Epoch {epoch_i+1}/{epochs} [{timr:.3f}secs] -> Train loss: {epoch_loss:.5f} | Validation loss: {val_loss:.5f}')
                else: 
                    timr = time.time() - timr
                    print(f'Epoch {epoch_i+1}/{epochs} [{timr:.3f}secs] -> Train loss: {epoch_loss:.5f}')


        
class VanillaTransformer(torch.nn.Module):
    def __init__(self, model_params: dict):
        super(VanillaTransformer, self).__init__()
        # Set model vars
        expected_vars = ['d_model','block_size','num_heads','num_layers','dim_feedforward','device']
        for v in expected_vars:
            assert v in model_params.keys(), f'Key "{v}" is missing on params dict'
            vars(self)[v] = model_params[v]
        
        self.train_loss_history = []
        self.validation_loss_history = []

        self.pos_emb = nn.Embedding(num_embeddings=self.block_size, embedding_dim=self.d_model)

        self.encoder_embedding = torch.nn.Linear(in_features=1, out_features=self.d_model)
        self.decoder_embedding = torch.nn.Linear(in_features=1, out_features=self.d_model)
        self.output_layer = torch.nn.Linear(in_features=self.d_model, out_features=1)
        # self.scaler = nn.BatchNorm1d(num_features=self.block_size)

        self.transformer = torch.nn.Transformer(
            nhead=self.num_heads,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
        )
        self.mask = True
    def is_transformer(self, ): return True

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward function of the model.

        Args:
            src: input sequence to the encoder [bs, src_seq_len, num_features]
            tgt: input sequence to the decoder [bs, tgt_seq_len, num_features]

        Returns:
            torch.Tensor: predicted sequence [bs, tgt_seq_len, feat_dim]
        """
        B, T, C = src.shape
        # src = self.scaler(src)
        # tgt = self.scaler(tgt)

        src = self.encoder_embedding(src)
        src = src +  self.pos_emb(torch.arange(src.shape[1]).to(self.device))# (B, T) --> (B, T, Emb) 
        # Generate mask to avoid attention to future outputs.[tgt_seq_len, tgt_seq_len]
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        # Embed decoder input and add positional encoding. [bs, tgt_seq_len, embed_dim]
        tgt = self.decoder_embedding(tgt)
        tgt = tgt +  self.pos_emb(torch.arange(tgt.shape[1]).to(self.device))

        # Get prediction from transformer and map to output dimension.[bs, tgt_seq_len, embed_dim]
        pred = self.transformer(src, tgt, tgt_mask=tgt_mask)
        pred = self.output_layer(pred)
        return pred
    
    @torch.no_grad()
    def validate(self, data):
        self.eval()
        val_loader = DataLoader(data, batch_size=1024, shuffle=False)
        loss_fn = nn.MSELoss()
        val_loss = 0.0
        #
        with torch.no_grad():
            for enc_x, dec_x, tgt_y in val_loader:
                pred_y = self(enc_x, dec_x)
                val_loss += loss_fn(pred_y, tgt_y)
        val_loss = val_loss/len(val_loader)
        self.train()
        return val_loss
    
    @torch.no_grad()
    def predict(self, src, forecast_horizon):
        self.eval()
        src = src[:, -self.block_size:, :]
        output = src[0, -1].clone().view(1, 1, 1) # first value
        for i in range(forecast_horizon):
            y_next = self(src, output[:, -self.block_size:, :])
            y_next = y_next[:, -1:, :]
            # concatenate in each batch along the time dimension
            output = torch.concat((output, y_next), dim=1)
        return output[:, 1: ,: ] # remove first value (copy from last history step)
    
    # @torch.no_grad()
    # def predict(self, src, forecast_horizon):
    #     self.eval()
    #     output = torch.zeros(1, forecast_horizon + 1, 1).to('cuda')
    #     output[0, 0, 0] = src[0, -1] # first value
    #     for i in range(forecast_horizon):
    #         y = self(src, output[:, -self.block_size:, :])[0,i,0]
    #         output[0,i+1,0] = y
    #     return output[:,1:,:] # remove first value (copy from last history step)

    def fit(self, conf):
        self.train()
        expected_vars = ['epochs','lr','batch_size','train_dataset',]
        for v in expected_vars:
            assert v in conf.keys(), f'Key "{v}" is missing on params dict'
        #
        epochs = conf['epochs']
        verbose = conf['verbose']
        train_dataset = conf['train_dataset']
        val_dataset = conf.get('train_dataset',None)
        validate_freq = conf.get('validate_freq',100)
        early_stop = conf.get('early_stop', None)
        #
        optimizer = torch.optim.AdamW(self.parameters(), lr=conf['lr'])
        loss_fn = nn.MSELoss()
        train_loader = DataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=True)
        for epoch_i in range(epochs):
            timr = time.time()
            epoch_loss = .0
            val_loss = -1
            for enc_x, dec_x, tgt_y in train_loader:
                optimizer.zero_grad() # current batch zero-out the loss
                pred_y = self(enc_x, dec_x)
                loss = loss_fn(pred_y, tgt_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss        
            # end epoch
            epoch_loss = epoch_loss/len(train_loader)
            self.train_loss_history.append(epoch_loss.to('cpu').detach().numpy())
            if early_stop is not None:
                if early_stop.step(epoch_loss): break
            
            to_validate = (val_dataset is not None) and (epoch_i % validate_freq == 0)
            if verbose:
                if to_validate: 
                    val_loss = self.validate(val_dataset)
                    self.validation_loss_history.append(val_loss.to('cpu').detach().numpy())
                    timr = time.time() - timr
                    # torch.save(self, f'former_{epoch_i}.model')
                    print(f'Epoch {epoch_i+1}/{epochs} [{timr:.3f}secs] -> Train loss: {epoch_loss:.5f} | Validation loss: {val_loss:.5f}')
                else: 
                    timr = time.time() - timr
                    print(f'Epoch {epoch_i+1}/{epochs} [{timr:.3f}secs] -> Train loss: {epoch_loss:.5f}')
        # torch.save(self, f'former_last.model')