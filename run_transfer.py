# python
import numpy as np
import numpy as np
import torch
# local
from models.transformer import DecoderOnlyTransformer
from utils.m4 import MultiSerieGenerator
from utils.ml import DecoderDataset

TRACK = False
PAD = -20
model_name ='decoder_transformer'
run_sp = 'Weekly'

#
# Inicializations
#
block_size = 512
n_series = 500
#
# Model Hiperparams
#
model = DecoderOnlyTransformer({
    'd_model': 32, 
    'num_heads': 4, 
    'num_layers': 4,
    'dim_feedforward':128,
    'block_size':block_size,
    'device':'cuda',
    'pad_token':PAD
}).to('cuda')
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Num of weights:',pytorch_total_params)
train_dataset = DecoderDataset(*MultiSerieGenerator(['Weekly'], device='cuda',verbose=True).get_batches(block_size, n_series))
#
#
epoch_offset = 20
model = torch.load(f'trained/decoder_only_weekly_{epoch_offset}.model')
epoch_offset += 1 
#
#
batch_size = 800
epochs = 100
lr = 1e-3
train_conf = {
    'epochs':epochs,
    'epoch_offset':epoch_offset,
    'lr':lr, 
    'batch_size':batch_size,
    'verbose':True,
    'train_dataset':train_dataset
}
model.fit(train_conf) 