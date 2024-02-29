import numpy as np
import time
import torch
from torch import nn
from utils.timeserie import split_sequence
from torch.utils.data import DataLoader
#
from utils.ml import SimpleDataset

class Experiment():

    def __init__(self, config: dict):
        # Set experiment config
        expected_vars = ['model','input_len','feature_dim','frequency',
                         'device','scaler','verbose']
        for v in expected_vars:
            assert v in config.keys(), f'Key "{v}" is missing on params dict'
            vars(self)[v] = config[v]
        self.config = config
        #
        # Pre-configuration (to produce same result in inference/predict)
        #
        np.random.seed(7); torch.manual_seed(7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(7)
        #
        #
        #
        self.model = self.model.to(self.device)



    def split_chunks(self, linear_serie, expand_dim=True):
        x, y = split_sequence(linear_serie, self.input_len)
        x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        if expand_dim:
            x, y = x.unsqueeze(-1), y.unsqueeze(-1)
        return x, y

    def set_dataset(self, linear_serie, train=False, validation=False):
        if self.scaler is not None:
            if train: # FIT Scaler
                if self.verbose: print('Scaler FIT')
                linear_serie = self.scaler.fit_transform(linear_serie.reshape(-1,1)).reshape(-1)
            if validation:
                linear_serie = self.scaler.transform(linear_serie.reshape(-1,1)).reshape(-1)

            
        x, y = self.split_chunks(linear_serie)
        x, y = x.to(self.device), y.to(self.device)
        data = SimpleDataset(x, y)
        # Save
        if train:
            self.train_dataset = data
        if validation:
            self.validation_dataset = data
        
        return data
    

    def train(self, train_conf):
        expected_vars = ['epochs','lr','batch_size']
        for v in expected_vars:
            assert v in train_conf.keys(), f'Key "{v}" is missing on params dict'
        #
        epochs = train_conf['epochs']
        verbose = train_conf['verbose']
        #
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_conf['lr'])
        loss_fn = nn.MSELoss()
        train_loader = DataLoader(self.train_dataset, batch_size=train_conf['batch_size'], shuffle=False)
        
        loss_history = []
        for epoch_i in range(epochs):
            timr = time.time()
            epoch_loss = .0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad() # current batch zero-out the loss
                pred_y = self.model(batch_x)
                loss = loss_fn(pred_y, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            # end epoch
            epoch_loss = epoch_loss/len(train_loader)
            loss_history.append(epoch_loss.to('cpu').detach().numpy())
            timr = time.time() - timr
            if verbose: print(f'Epoch {epoch_i+1}/{epochs} [{timr:.3f}secs] -> Train loss: {epoch_loss:.5f}')
    
    def predict(self, linear_serie):
        if self.scaler is not None:
            linear_serie = self.scaler.transform(linear_serie.reshape(-1,1)).reshape(-1)
