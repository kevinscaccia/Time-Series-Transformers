import torch
from torch import nn
from torch.utils.data import DataLoader
import time


class SimpleCNN(nn.Module):
    def __init__(self, input_len, model_dim=64):
        super(SimpleCNN, self).__init__()
        self.conv1d = nn.Conv1d(input_len, model_dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(model_dim, 50)
        self.fc2 = nn.Linear(50, 1)
        #
        self.train_loss_history = []
        self.validation_loss_history = []
        
    def forward(self,x):
        # cnn filter
        x = self.relu(self.conv1d(x))
        # bridge between cnn and dnn
        x = self.flatten(x)
        # dense 1
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def is_transformer(self,):
        return False
    
    def fit(self, conf):
        expected_vars = ['epochs','lr','batch_size','train_dataset']
        for v in expected_vars:
            assert v in conf.keys(), f'Key "{v}" is missing on params dict'
        #
        
        epochs = conf['epochs']
        verbose = conf['verbose']
        train_dataset = conf['train_dataset']
        val_dataset = conf.get('train_dataset',None)
        early_stop = conf.get('early_stop', None)
        #
        optimizer = torch.optim.AdamW(self.parameters(), lr=conf['lr'])
        loss_fn = nn.MSELoss()
        train_loader = DataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=False)
        
        for epoch_i in range(epochs):
            timr = time.time()
            epoch_loss = .0
            val_loss = -1
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad() # current batch zero-out the loss
                pred_y = self(batch_x)
                loss = loss_fn(pred_y, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            # end epoch
            epoch_loss = epoch_loss/len(train_loader)
            if early_stop is not None:
                if early_stop.step(epoch_loss):
                    print(f'    Early Stopping on epoch {epoch_i}')
                    break
            # if there are a validation set
            if val_dataset is not None:
                val_loss = self.validate(val_dataset)
                self.validation_loss_history.append(val_loss.to('cpu').detach().numpy())
            self.train_loss_history.append(epoch_loss.to('cpu').detach().numpy())
            #     
            timr = time.time() - timr
            if verbose: 
                print(f'Epoch {epoch_i+1}/{epochs} [{timr:.3f}secs] -> Train loss: {epoch_loss:.5f} | Validation loss: {val_loss:.5f}')
    
    def validate(self, data):
        val_loader = DataLoader(data, batch_size=1024, shuffle=False)
        loss_fn = nn.MSELoss()
        val_loss = 0.0
        #
        self.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred_y = self(batch_x)
                val_loss += loss_fn(pred_y, batch_y)
        val_loss = val_loss/len(val_loader)
        return val_loss
    
    def predict(self, ts, forecast_horizon):
        self.eval()
        with torch.no_grad():
            output = torch.zeros(forecast_horizon)
            for i in range(forecast_horizon):
                y = self(ts)
                output[i] = y.flatten()
                ts = torch.concat((ts[:,1:,:], y.unsqueeze(-1)), 1)
        return output