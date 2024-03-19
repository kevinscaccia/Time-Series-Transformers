import numpy as np
from torch import nn

def naive_predict(train_y, test_y, fh):
    pred_y = np.asarray([train_y[-1]]*fh) 
    return pred_y


class NaivePredictor(nn.Module):
    def __init__(self,):
        super(NaivePredictor, self).__init__()
    
    def is_transformer(self,):
        return False

    def __call__(self, x):
        return x[:,-1,:] # last timestep
    
    def fit(self, conf):
        pass

    def predict(self, x, forecast_horizon):
        y = self(x).repeat((1, forecast_horizon)).unsqueeze(-1)
        # print(y)
        # print(y.shape)
        # raise Exception
        return y