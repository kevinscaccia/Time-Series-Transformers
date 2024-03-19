import sys
sys.path.append('/workspace/Time-Series-Transformers/')

from experiment import Experiment
from utils.m4 import (smape, mase, get_error_dict, M4DatasetGenerator)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#
# Models
#
import torch
from models.benchmark import NaivePredictor
from models.cnn import SimpleCNN
from utils.plot import plot_predictions

used_freq = ['Hourly','Daily','Weekly','Monthly','Quarterly','Yearly']
m4_data = M4DatasetGenerator(used_freq)

model_name = 'cnn'
ERRORS = {model_name:get_error_dict(used_freq)}
np.random.seed(777)
print(f'''
        Benchmark Start model '{model_name}'
      ''')
for train_serie, test_serie, serie_id, fh, freq, serie_sp in m4_data.generate(1, random=False):
    input_len = 24*4
    fh = len(test_serie)
    exp = Experiment(
        {
            # Model
            'model': SimpleCNN(input_len, model_dim=64),
            'input_len':input_len,
            'forecast_horizon':fh,
            'feature_dim':1,
            # Data
            'frequency':serie_sp.lower(),
            'scaler':MinMaxScaler((-1,1)),
            'decompose': False, #detrend and de-sazonalize
            'freq':freq,
            # Others
            'device':'cuda',
            'verbose':False,
        })
    exp.set_dataset(linear_serie=train_serie, train=True)
    exp.set_dataset(linear_serie=test_serie)
    exp.train({
        'epochs':1024,
        'validate_freq':1,
        'lr':1e-3,#1e-5,
        'batch_size':512,
        'verbose':False,
    })
    
    last_train_values = train_serie[-input_len:]
    next_validation_values = test_serie[:fh]
    pred_y = exp.predict(last_train_values, fh)
    #
    exp.print_metrics(next_validation_values, pred_y[:fh])
    #+
    plot_predictions(train_serie, next_validation_values, pred_y)
    exp.train_history(offset=100)
    # Vanila TF
    ERRORS[model_name][serie_sp]['sMAPE'].append(smape(test_serie, pred_y))
    ERRORS[model_name][serie_sp]['MASE'].append(mase(train_serie[:-fh], test_serie, pred_y, freq))  

# print("---------FINAL RESULTS---------")
for model, err in ERRORS.items():
    print(f'Model : {model}')
    for sp, sp_err in err.items():
        print(f'  {sp}: ')
        print(f'    sMAPE: {np.mean(sp_err["sMAPE"])*100:.3f}')
        print(f'    MASE: {np.mean(sp_err["MASE"]):.3f}')
  