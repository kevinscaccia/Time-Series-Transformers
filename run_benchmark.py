# python
import argparse
import numpy as np
import pandas as pd
import json
# ml
from sklearn.preprocessing import MinMaxScaler
# local
from models.benchmark import NaivePredictor
from models.cnn import SimpleCNN
from utils.plot import plot_predictions
from experiment import Experiment
from utils.m4 import smape, mase, M4DatasetGenerator
from utils.ml import print_num_weights

# Neural models
from neuralforecast.core import NeuralForecast
from neuralforecast.models import Informer, Autoformer, FEDformer, PatchTST, NBEATS, NHITS
#
# Parse args
#
parser = argparse.ArgumentParser(prog='M4Benchmark', description='Benchmark on M4 Dataset')
parser.add_argument('--model', required=True)
parser.add_argument('--freq', required=True)
args = parser.parse_args()
model_name = args.model
run_sp = args.freq
assert(model_name in ['CNN','Naive','Informer'])
assert run_sp in ['Hourly','Daily','Weekly','Monthly','Quarterly','Yearly']

prediction_frequency = {'Hourly':'h','Weekly':'W',}[run_sp]

np.random.seed(123)


def get_model(model_name, model_conf):
    if model_name == 'CNN':
        return SimpleCNN(model_conf['block_size'], model_dim=64)
    elif model_name == 'Naive':
        return NaivePredictor()
    elif model_name == 'Informer':
        hidden_size = 64
        n_head = 4
        conv_hidden_size = 8
        learning_rate = 0.001
        max_steps = 100
        batch_size = 1024
        return Informer(hidden_size=hidden_size, n_head=n_head, conv_hidden_size=conv_hidden_size,
                            input_size=model_conf['input_size'], # Input size
                            h=model_conf['forecasting_horizon'], # Forecasting horizon
                            max_steps=max_steps, # Number of training iterations
                            batch_size=batch_size, learning_rate=learning_rate, )

metrics_table = {'serie_id':[],'smape':[],'mase':[],}
smape_list, mase_list = [], []
m4_data = M4DatasetGenerator([run_sp])
num_of_series = m4_data.data_dict[run_sp]['num']
block_size = m4_data.data_dict[run_sp]['fh']
fh = m4_data.data_dict[run_sp]['fh']
for train_serie, test_serie, serie_id, fh, freq, serie_sp in m4_data.generate(random=False):
    assert fh == block_size
    # synthetic days
    train_daterange = pd.date_range(start='1980', periods=len(train_serie), freq=prediction_frequency)
    test_daterange = pd.date_range(start=train_daterange[-1], periods=len(test_serie)+1, freq=prediction_frequency)[1:] # len + 1 because the first day is on train dates
    #
    model_conf = {}
    model_conf['input_size'] = min(fh*4, len(train_serie)//10)
    model_conf['forecasting_horizon'] = fh

    model = get_model(model_name, model_conf)
    print_num_weights(model)
    
    if model_name in ['Informer','autoformer','fedformer','patchtst','NHITS']:
        train_daterange = pd.date_range(start='1980', periods=len(train_serie), freq=prediction_frequency)
        test_daterange = pd.date_range(start=train_daterange[-1], periods=len(test_serie)+1, freq=prediction_frequency)[1:] # len + 1 because the first day is on train dates
        nf = NeuralForecast(models=[model], freq=prediction_frequency, local_scaler_type='standard')
        train_df = pd.DataFrame({
            'unique_id':serie_id,
            'y':train_serie, 
            'ds':train_daterange
            })
        val_size = 0#int(.1 * len(train_serie)) # 20% for validation

        # model train
        nf.fit(df=train_df, val_size=val_size, verbose=False)
        pred_y = nf.predict()
        #
        assert all(pred_y.ds == test_daterange) # check 
        pred_y = pred_y[model_name].values
    else:
        exp_conf = {
                # Model
                'model': model,
                'model_n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 
                'input_len':block_size,
                'forecast_horizon':fh,
                'feature_dim':1,
                # Data
                'frequency':serie_sp.lower(),
                'scaler': MinMaxScaler((-1,1)),
                'decompose': False, #detrend and de-sazonalize
                'freq':freq,
                # Others
                'device':'cuda',
                'verbose':False,
        }
        train_conf = {
            'epochs':512,
            'lr':1e-3, 
            'batch_size':512,
            'validate_freq':10,
            'verbose':False,
        }
        exp = Experiment(exp_conf)
        exp.set_dataset(linear_serie=train_serie, train=True)
        # exp.set_dataset(linear_serie=test_serie)
        exp.train(train_conf)
        # test
        last_train_values = train_serie[-block_size:]
        pred_y = exp.predict(last_train_values, fh)
    
    # check if negative or extreme (M4)
    for i in range(len(pred_y)):
        if pred_y[i] < 0:
            pred_y[i] = 0
                
        if pred_y[i] > (1000 * max(train_serie)):
            pred_y[i] = max(train_serie)
    # Metrics
    serie_smape = smape(test_serie, pred_y)*100
    serie_mase = mase(train_serie, test_serie, pred_y, freq)
    metrics_table['serie_id'].append(serie_id)
    metrics_table['smape'].append(serie_smape)
    metrics_table['mase'].append(serie_mase)
    print(f'Serie {serie_id}-{serie_sp} Finished')
#
metrics_dict = {
    'smape_mean': np.round(np.mean(metrics_table['smape'], dtype=float), 3), 
    'mase_mean':  np.round(np.mean(metrics_table['mase'], dtype=float), 3),
    #
    'smape_std':  np.round(np.std(metrics_table['smape'], dtype=float), 3),
    'mase_std':   np.round(np.std(metrics_table['mase'], dtype=float), 3),
}
json.dump(metrics_table, open(f'./results/{run_sp}_{model_name}_metrics_table.json','w'))
print(f'''
    Experiment Finished
''')
for k, v in metrics_dict.items(): print(f'      {k}: {v}')