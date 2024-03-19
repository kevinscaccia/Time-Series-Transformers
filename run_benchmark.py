# python
import argparse
import numpy as np
# ml
from sklearn.preprocessing import MinMaxScaler
import mlflow
# local
from models.benchmark import NaivePredictor
from models.cnn import SimpleCNN
from utils.plot import plot_predictions
from experiment import Experiment
from utils.m4 import smape, mase, M4DatasetGenerator
#
# Parse args
#
parser = argparse.ArgumentParser(prog='M4Benchmark', description='Benchmark on M4 Dataset')
parser.add_argument('--model', required=True)
parser.add_argument('--freq', required=True)
args = parser.parse_args()
model_name = args.model
run_sp = args.freq
assert(model_name in ['cnn','naive'])
assert run_sp in ['Hourly','Daily','Weekly','Monthly','Quarterly','Yearly']
#
#
#
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment(f"M4Benchmark {model_name}")
mlflow.set_experiment_tag('model', model_name)

np.random.seed(123)

def get_model(model_name, model_conf):
    if model_name == 'cnn':
        return SimpleCNN(model_conf['block_size'], model_conf['d_model'])
    elif model_name == 'naive':
        return NaivePredictor()


metrics_table = {'serie_id':[],'smape':[],'mase':[],}
smape_list, mase_list = [], []
m4_data = M4DatasetGenerator([run_sp])
num_of_series = m4_data.data_dict[run_sp]['num']
block_size = m4_data.data_dict[run_sp]['fh']
fh = m4_data.data_dict[run_sp]['fh']
#
d_model = 64
batch_size = 512
epochs = 512
scaler = MinMaxScaler((-1,1))
decompose = False
lr = 1e-3
#
model_conf = {'block_size':block_size, 'd_model':d_model}
#
#
with mlflow.start_run(run_name=f'{run_sp}'):
    mlflow.log_param('model_name', model_name)
    mlflow.log_param('d_model', d_model)
    mlflow.log_param('block_size', block_size)
    mlflow.log_param('forecast_horizon', fh)
    mlflow.log_param('decompose', decompose)

    mlflow.log_param('series', num_of_series)
    mlflow.log_param('scaler', scaler)
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('epochs', epochs)
    mlflow.log_param('lr', lr)
    
    for train_serie, test_serie, serie_id, fh, freq, serie_sp in m4_data.generate(random=False):
        assert fh == block_size
        model = get_model(model_name, model_conf) # VanillaTransformer({'d_model': 16, 'block_size':input_len,'num_heads': 4, 'num_layers': 4,'dim_feedforward':128,'device':'cuda'}),
        
        exp_conf = {
                # Model
                'model': model,
                'model_n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 
                'input_len':block_size,
                'forecast_horizon':fh,
                'feature_dim':1,
                # Data
                'frequency':serie_sp.lower(),
                'scaler':scaler,
                'decompose': decompose, #detrend and de-sazonalize
                'freq':freq,
                # Others
                'device':'cuda',
                'verbose':False,
        }
        train_conf = {
            'epochs':epochs,
            'lr':lr, 
            'batch_size':batch_size,
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

        # pred_y[pred_y < 0] = 0
        # # Substitua os valores maiores que 1000 vezes o máximo de train_serie pelo máximo de train_serie
        # pred_y[pred_y > (1000 * np.max(train_serie))] = np.max(train_serie)

        # Metrics
        metrics_table['serie_id'].append(serie_id)
        metrics_table['smape'].append(smape(test_serie, pred_y)*100)
        metrics_table['mase'].append(mase(train_serie, test_serie, pred_y, freq))
        print(f'Serie {serie_id}-{serie_sp} Finished')
    #
    metrics_dict = {
        'smape_mean': np.round(np.mean(metrics_table['smape'], dtype=float), 3), 
        'mase_mean':  np.round(np.mean(metrics_table['mase'], dtype=float), 3),
        #
        'smape_std':  np.round(np.std(metrics_table['smape'], dtype=float), 3),
        'mase_std':   np.round(np.std(metrics_table['mase'], dtype=float), 3),
    }
    mlflow.log_metrics(metrics_dict)
    mlflow.log_table(metrics_table, artifact_file='metrics_table')

print(f'''
    Experiment Finished
''')
for k, v in metrics_dict.items(): print(f'      {k}: {v}')