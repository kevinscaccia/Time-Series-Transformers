import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
THEME = 'plotly_dark'

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


def plot_serie(x):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0, len(x)), y=x, mode='lines', 
                              name=f'History', line=dict(color='#0099ff'), 
                              hovertemplate='$%{y:.2f}'))
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


def plot_train_history(train_l, val_l=None, offset=0,):
    plt.plot(np.arange(offset+1,len(train_l)+1),train_l[offset:],'r', label='Train')
    if val_l is not None:
        plt.plot(np.arange(offset+1,len(train_l)+1),val_l[offset:],'b', label='Validation')
    plt.legend()
    plt.show()

    
