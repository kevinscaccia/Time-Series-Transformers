import numpy as np

def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        
        end_ix = i + n_steps
        
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.asarray(x), np.asarray(y)

# Função para gerar uma série temporal aleatória com tendência, ruído e sazonalidade
def generate_time_series(length=100, trend_strength=0.0, seasonal_strength=0.0, noise_strength=1.0, scale_fac=2):
    time = np.arange(length)
    trend = trend_strength * time / length
    seasonal = seasonal_strength * np.sin((scale_fac) * np.pi * time / 12)
    noise = noise_strength * np.random.randn(length)
    series = trend + seasonal + noise
    return series

