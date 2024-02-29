import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_data(num_steps: int, interval: float = 0.1, plot=True) -> None:
    x = np.linspace(0, num_steps * interval, num_steps)
    y = np.sin(x) + np.random.normal(0, 0.01, x.shape)
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(x[:100], y[:100], label='Sinusoidal Time Series', color='green')
        plt.title('Sinusoidal Time Series')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    return torch.tensor(y, dtype=torch.float32)


def generate_serie(num_points, amplitude=1, frequency=1, plot=False):
    # Generate time series data
    time = np.linspace(0, 10, num_points)  # Adjust the range if needed
    data = amplitude * np.sin(2 * np.pi * frequency * time) 
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(time, data, label='Sinusoidal Time Series', color='green')
        plt.title('Sinusoidal Time Series')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    return torch.tensor(data, dtype=torch.float32)


def generate_serie_2(num_points, noise=True, plot=False):
    time = np.arange(0, num_points, 0.1)    
    data   = np.sin(time) + np.sin(time * 0.05)  
    if noise:
        data = data + np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(time, data, label='Sinusoidal Time Series', color='green')
        plt.title('Sinusoidal Time Series')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    return torch.tensor(data, dtype=torch.float32) 
