import numpy as np

def calculate_snr(signal, noise):
    '''Calculate Signal-to-Noise Ratio (SNR)'''
    return 10 * np.log10(np.var(signal) / np.var(noise))

def calculate_rmse(original, optimized):
    '''Calculate Root Mean Square Error (RMSE)'''
    return np.sqrt(np.mean((original - optimized)**2))
