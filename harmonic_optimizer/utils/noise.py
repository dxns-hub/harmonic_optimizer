import numpy as np

def add_gaussian_noise(signal, mean=0, std=0.1):
    '''Add Gaussian noise to a signal'''
    return signal + np.random.normal(mean, std, signal.shape)

def add_speckle_noise(signal, scale=0.1):
    '''Add speckle noise to a signal'''
    speckle = np.random.rayleigh(scale, signal.shape)
    return signal * (1 + speckle)
