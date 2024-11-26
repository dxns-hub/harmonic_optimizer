import matplotlib.pyplot as plt

def plot_signals(original, noisy, optimized, title="Signal Optimization"):
    '''Plot original, noisy, and optimized signals'''
    plt.figure(figsize=(10, 6))
    plt.plot(original, label='Original', alpha=0.7)
    plt.plot(noisy, label='Noisy', alpha=0.5)
    plt.plot(optimized, label='Optimized', alpha=0.9)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
