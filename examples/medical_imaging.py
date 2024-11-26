from harmonic_optimizer.applications.medical import MedicalOptimizer
from harmonic_optimizer.core import OptimizationParams
import numpy as np

# Create optimization parameters
params = OptimizationParams(R=1.2, F=0.8, E=1.0, constant=(1 + np.sqrt(5)) / 2)
optimizer = MedicalOptimizer(params)

# Generate a synthetic medical signal
t = np.linspace(0, 10, 1000)
signal = np.sin(t) + 0.1 * np.random.randn(len(t))

# Optimize the signal
optimized_signal = optimizer.optimize_medical_image(signal)

# Plot the results
from harmonic_optimizer.utils.visualization import plot_signals
plot_signals(signal, signal + 0.1 * np.random.randn(len(t)), optimized_signal, title="Medical Signal Optimization")
