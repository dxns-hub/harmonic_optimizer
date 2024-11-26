import pytest
import numpy as np
from harmonic_optimizer.core import HarmonicOptimizer, OptimizationParams

def test_optimizer_initialization():
    params = OptimizationParams(R=1.0, F=1.0, E=1.0, constant=np.pi)
    optimizer = HarmonicOptimizer(params)
    assert optimizer is not None
    assert optimizer.params.constant == np.pi

def test_signal_optimization():
    # Create a simple test signal
    t = np.linspace(0, 10, 1000)
    test_signal = np.sin(t) + 0.1 * np.random.randn(len(t))
    
    # Initialize optimizer
    params = OptimizationParams(R=1.0, F=1.0, E=1.0, constant=np.pi)
    optimizer = HarmonicOptimizer(params)
    
    # Optimize signal
    optimized = optimizer.optimize(test_signal)
    
    # Check that optimization improved SNR
    original_noise = np.std(test_signal - np.sin(t))
    optimized_noise = np.std(optimized - np.sin(t))
    assert optimized_noise < original_noise
