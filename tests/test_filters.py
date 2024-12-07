import pytest
import numpy as np
from sygnals.core.filters import low_pass_filter, high_pass_filter

def test_low_pass_filter():
    fs = 1000
    t = np.arange(0,1,1/fs)
    # High-frequency signal
    x = np.sin(2*np.pi*300*t)
    y = low_pass_filter(x, cutoff=100, fs=fs)
    # After low-pass at 100Hz, amplitude should be greatly reduced
    assert np.max(np.abs(y)) < np.max(np.abs(x))

def test_high_pass_filter():
    fs = 1000
    t = np.arange(0,1,1/fs)
    # Low-frequency signal
    x = np.sin(2*np.pi*10*t)
    y = high_pass_filter(x, cutoff=100, fs=fs)
    # After high-pass at 100Hz, amplitude should be reduced
    assert np.max(np.abs(y)) < np.max(np.abs(x))
