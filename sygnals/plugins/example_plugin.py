from sygnals.core.plugin_manager import register_plugin
import numpy as np

@register_plugin
def custom_filter(data, alpha=0.5):
    """Custom exponential smoothing filter."""
    smoothed_data = []
    for i, value in enumerate(data):
        if i == 0:
            smoothed_data.append(value)
        else:
            smoothed_data.append(alpha * value + (1 - alpha) * smoothed_data[-1])
    return np.array(smoothed_data)

@register_plugin
def square_signal(data):
    """Square the signal values."""
    return data ** 2
