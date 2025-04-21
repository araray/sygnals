# sygnals/core/ml_utils/scaling.py

"""
Feature scaling implementations.

Wraps common scalers from scikit-learn for easy use within Sygnals workflows.
Scaling is often crucial for the performance of many machine learning algorithms.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, Literal, Dict, Any, Tuple

# Import scalers from scikit-learn
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    # Define dummy classes if sklearn is not available to avoid import errors downstream,
    # but raise NotImplementedError on use.
    class StandardScaler:
        def __init__(self, *args, **kwargs): pass
        def fit(self, X): pass
        def transform(self, X): raise NotImplementedError("scikit-learn is required for StandardScaler.")
        def fit_transform(self, X): raise NotImplementedError("scikit-learn is required for StandardScaler.")
        def inverse_transform(self, X): raise NotImplementedError("scikit-learn is required for StandardScaler.")

    class MinMaxScaler:
         def __init__(self, *args, **kwargs): pass
         def fit(self, X): pass
         def transform(self, X): raise NotImplementedError("scikit-learn is required for MinMaxScaler.")
         def fit_transform(self, X): raise NotImplementedError("scikit-learn is required for MinMaxScaler.")
         def inverse_transform(self, X): raise NotImplementedError("scikit-learn is required for MinMaxScaler.")

    class RobustScaler:
         def __init__(self, *args, **kwargs): pass
         def fit(self, X): pass
         def transform(self, X): raise NotImplementedError("scikit-learn is required for RobustScaler.")
         def fit_transform(self, X): raise NotImplementedError("scikit-learn is required for RobustScaler.")
         def inverse_transform(self, X): raise NotImplementedError("scikit-learn is required for RobustScaler.")

logger = logging.getLogger(__name__)

# Type alias for scaler instances
ScalerType = Union[StandardScaler, MinMaxScaler, RobustScaler]

def apply_scaling(
    features: NDArray[np.float64],
    scaler_type: Literal['standard', 'minmax', 'robust'] = 'standard',
    scaler_params: Optional[Dict[str, Any]] = None,
    fit: bool = True,
    scaler_instance: Optional[ScalerType] = None
) -> Tuple[NDArray[np.float64], ScalerType]:
    """
    Applies feature scaling to the input feature array.

    Can either fit a new scaler or apply a pre-fitted one.

    Args:
        features: The input feature array (NumPy array, float64).
                  Expected shape: (n_samples, n_features) or (n_frames, n_features).
        scaler_type: The type of scaler to use ('standard', 'minmax', 'robust').
                     Ignored if `scaler_instance` is provided. (Default: 'standard')
        scaler_params: Optional dictionary of parameters to pass to the scaler constructor
                       (e.g., `{'with_mean': False}` for StandardScaler).
        fit: If True (default), fit a new scaler to the data before transforming.
             If False, `scaler_instance` must be provided and pre-fitted.
        scaler_instance: Optional pre-fitted scaler instance (e.g., from a previous fit).
                         If provided, `scaler_type` and `scaler_params` are ignored,
                         and `fit` must be False.

    Returns:
        A tuple containing:
        - scaled_features (NDArray[np.float64]): The scaled feature array.
        - fitted_scaler (ScalerType): The scaler instance used (either newly fitted or the one provided).

    Raises:
        ValueError: If parameters are invalid (e.g., `fit=False` without `scaler_instance`).
        ImportError: If scikit-learn is not installed.
        NotImplementedError: If scikit-learn is not installed and scaling is attempted.
        Exception: For errors during scikit-learn scaling.
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn package is required for feature scaling. Please install it (`pip install scikit-learn`).")

    if features.ndim != 2:
        # Ensure input is 2D (n_samples, n_features)
        # If 1D, reshape assuming it's a single feature across samples/frames
        if features.ndim == 1:
             logger.debug("Input features are 1D. Reshaping to (n_samples, 1) for scaling.")
             features = features.reshape(-1, 1)
        else:
             raise ValueError(f"Input features must be 1D or 2D (samples/frames x features), got shape {features.shape}")

    scaler_params = scaler_params or {}
    logger.info(f"Applying feature scaling: type={scaler_type if scaler_instance is None else 'provided'}, fit={fit}")

    current_scaler: ScalerType

    if fit:
        if scaler_instance is not None:
            logger.warning("`scaler_instance` provided but `fit=True`. Ignoring provided instance and fitting a new scaler.")

        # Create and fit a new scaler
        logger.debug(f"Fitting new '{scaler_type}' scaler with params: {scaler_params}")
        try:
            if scaler_type == 'standard':
                current_scaler = StandardScaler(**scaler_params)
            elif scaler_type == 'minmax':
                current_scaler = MinMaxScaler(**scaler_params)
            elif scaler_type == 'robust':
                current_scaler = RobustScaler(**scaler_params)
            else:
                raise ValueError(f"Unsupported scaler_type: '{scaler_type}'. Choose 'standard', 'minmax', or 'robust'.")

            scaled_features = current_scaler.fit_transform(features)
        except Exception as e:
            logger.error(f"Error fitting/transforming with {scaler_type} scaler: {e}")
            raise
    else:
        # Apply a pre-fitted scaler
        if scaler_instance is None:
            raise ValueError("`scaler_instance` must be provided when `fit=False`.")
        logger.debug("Applying pre-fitted scaler.")
        current_scaler = scaler_instance
        try:
            # Check if scaler is actually fitted (has attributes like 'scale_')
            # This check might vary slightly depending on the scaler type
            if not hasattr(current_scaler, 'scale_') and not hasattr(current_scaler, 'min_'): # Check attributes common after fitting
                 raise ValueError("Provided `scaler_instance` does not appear to be fitted.")
            scaled_features = current_scaler.transform(features)
        except ValueError as e: # Catch specific errors like not fitted
             logger.error(f"Error applying pre-fitted scaler: {e}")
             raise
        except Exception as e:
            logger.error(f"Error transforming with pre-fitted scaler: {e}")
            raise

    # Ensure output is float64
    return scaled_features.astype(np.float64, copy=False), current_scaler

# --- Convenience Wrappers (Optional) ---

def standard_scale(
    features: NDArray[np.float64],
    with_mean: bool = True,
    with_std: bool = True
) -> Tuple[NDArray[np.float64], StandardScaler]:
    """Applies StandardScaler."""
    params = {'with_mean': with_mean, 'with_std': with_std}
    # Type hint helps static analysis, but runtime check handles it
    scaled_features, scaler = apply_scaling(features, scaler_type='standard', scaler_params=params)
    return scaled_features, scaler # type: ignore

def minmax_scale(
    features: NDArray[np.float64],
    feature_range: Tuple[float, float] = (0, 1)
) -> Tuple[NDArray[np.float64], MinMaxScaler]:
    """Applies MinMaxScaler."""
    params = {'feature_range': feature_range}
    scaled_features, scaler = apply_scaling(features, scaler_type='minmax', scaler_params=params)
    return scaled_features, scaler # type: ignore

def robust_scale(
    features: NDArray[np.float64],
    with_centering: bool = True,
    with_scaling: bool = True,
    quantile_range: Tuple[float, float] = (25.0, 75.0)
) -> Tuple[NDArray[np.float64], RobustScaler]:
    """Applies RobustScaler."""
    params = {'with_centering': with_centering, 'with_scaling': with_scaling, 'quantile_range': quantile_range}
    scaled_features, scaler = apply_scaling(features, scaler_type='robust', scaler_params=params)
    return scaled_features, scaler # type: ignore
