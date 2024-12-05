import numpy as np
import math

SAFE_GLOBALS = {
    "np": np,
    "math": math,
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "pi": np.pi
}

def evaluate_expression(expression, variables):
    """Safely evaluate a custom mathematical expression."""
    safe_locals = {**SAFE_GLOBALS, **variables}
    return eval(expression, {"__builtins__": None}, safe_locals)
