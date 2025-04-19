# sygnals/core/custom_exec.py

"""
Provides functionality to safely evaluate custom mathematical expressions
using a restricted set of allowed functions and variables.
"""
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)

# Define allowed global functions and modules for eval
SAFE_GLOBALS = {
    "np": np,
    "math": math,
    # Allow common numpy functions directly
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e,
    # Allow common math functions directly (if different from numpy or preferred)
    # 'math_sqrt': math.sqrt, # Example if needed
}

def evaluate_expression(expression: str, variables: dict) -> any:
    """
    Safely evaluate a custom mathematical expression using restricted globals.

    Args:
        expression: The mathematical expression string to evaluate.
        variables: A dictionary of variable names and their values available
                   to the expression (e.g., {'x': 5, 'y': 3}).

    Returns:
        The result of the evaluated expression.

    Raises:
        NameError: If the expression uses disallowed functions or variables.
        SyntaxError: If the expression has invalid syntax.
        TypeError: If operations are attempted on incompatible types.
        Exception: For other potential evaluation errors.
    """
    # Combine predefined safe globals with user-provided variables
    # User variables can override safe globals if names clash (use with caution)
    safe_locals = {**SAFE_GLOBALS, **variables}
    logger.debug(f"Evaluating expression: '{expression}' with variables: {list(variables.keys())}")

    try:
        # Evaluate the expression with restricted builtins and combined locals/globals
        # Using {"__builtins__": {}} prevents access to standard builtins like open(), etc.
        result = eval(expression, {"__builtins__": {}}, safe_locals)
        return result
    except NameError as e:
        logger.error(f"Evaluation failed: NameError - '{e}'. Ensure all variables and functions are defined in SAFE_GLOBALS or provided variables.")
        raise
    except SyntaxError as e:
        logger.error(f"Evaluation failed: SyntaxError - {e}")
        raise
    except TypeError as e:
        logger.error(f"Evaluation failed: TypeError - {e}")
        raise
    except Exception as e:
        # Catch any other unexpected errors during evaluation
        logger.error(f"Evaluation failed: Unexpected error - {e}", exc_info=True)
        raise
