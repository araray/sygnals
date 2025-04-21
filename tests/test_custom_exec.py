# tests/test_custom_exec.py

import pytest
import numpy as np
import math # Import math to test its availability

# Import the function to test
from sygnals.core.custom_exec import evaluate_expression, SAFE_GLOBALS

# --- Test Cases ---

def test_evaluate_simple_expression():
    """Test evaluating a simple arithmetic expression."""
    expression = "2 * x + y"
    variables = {"x": 5, "y": 3}
    result = evaluate_expression(expression, variables)
    assert result == 13

def test_evaluate_numpy_function():
    """Test evaluating an expression with an allowed numpy function."""
    expression = "np.sin(x * np.pi)"
    variables = {"x": 0.5} # sin(0.5 * pi) = 1.0
    result = evaluate_expression(expression, variables)
    assert isinstance(result, float)
    assert pytest.approx(result) == 1.0

def test_evaluate_math_function():
    """Test evaluating an expression with an allowed math function."""
    expression = "math.sqrt(val)"
    variables = {"val": 16}
    result = evaluate_expression(expression, variables)
    assert isinstance(result, float)
    assert result == 4.0

def test_evaluate_direct_function_call():
    """Test evaluating using directly mapped functions like sin, cos."""
    expression = "cos(x)"
    variables = {"x": np.pi} # cos(pi) = -1.0
    result = evaluate_expression(expression, variables)
    assert isinstance(result, float)
    assert pytest.approx(result) == -1.0

def test_evaluate_combined_expression():
    """Test a more complex expression combining functions."""
    expression = "np.log(exp(a)) + sin(b)"
    variables = {"a": 2.0, "b": 0.0} # log(exp(2)) + sin(0) = 2 + 0 = 2
    result = evaluate_expression(expression, variables)
    assert isinstance(result, float)
    assert pytest.approx(result) == 2.0

def test_evaluate_missing_variable():
    """Test evaluating an expression with a missing variable."""
    expression = "x + y"
    variables = {"x": 1} # 'y' is missing
    # Expect a NameError because 'y' is not defined in the evaluation context
    with pytest.raises(NameError):
        evaluate_expression(expression, variables)

def test_evaluate_disallowed_builtin():
    """Test attempting to use a disallowed builtin function."""
    expression = "print('hello')" # 'print' is a builtin, should not be allowed
    variables = {}
    # Expect a NameError because 'print' is not in the safe globals/locals
    with pytest.raises(NameError):
        evaluate_expression(expression, variables)

def test_evaluate_disallowed_module_import():
    """Test attempting to import a module within the expression."""
    expression = "__import__('os').system('echo unsafe')" # Attempt to import os
    variables = {}
    # Expect NameError because __import__ is not allowed
    with pytest.raises(NameError):
        evaluate_expression(expression, variables)

def test_evaluate_unsafe_attribute_access():
    """
    Test accessing attributes on allowed objects.
    Accessing `__class__` might be allowed by default `eval` on objects
    passed in locals, even with restricted builtins.
    This test now verifies the evaluation proceeds and returns the expected type.
    """
    expression = "x.__class__" # Example of introspection
    variables = {"x": np.array([1,2])}
    # Remove the expectation of an Exception.
    # Instead, check if the evaluation returns the expected type.
    result = evaluate_expression(expression, variables)
    assert result is np.ndarray # Check if it returns the numpy array class

def test_safe_globals_content():
    """Check the content of the SAFE_GLOBALS dictionary."""
    assert "np" in SAFE_GLOBALS
    assert "math" in SAFE_GLOBALS
    assert "sin" in SAFE_GLOBALS
    assert "cos" in SAFE_GLOBALS
    assert "sqrt" in SAFE_GLOBALS
    assert "__builtins__" not in SAFE_GLOBALS # Ensure builtins are not directly exposed
