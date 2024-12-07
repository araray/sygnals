import pytest
import numpy as np
from sygnals.core.custom_exec import evaluate_expression

def test_evaluate_expression():
    expr = "sin(x)"
    x_values = {"x": np.pi/2}
    result = evaluate_expression(expr, x_values)
    assert pytest.approx(result, 1e-7) == 1.0
