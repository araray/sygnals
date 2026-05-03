# sygnals/core/custom_exec.py
# Drop-in replacement for the previous eval-based implementation.
#
# Background
# ----------
# The previous implementation called eval() with `{"__builtins__": {}}` and a
# locals dict containing numpy / math etc. This is NOT a sandbox: an attacker
# can break out via subclass traversal of the `object` MRO, regardless of
# which names are exposed in SAFE_GLOBALS:
#
#   ().__class__.__base__.__subclasses__()
#       → list of every class loaded in the interpreter, including
#         _frozen_importlib.BuiltinImporter, which can import arbitrary
#         modules (os, subprocess, ...).
#
# A demonstration PoC ran `id` and `whoami` as root through the previous
# implementation. See poc_custom_exec_rce.py.
#
# Approach
# --------
# This replacement parses the user expression with the `ast` module, walks
# the resulting tree, and rejects any node type that isn't on a small
# whitelist. Specifically:
#
#   - All attribute access is forbidden (no x.foo, no x.__class__).
#   - All names starting with `_` are forbidden.
#   - Comprehensions, walrus, lambdas, generators, imports, statements of
#     any kind are forbidden (they are simply not allowed node types).
#   - Function calls are only allowed against bare names in a whitelist of
#     callables (no x.method() and no f()() style indirection).
#
# After validation the AST is evaluated by a small recursive interpreter.
# This is a few hundred lines and covers the same use cases the previous
# implementation supported (arithmetic, comparison, logic, conditional
# expressions, list/tuple literals, indexing, math/numpy function calls).
#
# Public API
# ----------
#   evaluate_expression(expr: str, variables: Mapping[str, Any]) -> Any
#   class UnsafeExpressionError(ValueError)

from __future__ import annotations

import ast
import logging
import math
import operator
from typing import Any, Callable, Mapping, Set

import numpy as np

logger = logging.getLogger(__name__)


# --- Whitelist tables ---------------------------------------------------------

# Callables exposed by name. Add carefully. Each entry must be safe to call with
# user-supplied arguments — i.e. it must not have side effects, must not access
# the filesystem or network, and must not perform attribute lookups via its
# argument (e.g. __init_subclass__, __call__ overrides, etc).
_SAFE_FUNCS: dict[str, Callable[..., Any]] = {
    # trig
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    # exp/log
    "exp": np.exp,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "cbrt": np.cbrt,
    # powers / abs / sign
    "abs": np.abs,
    "pow": np.power,
    "sign": np.sign,
    # rounding
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "trunc": np.trunc,
    # element-wise min/max
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
    # reductions (operate on user-provided arrays)
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "mean": np.mean,
    "std": np.std,
    "median": np.median,
    "var": np.var,
    # constructors over literal-only inputs
    "array": np.array,
}

# Constants exposed by bare name.
_SAFE_CONSTS: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "True": True,
    "False": False,
    "None": None,
    "inf": math.inf,
    "nan": math.nan,
}

# Operator dispatch tables for the recursive interpreter.
_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.MatMult: operator.matmul,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
}
_UNARYOPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}
_CMPOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

# Whitelisted AST node types. Anything not in this set is rejected.
_ALLOWED_NODES: Set[type] = {
    ast.Expression,
    # literals
    ast.Constant,
    # names + load context
    ast.Name,
    ast.Load,
    # operators
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    # boolean ops
    ast.And,
    ast.Or,
    # ternary
    ast.IfExp,
    # function call
    ast.Call,
    ast.keyword,
    # collection literals + indexing
    ast.List,
    ast.Tuple,
    ast.Subscript,
    ast.Slice,
}
_ALLOWED_NODES |= set(_BINOPS) | set(_UNARYOPS) | set(_CMPOPS)


# --- Public surface -----------------------------------------------------------


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains a node type that is not whitelisted."""


def evaluate_expression(expression: str, variables: Mapping[str, Any]) -> Any:
    """Safely evaluate a restricted-grammar expression.

    Allowed:
        * Arithmetic, comparison, logical, bitwise operators
        * Conditional expressions (a if b else c)
        * Whitelisted functions (sin, cos, exp, log, sqrt, np reductions, ...)
        * Whitelisted constants (pi, e, tau, inf, nan, True, False, None)
        * User-supplied variables (any value, by name)
        * List/tuple literals
        * Subscripting and slicing (x[0], x[::2])

    Forbidden:
        * All attribute access (x.foo and x.__class__ alike)
        * All names beginning with `_`
        * Comprehensions, generators, lambdas, walrus
        * Imports, statements of any kind
        * Indirect function calls (only bare-name calls allowed)

    Args:
        expression: The user-supplied expression string.
        variables: Mapping of variable name -> value, available to the expression.

    Returns:
        The evaluated result.

    Raises:
        UnsafeExpressionError: For disallowed nodes/attributes/calls.
        SyntaxError: For malformed expressions.
        NameError: For references to undefined names.
        ValueError, TypeError, ZeroDivisionError, etc.: From operator semantics.
    """
    logger.debug("Evaluating user expression: %r", expression)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        logger.debug("Syntax error in expression: %r", expression)
        raise

    _validate_tree(tree)
    return _eval_node(tree, variables)


# --- Internals ----------------------------------------------------------------


def _validate_tree(tree: ast.AST) -> None:
    """Walk the AST and reject anything not on the whitelist.

    Most importantly: rejects Attribute nodes outright (no x.foo at all),
    so the subclass-traversal escape vector is unreachable.
    """
    for node in ast.walk(tree):
        nt = type(node)

        # Old-style numbered-node compatibility: ast.Num/Str/Bytes are deprecated
        # since 3.8 but still appear in some toolchains.
        if nt.__name__ in ("Num", "Str", "Bytes"):
            continue

        if nt is ast.Attribute:
            raise UnsafeExpressionError(
                "attribute access is not allowed (this blocks dunder traversal "
                "and method lookup; rewrite using bare-name function calls)"
            )

        if nt is ast.Name:
            assert isinstance(node, ast.Name)
            if node.id.startswith("_"):
                raise UnsafeExpressionError(
                    f"names beginning with underscore are not allowed: {node.id!r}"
                )

        if nt is ast.Call:
            assert isinstance(node, ast.Call)
            # Only allow direct calls of bare names: f(...) and not x.f(...) or
            # f()() . The Attribute check above already covers x.f. Reject any
            # callable that isn't a Name.
            if not isinstance(node.func, ast.Name):
                raise UnsafeExpressionError("indirect or chained calls are not allowed")
            for kw in node.keywords:
                # **kwargs would have kw.arg is None — disallow because it
                # could pull from an unknown user dict that contains dunder
                # keys; the args themselves are still validated by the walker.
                if kw.arg is None:
                    raise UnsafeExpressionError("**kwargs is not allowed")
                if kw.arg.startswith("_"):
                    raise UnsafeExpressionError(
                        f"keyword arguments beginning with underscore are not allowed: {kw.arg!r}"
                    )

        if nt is ast.keyword:
            # already covered above when we hit the parent Call
            continue

        if nt not in _ALLOWED_NODES:
            raise UnsafeExpressionError(
                f"disallowed expression construct: {nt.__name__} "
                "(only arithmetic / comparison / logic / function calls / "
                "literals / subscripts are permitted)"
            )


def _eval_node(node: ast.AST, env: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in _SAFE_CONSTS:
            return _SAFE_CONSTS[node.id]
        if node.id in _SAFE_FUNCS:
            return _SAFE_FUNCS[node.id]
        raise NameError(f"name {node.id!r} is not defined")

    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](
            _eval_node(node.left, env), _eval_node(node.right, env)
        )

    if isinstance(node, ast.UnaryOp):
        return _UNARYOPS[type(node.op)](_eval_node(node.operand, env))

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        for op, right in zip(node.ops, node.comparators):
            right_val = _eval_node(right, env)
            if not _CMPOPS[type(op)](left, right_val):
                return False
            left = right_val
        return True

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = _eval_node(v, env)
                if not result:
                    return result
            return result
        # Or
        result = False
        for v in node.values:
            result = _eval_node(v, env)
            if result:
                return result
        return result

    if isinstance(node, ast.IfExp):
        return _eval_node(node.body if _eval_node(node.test, env) else node.orelse, env)

    if isinstance(node, ast.Call):
        # Validator already guaranteed func is a bare Name and there's no **kwargs.
        assert isinstance(node.func, ast.Name)
        target = _SAFE_FUNCS.get(node.func.id) or env.get(node.func.id)
        if target is None or not callable(target):
            raise UnsafeExpressionError(
                f"call to non-whitelisted or non-callable {node.func.id!r}"
            )
        args = [_eval_node(a, env) for a in node.args]
        kwargs = {kw.arg: _eval_node(kw.value, env) for kw in node.keywords}
        return target(*args, **kwargs)

    if isinstance(node, ast.List):
        return [_eval_node(e, env) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(e, env) for e in node.elts)

    if isinstance(node, ast.Subscript):
        target = _eval_node(node.value, env)
        if isinstance(node.slice, ast.Slice):
            lo = (
                _eval_node(node.slice.lower, env)
                if node.slice.lower is not None
                else None
            )
            hi = (
                _eval_node(node.slice.upper, env)
                if node.slice.upper is not None
                else None
            )
            st = (
                _eval_node(node.slice.step, env)
                if node.slice.step is not None
                else None
            )
            return target[lo:hi:st]
        return target[_eval_node(node.slice, env)]

    raise UnsafeExpressionError(
        f"validator missed an unsupported node: {type(node).__name__}"
    )


# Backwards-compatibility shim. The previous implementation exposed
# SAFE_GLOBALS as a module-level dict; some tests / users may import it.
# We provide a read-only view that combines the constants and callables.
SAFE_GLOBALS: dict[str, Any] = {**_SAFE_CONSTS, **_SAFE_FUNCS}
