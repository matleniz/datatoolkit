"""Formula transform: a whitelisted arithmetic expression over columns and @variables.

Variables are statistics fitted on train and frozen for apply (test never
contributes). The expression is parsed with ``ast`` after rewriting ``@name`` to
a reserved identifier; evaluation is vectorised with numpy — never ``eval``.
"""

from __future__ import annotations

import ast
import re
from typing import Literal, NoReturn

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from dtk_engine.errors import KeyParamsError
from dtk_engine.ops._util import json_scalar as _json
from dtk_engine.params import column_field
from dtk_engine.transform_registry import TransformParams, transform

IDENTIFIER = r"^[A-Za-z_][A-Za-z0-9_]*$"
_VAR_PREFIX = "_dtk_var_"
_COL_PREFIX = "_dtk_col_"  # df.col / df["col"] references
_VAR_TOKEN = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)")
_DIV_EPS = 1e-12

_ALLOWED_FUNCS = frozenset(
    {
        "log",
        "log1p",
        "log2",
        "log10",
        "exp",
        "sqrt",
        "abs",
        "round",
        "min",
        "max",
        "sin",
        "cos",
        "tanh",
        "floor",
        "ceil",
        "sign",
        "square",
        "clip",
        "where",
        "isnull",
    }
)
_ALLOWED_CONSTANTS = frozenset({"pi"})
_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.BitAnd,
    ast.BitOr,
)
_UNARYOPS = (ast.UAdd, ast.USub, ast.Not, ast.Invert)
_CMPOPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_STATS = Literal["mean", "median", "std", "min", "max", "q25", "q75", "count"]
_CONSTANT_VALUES = {"pi": float(np.pi)}
_NP_MODULES = frozenset({"np", "numpy"})
# np.<name> -> canonical whitelist name.
_NP_FUNCS: dict[str, str] = {
    **{
        f: f
        for f in (
            "log", "log1p", "log2", "log10", "exp", "sqrt", "abs", "round",
            "floor", "ceil", "sign", "square", "sin", "cos", "tanh", "clip",
            "where",
        )
    },
    "minimum": "min",
    "maximum": "max",
    "isnan": "isnull",
}
_NP_CONSTANTS: dict[str, str] = {"pi": "pi"}
# Exact arity for functions that are not min/max/round (those keep flexible rules).
_FUNC_ARITY: dict[str, int] = {
    "log": 1,
    "log1p": 1,
    "log2": 1,
    "log10": 1,
    "exp": 1,
    "sqrt": 1,
    "abs": 1,
    "sin": 1,
    "cos": 1,
    "tanh": 1,
    "floor": 1,
    "ceil": 1,
    "sign": 1,
    "square": 1,
    "isnull": 1,
    "clip": 3,
    "where": 3,
}


class FormulaVariable(TransformParams):
    name: str = Field(
        pattern=IDENTIFIER,
        description="Variable name used as @name in the expression",
    )
    stat: _STATS = Field(description="Statistic computed on train (NaN ignored)")
    column: str = column_field(
        ..., "Column the statistic is taken from", source="step", dtype="numeric"
    )


class FormulaParams(TransformParams):
    name: str = Field(
        pattern=IDENTIFIER,
        description="Output column (new or overwrite)",
    )
    expr: str = Field(
        description=(
            "Arithmetic expression over columns, numbers, functions and @variables"
        )
    )
    variables: list[FormulaVariable] = Field(
        default_factory=list,
        description="Named train statistics available as @name in expr",
    )

    @field_validator("expr")
    @classmethod
    def _expr_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("formula: expression is empty")
        return v

    @model_validator(mode="after")
    def _validate_expr(self) -> FormulaParams:
        declared = {v.name for v in self.variables}
        if len(declared) != len(self.variables):
            raise ValueError("formula: duplicate variable name in `variables`")
        _check_expr(self.expr, declared)
        return self


def _rewrite_vars(expr: str) -> str:
    return _VAR_TOKEN.sub(lambda m: f"{_VAR_PREFIX}{m.group(1)}", expr)


def _restore_vars(text: str) -> str:
    """Undo ``_rewrite_vars`` inside a string literal (e.g. ``df["a@b"]``)."""
    return text.replace(_VAR_PREFIX, "@")


def _not_allowed(msg: str) -> NoReturn:
    raise ValueError(f"formula: {msg}")


def _refuse(construct: str, detail: str = "") -> NoReturn:
    msg = f"{construct} is not allowed in a formula"
    if detail:
        msg += f" ({detail})"
    _not_allowed(msg)


def _dotted(node: ast.AST) -> list[str] | None:
    """``np.random.rand`` -> ``["np", "random", "rand"]``; None if not a name chain."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


def _check_dunder(attr: str) -> None:
    if attr.startswith("__"):
        _not_allowed(f"dunder attribute access ({attr}) is not allowed in a formula")


def _np_allowed() -> str:
    return ", ".join(f"np.{n}" for n in sorted(_NP_FUNCS) + sorted(_NP_CONSTANTS))


class _Normalizer(ast.NodeTransformer):
    """Map the Python-flavoured forms onto the canonical mini-language AST.

    ``np.f(...)`` / ``numpy.f(...)`` -> ``f(...)``, ``np.pi`` -> ``pi``,
    ``a if c else b`` -> ``where(c, a, b)``, ``df.col`` / ``df["col"]`` -> a
    column reference. Every other attribute / subscript / method call is refused
    here with a message naming the construct; the rest is checked by the walk.
    """

    def visit_Call(self, node: ast.Call) -> ast.AST:
        func = node.func
        if isinstance(func, ast.Attribute):
            _check_dunder(func.attr)
            parts = _dotted(func)
            if parts is not None and parts[0] in _NP_MODULES:
                name = ".".join(["np", *parts[1:]])
                if len(parts) != 2 or parts[1] not in _NP_FUNCS:
                    _not_allowed(
                        f"{name} is not allowed in a formula; "
                        f"allowed numpy names: {_np_allowed()}"
                    )
                node.func = ast.Name(id=_NP_FUNCS[parts[1]], ctx=ast.Load())
                if parts[1] in ("minimum", "maximum") and len(node.args) != 2:
                    _not_allowed(
                        f"{name}() takes 2 arguments, got {len(node.args)}"
                    )
            elif parts is not None and parts[0] == "df" and len(parts) == 2:
                _not_allowed(
                    f"method calls like df.{func.attr}(...) are not allowed "
                    "in a formula"
                )
            else:
                _not_allowed(
                    f"method calls like .{func.attr}(...) are not allowed in a "
                    "formula; call functions by name (log(x)) or as np.log(x)"
                )
        elif not isinstance(func, ast.Name):
            _not_allowed(
                "calling the result of an expression is not allowed in a "
                "formula; call functions by name (log(x)) or as np.log(x)"
            )
        node.args = [self.visit(arg) for arg in node.args]
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        _check_dunder(node.attr)
        parts = _dotted(node)
        if parts is not None and len(parts) == 2:
            root, attr = parts
            if root in _NP_MODULES:
                if attr in _NP_CONSTANTS:
                    return ast.Name(id=_NP_CONSTANTS[attr], ctx=ast.Load())
                if attr in _NP_FUNCS:
                    _not_allowed(f"np.{attr} must be called, e.g. np.{attr}(x)")
                _not_allowed(
                    f"np.{attr} is not allowed in a formula; "
                    f"allowed numpy names: {_np_allowed()}"
                )
            if root == "df":
                return ast.Name(id=f"{_COL_PREFIX}{attr}", ctx=ast.Load())
        _not_allowed(
            "attribute access is only allowed as np.<function> or df.<column>"
            + (f" (got {'.'.join(parts)})" if parts else "")
        )

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id == "df":
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                col = _restore_vars(key.value)
                return ast.Name(id=f"{_COL_PREFIX}{col}", ctx=ast.Load())
            _not_allowed('df[...] needs a column name in quotes, e.g. df["Age"]')
        _not_allowed('subscript is only allowed as df["column"]')

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        return ast.Call(
            func=ast.Name(id="where", ctx=ast.Load()),
            args=[self.visit(node.test), self.visit(node.body), self.visit(node.orelse)],
            keywords=[],
        )


# Named refusals for constructs the walk never accepts.
_REFUSED_NODES: tuple[tuple[type, str], ...] = (
    (ast.Lambda, "lambda"),
    (ast.ListComp, "a comprehension"),
    (ast.GeneratorExp, "a comprehension"),
    (ast.DictComp, "a comprehension"),
    (ast.SetComp, "a comprehension"),
    (ast.List, "a list literal"),
    (ast.Tuple, "a tuple"),
    (ast.Dict, "a dict literal"),
    (ast.Set, "a set literal"),
    (ast.NamedExpr, "assignment (:=)"),
    (ast.Await, "await"),
    (ast.Yield, "yield"),
    (ast.YieldFrom, "yield"),
    (ast.Starred, "*unpacking"),
    (ast.JoinedStr, "an f-string"),
    (ast.FormattedValue, "an f-string"),
    (ast.Slice, "a slice"),
)
_OP_SYMBOLS: dict[type, str] = {
    ast.MatMult: "@ (write @name with no space for a variable)",
    ast.BitXor: "^ (use ** for powers)",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.Is: "'is'",
    ast.IsNot: "'is not'",
    ast.In: "'in'",
    ast.NotIn: "'not in'",
}


def _op_label(op: ast.AST) -> str:
    return _OP_SYMBOLS.get(type(op), type(op).__name__)


def _check_expr(expr: str, declared_vars: set[str]) -> ast.Expression:
    """Parse, normalize and whitelist-walk ``expr``; return the canonical AST.

    Raises ValueError naming the refused construct.
    """
    rewritten = _rewrite_vars(expr)
    try:
        tree = ast.parse(rewritten, mode="eval")
    except SyntaxError as exc:
        if re.match(r"\s*(import|from)\b", expr):
            _refuse("import")
        if "@" in rewritten:
            _not_allowed("invalid or incomplete @name")
        raise ValueError(f"formula: invalid expression ({exc.msg})") from exc
    tree = _Normalizer().visit(tree)

    used_vars: set[str] = set()

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            walk(node.body)
            return
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                _refuse(
                    f"text constant {_restore_vars(node.value)!r}",
                    'only df["column"] takes a string',
                )
            if type(node.value) not in (int, float):
                _refuse(f"constant {node.value!r}", "only numbers are")
            return
        if isinstance(node, ast.Name):
            if node.id.startswith(_VAR_PREFIX):
                used_vars.add(node.id[len(_VAR_PREFIX) :])
            elif node.id.startswith("__") and node.id.endswith("__"):
                _refuse(f"dunder name {node.id}")
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _UNARYOPS):
                _refuse(_op_label(node.op))
            walk(node.operand)
            return
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _BINOPS):
                _refuse(_op_label(node.op))
            walk(node.left)
            walk(node.right)
            return
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                walk(value)
            return
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if not isinstance(op, _CMPOPS):
                    _refuse(_op_label(op))
            walk(node.left)
            for comparator in node.comparators:
                walk(comparator)
            return
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)  # _Normalizer guarantees it
            fname = node.func.id
            if node.keywords:
                _refuse("keyword arguments", f"in {fname}()")
            if fname not in _ALLOWED_FUNCS:
                _not_allowed(
                    f"unknown function {fname}() is not allowed in a formula; "
                    f"allowed: {', '.join(sorted(_ALLOWED_FUNCS))}"
                )
            if not node.args:
                _not_allowed(f"{fname}() needs at least one argument")
            if fname == "round":
                if len(node.args) > 2:
                    _not_allowed("round() takes at most 2 arguments")
                if len(node.args) == 2:
                    dec = node.args[1]
                    if (
                        not isinstance(dec, ast.Constant)
                        or type(dec.value) is not int
                    ):
                        _not_allowed(
                            "round() second argument must be an integer constant"
                        )
            elif fname in ("min", "max"):
                pass  # one or more args
            else:
                expected = _FUNC_ARITY[fname]
                if len(node.args) != expected:
                    _not_allowed(
                        f"{fname}() takes {expected} argument"
                        f"{'' if expected == 1 else 's'}, got {len(node.args)}"
                    )
            for arg in node.args:
                walk(arg)
            return
        for typ, label in _REFUSED_NODES:
            if isinstance(node, typ):
                _refuse(label)
        _refuse(type(node).__name__)

    walk(tree)
    unknown = used_vars - declared_vars
    if unknown:
        raise ValueError(
            "formula: unknown @variable "
            f"{sorted(unknown)}; declare it in `variables`"
        )
    return tree


def _column_refs(tree: ast.AST) -> dict[str, str]:
    """Map env key -> DataFrame column for every column the expression reads."""
    cols: dict[str, str] = {}

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if node.id.startswith(_VAR_PREFIX) or node.id in _ALLOWED_CONSTANTS:
                return
            if node.id.startswith(_COL_PREFIX):
                cols[node.id] = node.id[len(_COL_PREFIX) :]
            else:
                cols[node.id] = node.id

        def visit_Call(self, node: ast.Call) -> None:
            # Function name is not a column; only walk args.
            for arg in node.args:
                self.visit(arg)

    Visitor().visit(tree)
    return cols


def _stat(series: pd.Series, stat: str) -> float | None:
    s = pd.to_numeric(series, errors="coerce")
    if stat == "count":
        return float(s.notna().sum())
    s = s.dropna()
    if s.empty:
        return None
    if stat == "mean":
        v = s.mean()
    elif stat == "median":
        v = s.median()
    elif stat == "std":
        v = s.std(ddof=0)
    elif stat == "min":
        v = s.min()
    elif stat == "max":
        v = s.max()
    elif stat == "q25":
        v = s.quantile(0.25)
    else:  # q75
        v = s.quantile(0.75)
    return _json(float(v))


def _fit_formula(df: pd.DataFrame, params: FormulaParams) -> dict:
    out: dict[str, float | None] = {}
    for var in params.variables:
        if var.column not in df.columns:
            raise KeyParamsError(
                f"formula: unknown column {var.column!r} for variable {var.name!r}"
            )
        out[var.name] = _stat(df[var.column], var.stat)
    return {"variables": out}


def _np_func(name: str, args: list[np.ndarray]) -> np.ndarray:
    if name == "log":
        return np.log(args[0])
    if name == "log1p":
        return np.log1p(args[0])
    if name == "log2":
        return np.log2(args[0])
    if name == "log10":
        return np.log10(args[0])
    if name == "exp":
        return np.exp(args[0])
    if name == "sqrt":
        return np.sqrt(args[0])
    if name == "abs":
        return np.abs(args[0])
    if name == "sin":
        return np.sin(args[0])
    if name == "cos":
        return np.cos(args[0])
    if name == "tanh":
        return np.tanh(args[0])
    if name == "floor":
        return np.floor(args[0])
    if name == "ceil":
        return np.ceil(args[0])
    if name == "sign":
        return np.sign(args[0])
    if name == "square":
        return np.square(args[0])
    if name == "isnull":
        return np.isnan(args[0]).astype(float)
    if name == "clip":
        return np.clip(args[0], args[1], args[2])
    if name == "where":
        cond, a, b = args
        out = np.where(cond != 0, a, b).astype(float, copy=False)
        out = np.asarray(out, dtype=float)
        out[np.isnan(cond)] = np.nan
        return out
    if name == "round":
        return np.round(args[0])
    if name == "min":
        return np.minimum.reduce(args)
    return np.maximum.reduce(args)  # max


def _cmp(op: ast.cmpop, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    return left >= right  # GtE


def _eval_compare(node: ast.Compare, env: dict[str, np.ndarray], n: int) -> np.ndarray:
    """Comparisons evaluate to 0/1; any NaN operand yields NaN (missing in → out)."""
    left = _eval_node(node.left, env, n)
    values = [left]
    mask = np.ones(n, dtype=bool)
    current = left
    for op, comparator in zip(node.ops, node.comparators, strict=True):
        right = _eval_node(comparator, env, n)
        values.append(right)
        mask &= _cmp(op, current, right)
        current = right
    out = np.where(mask, 1.0, 0.0)
    nan = np.zeros(n, dtype=bool)
    for v in values:
        nan |= np.isnan(v)
    out[nan] = np.nan
    return out


def _logical(values: list[np.ndarray], combine) -> np.ndarray:
    """Element-wise boolean op on truthiness (!= 0) -> 0/1; any NaN -> NaN."""
    out = combine([v != 0 for v in values]).astype(float)
    for v in values:
        out[np.isnan(v)] = np.nan
    return out


def _safe_div(op, a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """``op(a, b)`` with NaN where ``|b|`` is ~0 (``/``, ``//``, ``%``)."""
    out = np.full(n, np.nan, dtype=float)
    ok = np.abs(b) >= _DIV_EPS
    op(a, b, out=out, where=ok)
    return out


def _eval_node(node: ast.AST, env: dict[str, np.ndarray], n: int) -> np.ndarray:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env, n)
    if isinstance(node, ast.Constant):
        return np.full(n, float(node.value), dtype=float)
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        a = _eval_node(node.operand, env, n)
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return _logical([a], lambda m: ~m[0])
        return -a if isinstance(node.op, ast.USub) else np.asarray(a, dtype=float)
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, env, n) for v in node.values]
        reduce = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        return _logical(values, reduce.reduce)
    if isinstance(node, ast.BinOp):
        a = _eval_node(node.left, env, n)
        b = _eval_node(node.right, env, n)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return _safe_div(np.divide, a, b, n)
        if isinstance(node.op, ast.FloorDiv):
            return _safe_div(np.floor_divide, a, b, n)
        if isinstance(node.op, ast.Mod):
            return _safe_div(np.mod, a, b, n)
        if isinstance(node.op, ast.BitAnd):
            return _logical([a, b], np.logical_and.reduce)
        if isinstance(node.op, ast.BitOr):
            return _logical([a, b], np.logical_or.reduce)
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            return np.power(a, b)
    if isinstance(node, ast.Compare):
        return _eval_compare(node, env, n)
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)
        # round(x, ndigits): ndigits is an int constant (checked in _check_expr).
        if node.func.id == "round" and len(node.args) == 2:
            x = _eval_node(node.args[0], env, n)
            dec = node.args[1]
            assert isinstance(dec, ast.Constant) and type(dec.value) is int
            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                return np.round(x, int(dec.value))
        args = [_eval_node(arg, env, n) for arg in node.args]
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            return _np_func(node.func.id, args)
    raise KeyParamsError(f"formula: refused {type(node).__name__}")


@transform(
    "formula",
    params_model=FormulaParams,
    fit=_fit_formula,
    title="Formula",
    description=(
        "Add a float column from a whitelisted expression over columns, "
        "numbers, functions (incl. where/clip/comparisons) and train-fitted "
        "@variables."
    ),
)
def formula(df: pd.DataFrame, params: FormulaParams, state: dict) -> pd.DataFrame:
    """Add a float column from a safe expression; @variables reuse train values."""
    tree = _check_expr(params.expr, {v.name for v in params.variables})
    n = len(df)
    env: dict[str, np.ndarray] = {
        name: np.full(n, value, dtype=float)
        for name, value in _CONSTANT_VALUES.items()
    }
    for key, col in _column_refs(tree).items():
        if col not in df.columns:
            raise KeyParamsError(f"formula: unknown column {col!r}")
        env[key] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    frozen = state.get("variables", {})
    for var in params.variables:
        key = f"{_VAR_PREFIX}{var.name}"
        value = frozen.get(var.name)
        fill = np.nan if value is None else float(value)
        env[key] = np.full(n, fill, dtype=float)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        result = _eval_node(tree, env, n)
    result = np.asarray(result, dtype=float)
    result[~np.isfinite(result)] = np.nan
    return df.assign(**{params.name: result})
