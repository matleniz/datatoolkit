"""Formula transform: a whitelisted arithmetic expression over columns and @variables.

Variables are statistics fitted on train and frozen for apply (test never
contributes). The expression is parsed with ``ast`` after rewriting ``@name`` to
a reserved identifier; evaluation is vectorised with numpy — never ``eval``.

Python-flavoured spellings (MAT-241) are translated onto the same whitelist
before the walk: ``np.f(x)`` -> ``f(x)``, ``a if c else b`` -> ``where(c, a, b)``,
``df.col`` / ``df["col"]`` -> column; ``and``/``or``/``not``/``&``/``|``/``~``
are element-wise on truthiness (!= 0) and return 0/1. Any other attribute,
subscript or method call is refused with a message naming the construct.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
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

_ALLOWED_CONSTANTS = frozenset({"pi"})
_STATS = Literal["mean", "median", "std", "min", "max", "q25", "q75", "count"]
_CONSTANT_VALUES = {"pi": float(np.pi)}
_NP_MODULES = frozenset({"np", "numpy"})
_NP_CONSTANTS: dict[str, str] = {"pi": "pi"}
_NP_BINARY = ("minimum", "maximum")  # np spellings that take exactly 2 arguments


def _where(cond: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.asarray(np.where(cond != 0, a, b), dtype=float)
    out[np.isnan(cond)] = np.nan
    return out


def _round(x: np.ndarray, ndigits: np.ndarray | None = None) -> np.ndarray:
    return np.round(x) if ndigits is None else np.round(x, int(ndigits[:1].sum()))


@dataclass(frozen=True)
class _Func:
    """One whitelisted function: the single place validation, errors and eval read."""

    impl: Callable[..., np.ndarray]
    arity: tuple[int, int | None] = (1, 1)  # (min, max); max None = n-ary
    np_name: str | None = None  # np.<np_name> spelling; defaults to the name
    int_tail: bool = False  # arguments after the first must be int constants


_FUNCS: dict[str, _Func] = {
    **{
        name: _Func(getattr(np, name))
        for name in (
            ("log", "log1p", "log2", "log10", "exp", "sqrt", "abs")
            + ("sin", "cos", "tanh", "floor", "ceil", "sign", "square")
        )
    },
    "round": _Func(_round, (1, 2), int_tail=True),
    "min": _Func(lambda *a: np.minimum.reduce(a), (1, None), "minimum"),
    "max": _Func(lambda *a: np.maximum.reduce(a), (1, None), "maximum"),
    "clip": _Func(np.clip, (3, 3)),
    "where": _Func(_where, (3, 3)),
    "isnull": _Func(lambda x: np.isnan(x).astype(float), np_name="isnan"),
}
# np.<name> -> canonical whitelist name (same function, numpy spelling).
_NP_FUNCS = {f.np_name or name: name for name, f in _FUNCS.items()}


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
        _compile_expr(self.expr, declared)
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


# Named refusals for constructs the compiler never accepts.
_REFUSED_NODES: dict[type, str] = {
    ast.Lambda: "lambda",
    ast.ListComp: "a comprehension",
    ast.GeneratorExp: "a comprehension",
    ast.DictComp: "a comprehension",
    ast.SetComp: "a comprehension",
    ast.List: "a list literal",
    ast.Tuple: "a tuple",
    ast.Dict: "a dict literal",
    ast.Set: "a set literal",
    ast.NamedExpr: "assignment (:=)",
    ast.Await: "await",
    ast.Yield: "yield",
    ast.YieldFrom: "yield",
    ast.Starred: "*unpacking",
    ast.JoinedStr: "an f-string",
    ast.FormattedValue: "an f-string",
    ast.Slice: "a slice",
}
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

Compiled = Callable[[dict[str, np.ndarray], int], np.ndarray]


def _logical(values: list[np.ndarray], combine) -> np.ndarray:
    """Element-wise boolean op on truthiness (!= 0) -> 0/1; any NaN -> NaN."""
    out = combine([v != 0 for v in values]).astype(float)
    for v in values:
        out[np.isnan(v)] = np.nan
    return out


def _safe_div(op) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """``op(a, b)`` with NaN where ``|b|`` is ~0 (``/``, ``//``, ``%``)."""

    def run(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.full(len(a), np.nan, dtype=float)
        op(a, b, out=out, where=np.abs(b) >= _DIV_EPS)
        return out

    return run


_BINOPS: dict[type, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: _safe_div(np.divide),
    ast.FloorDiv: _safe_div(np.floor_divide),
    ast.Mod: _safe_div(np.mod),
    ast.Pow: np.power,
    ast.BitAnd: lambda a, b: _logical([a, b], np.logical_and.reduce),
    ast.BitOr: lambda a, b: _logical([a, b], np.logical_or.reduce),
}
_UNARYOPS: dict[type, Callable[[np.ndarray], np.ndarray]] = {
    ast.UAdd: lambda a: np.asarray(a, dtype=float),
    ast.USub: operator.neg,
    ast.Not: lambda a: _logical([a], lambda m: ~m[0]),
    ast.Invert: lambda a: _logical([a], lambda m: ~m[0]),
}
_CMPOPS: dict[type, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _op_label(op: ast.AST) -> str:
    return _OP_SYMBOLS.get(type(op), type(op).__name__)


class _Compiler:
    """Validate the canonical AST against the whitelist and build its evaluator.

    One pass: each ``_<NodeType>`` handler refuses what is not allowed and
    returns a ``(env, n) -> array`` closure. Names read are collected in
    ``cols`` (env key -> DataFrame column) and ``used_vars``.
    """

    def __init__(self) -> None:
        self.cols: dict[str, str] = {}
        self.used_vars: set[str] = set()

    def compile(self, node: ast.AST) -> Compiled:
        handler = getattr(self, f"_{type(node).__name__}", None)
        if handler is None:
            _refuse(_REFUSED_NODES.get(type(node), type(node).__name__))
        return handler(node)

    def _Expression(self, node: ast.Expression) -> Compiled:
        return self.compile(node.body)

    def _Constant(self, node: ast.Constant) -> Compiled:
        if isinstance(node.value, str):
            _refuse(
                f"text constant {_restore_vars(node.value)!r}",
                'only df["column"] takes a string',
            )
        if type(node.value) not in (int, float):
            _refuse(f"constant {node.value!r}", "only numbers are")
        value = float(node.value)
        return lambda env, n: np.full(n, value, dtype=float)

    def _Name(self, node: ast.Name) -> Compiled:
        key = node.id
        if key.startswith(_VAR_PREFIX):
            self.used_vars.add(key[len(_VAR_PREFIX) :])
        elif key.startswith("__") and key.endswith("__"):
            _refuse(f"dunder name {key}")
        elif key not in _ALLOWED_CONSTANTS:
            self.cols[key] = key.removeprefix(_COL_PREFIX)
        return lambda env, n: env[key]

    def _UnaryOp(self, node: ast.UnaryOp) -> Compiled:
        fn = _UNARYOPS.get(type(node.op)) or _refuse(_op_label(node.op))
        operand = self.compile(node.operand)
        return lambda env, n: fn(operand(env, n))

    def _BinOp(self, node: ast.BinOp) -> Compiled:
        fn = _BINOPS.get(type(node.op)) or _refuse(_op_label(node.op))
        left, right = self.compile(node.left), self.compile(node.right)
        return lambda env, n: fn(left(env, n), right(env, n))

    def _BoolOp(self, node: ast.BoolOp) -> Compiled:
        values = [self.compile(v) for v in node.values]
        reduce = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        return lambda env, n: _logical([v(env, n) for v in values], reduce.reduce)

    def _Compare(self, node: ast.Compare) -> Compiled:
        """Comparisons evaluate to 0/1; any NaN operand yields NaN."""
        fns = [_CMPOPS.get(type(op)) or _refuse(_op_label(op)) for op in node.ops]
        operands = [self.compile(v) for v in [node.left, *node.comparators]]

        def run(env: dict[str, np.ndarray], n: int) -> np.ndarray:
            values = [v(env, n) for v in operands]
            mask = np.ones(n, dtype=bool)
            for fn, left, right in zip(fns, values, values[1:], strict=False):
                mask &= fn(left, right)
            out = np.where(mask, 1.0, 0.0)
            out[np.any([np.isnan(v) for v in values], axis=0)] = np.nan
            return out

        return run

    def _Call(self, node: ast.Call) -> Compiled:
        assert isinstance(node.func, ast.Name)  # _Normalizer guarantees it
        fname, args = node.func.id, node.args
        if node.keywords:
            _refuse("keyword arguments", f"in {fname}()")
        fn = _FUNCS.get(fname)
        if fn is None:
            _not_allowed(
                f"unknown function {fname}() is not allowed in a formula; "
                f"allowed: {', '.join(sorted(_FUNCS))}"
            )
        if not args:
            _not_allowed(f"{fname}() needs at least one argument")
        lo, hi = fn.arity
        if lo == hi and len(args) != lo:
            _not_allowed(
                f"{fname}() takes {lo} argument"
                f"{'' if lo == 1 else 's'}, got {len(args)}"
            )
        if hi is not None and len(args) > hi:
            _not_allowed(f"{fname}() takes at most {hi} arguments")
        if fn.int_tail and not all(
            isinstance(a, ast.Constant) and type(a.value) is int for a in args[1:]
        ):
            _not_allowed(f"{fname}() second argument must be an integer constant")
        compiled = [self.compile(a) for a in args]
        return lambda env, n: fn.impl(*(c(env, n) for c in compiled))


def _compile_expr(
    expr: str, declared_vars: set[str]
) -> tuple[Compiled, dict[str, str]]:
    """Parse, normalize, whitelist-check and compile ``expr``.

    Returns the evaluator and its env key -> column map. Raises ValueError
    naming the refused construct.
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
    compiler = _Compiler()
    run = compiler.compile(_Normalizer().visit(tree))
    unknown = compiler.used_vars - declared_vars
    if unknown:
        raise ValueError(
            "formula: unknown @variable "
            f"{sorted(unknown)}; declare it in `variables`"
        )
    return run, compiler.cols


_STAT_FUNCS: dict[str, Callable[[pd.Series], float]] = {
    "mean": pd.Series.mean,
    "median": pd.Series.median,
    "std": lambda s: s.std(ddof=0),
    "min": pd.Series.min,
    "max": pd.Series.max,
    "q25": lambda s: s.quantile(0.25),
    "q75": lambda s: s.quantile(0.75),
}


def _stat(series: pd.Series, stat: str) -> float | None:
    s = pd.to_numeric(series, errors="coerce")
    if stat == "count":
        return float(s.notna().sum())
    s = s.dropna()
    return None if s.empty else _json(float(_STAT_FUNCS[stat](s)))


def _fit_formula(df: pd.DataFrame, params: FormulaParams) -> dict:
    out: dict[str, float | None] = {}
    for var in params.variables:
        if var.column not in df.columns:
            raise KeyParamsError(
                f"formula: unknown column {var.column!r} for variable {var.name!r}"
            )
        out[var.name] = _stat(df[var.column], var.stat)
    return {"variables": out}


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
    run, cols = _compile_expr(params.expr, {v.name for v in params.variables})
    n = len(df)
    env: dict[str, np.ndarray] = {
        name: np.full(n, value, dtype=float)
        for name, value in _CONSTANT_VALUES.items()
    }
    for key, col in cols.items():
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
        result = run(env, n)
    result = np.asarray(result, dtype=float)
    result[~np.isfinite(result)] = np.nan
    return df.assign(**{params.name: result})
