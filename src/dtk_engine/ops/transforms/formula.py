"""Formula transform: a whitelisted arithmetic expression over columns and @variables.

Variables are statistics fitted on train and frozen for apply (test never
contributes). The expression is parsed with ``ast`` after rewriting ``@name`` to
a reserved identifier; evaluation is vectorised with numpy — never ``eval``.
"""

from __future__ import annotations

import ast
import re
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from dtk_engine.errors import KeyParamsError
from dtk_engine.ops._util import json_scalar as _json
from dtk_engine.params import column_field
from dtk_engine.transform_registry import TransformParams, transform

IDENTIFIER = r"^[A-Za-z_][A-Za-z0-9_]*$"
_VAR_PREFIX = "_dtk_var_"
_VAR_TOKEN = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)")
_DIV_EPS = 1e-12

_ALLOWED_FUNCS = frozenset(
    {"log", "log1p", "exp", "sqrt", "abs", "round", "min", "max"}
)
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_UNARYOPS = (ast.UAdd, ast.USub)
_STATS = Literal["mean", "median", "std", "min", "max", "q25", "q75", "count"]


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


def _refuse(construct: str, detail: str = "") -> None:
    msg = f"formula: refused {construct}"
    if detail:
        msg += f" ({detail})"
    raise ValueError(msg)


def _check_expr(expr: str, declared_vars: set[str]) -> ast.Expression:
    """Parse and whitelist-walk ``expr``; return the AST (or raise ValueError)."""
    rewritten = _rewrite_vars(expr)
    if "@" in rewritten:
        _refuse("@variable", "invalid or incomplete @name")
    try:
        tree = ast.parse(rewritten, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"formula: invalid expression ({exc.msg})") from exc

    used_vars: set[str] = set()

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            walk(node.body)
            return
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) or isinstance(node.value, bool):
                _refuse("Constant", f"non-numeric {node.value!r}")
            return
        if isinstance(node, ast.Name):
            if node.id.startswith(_VAR_PREFIX):
                used_vars.add(node.id[len(_VAR_PREFIX) :])
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _UNARYOPS):
                _refuse("UnaryOp", type(node.op).__name__)
            walk(node.operand)
            return
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _BINOPS):
                _refuse("BinOp", type(node.op).__name__)
            walk(node.left)
            walk(node.right)
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                _refuse("Call", "function must be a bare name")
            if node.keywords:
                _refuse("keyword arguments", node.func.id)
            if node.func.id not in _ALLOWED_FUNCS:
                _refuse(
                    "unknown function",
                    f"{node.func.id}(); allowed: {', '.join(sorted(_ALLOWED_FUNCS))}",
                )
            if not node.args:
                _refuse("Call", f"{node.func.id}() needs at least one argument")
            for arg in node.args:
                walk(arg)
            return
        # Named refusals for common injection / disallowed constructs.
        for typ, label in (
            (ast.Attribute, "Attribute"),
            (ast.Subscript, "Subscript"),
            (ast.Lambda, "Lambda"),
            (ast.List, "List"),
            (ast.Tuple, "Tuple"),
            (ast.Dict, "Dict"),
            (ast.Set, "Set"),
            (ast.Compare, "Compare"),
            (ast.BoolOp, "BoolOp"),
            (ast.IfExp, "IfExp"),
            (ast.ListComp, "ListComp"),
            (ast.GeneratorExp, "GeneratorExp"),
            (ast.DictComp, "DictComp"),
            (ast.SetComp, "SetComp"),
            (ast.Await, "Await"),
            (ast.Yield, "Yield"),
            (ast.Starred, "Starred"),
            (ast.JoinedStr, "JoinedStr"),
            (ast.FormattedValue, "FormattedValue"),
        ):
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


def _column_names(tree: ast.AST) -> set[str]:
    cols: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if not node.id.startswith(_VAR_PREFIX):
                cols.add(node.id)

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
    if name == "exp":
        return np.exp(args[0])
    if name == "sqrt":
        return np.sqrt(args[0])
    if name == "abs":
        return np.abs(args[0])
    if name == "round":
        if len(args) == 1:
            return np.round(args[0])
        return np.round(args[0], args[1].astype(int))
    if name == "min":
        return np.minimum.reduce(args)
    return np.maximum.reduce(args)  # max


def _eval_node(node: ast.AST, env: dict[str, np.ndarray], n: int) -> np.ndarray:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env, n)
    if isinstance(node, ast.Constant):
        return np.full(n, float(node.value), dtype=float)
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        a = _eval_node(node.operand, env, n)
        return -a if isinstance(node.op, ast.USub) else np.asarray(a, dtype=float)
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
            out = np.full(n, np.nan, dtype=float)
            ok = np.abs(b) >= _DIV_EPS
            np.divide(a, b, out=out, where=ok)
            return out
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            return np.power(a, b)
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)
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
        "numbers, functions and train-fitted @variables."
    ),
)
def formula(df: pd.DataFrame, params: FormulaParams, state: dict) -> pd.DataFrame:
    """Add a float column from a safe expression; @variables reuse train values."""
    tree = _check_expr(params.expr, {v.name for v in params.variables})
    n = len(df)
    env: dict[str, np.ndarray] = {}
    for col in _column_names(tree):
        if col not in df.columns:
            raise KeyParamsError(f"formula: unknown column {col!r}")
        env[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
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
