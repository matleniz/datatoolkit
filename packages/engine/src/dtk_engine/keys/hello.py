"""Plumbing check for the whole chain (engine -> contract -> client -> front)."""

import pandas as pd
import plotly.express as px
from pydantic import Field

from dtk_engine.params import KeyParams
from dtk_engine.registry import key
from dtk_engine.result import Result


class Params(KeyParams):
    n: int = Field(default=10, ge=1, le=1000, description="Number of points")


@key(id="hello", title="Hello", category="demo", description="Sum of squares 0..n-1.")
def run(params: Params) -> Result:
    df = pd.DataFrame({"x": range(params.n)})
    df["y"] = df["x"] ** 2
    result = Result(metrics={"n": params.n, "sum": int(df["y"].sum())})
    result.add_table("squares", df)
    result.add_figure("squares", px.line(df, x="x", y="y"))
    return result
