# Very large CSV: windowed rows and whole-file statistics (engine `duckdb`)

Scope: a workspace role whose `x` is `{"kind": "csv", "path": ..., "engine": "duckdb"}`
(optional extra `large`), with no `y` file, no merge, no `label_last`, and steps that are
only `filter_rows` / `drop_columns`. Such a role is never built as a pandas frame by the
calls below: DuckDB reads the parquet cached for the CSV (`sources/csv_duckdb.py`).
Any other role, step or option takes the unchanged pandas path (same request, same
response, no `engine` field). Code: `workspace/large.py`.

Same calls, same request bodies at the three doors:

| call | Python (`dtk_engine.`) | HTTP |
|---|---|---|
| windowed rows | `workspace_rows(ws, role, version, offset, limit, columns, filter, sort)` | `POST /api/workspace/rows` |
| statistics | `column_profiles(ws, role, version, columns, sample_rows)` | `POST /api/workspace/profiles` |
| head + shape | `preview_workspace(ws, role, head_rows)` | `POST /api/workspace/preview` |
| columns | `source_columns(spec)` | `POST /api/source/columns` |
| shape | `list_workspace_summaries()` | `GET /api/workspaces/summaries` |

## Windowed rows

Rows `[offset, offset + limit)` and the row count, read by DuckDB: the time does not
depend on `offset`, the memory on `limit` only. Response (the pandas one, plus `engine`):

```json
{
  "engine": "duckdb",
  "columns": [{"name": "age", "dtype": "float64", "kind": "number", "semantic": "numeric",
               "approximate": ["kind", "semantic"]}],
  "rows": [{"age": 43.0, "_rid": 2000000}],
  "total": 4000000,
  "total_unfiltered": 4000000,
  "version": 0
}
```

- `_rid`: the row's position in the CSV (after the header), kept through filters.
- `filter` / `sort` (view only) run in DuckDB too (sort: stable, missing last); a filter
  condition DuckDB cannot reproduce exactly falls back to pandas.
- `kind` / `semantic` come from the whole-file statistics below (cached per file and
  steps), hence `approximate`.

## Statistics

Always computed on the whole table (after the steps), never on the displayed rows nor
on a sample. `sample_rows` is accepted and ignored on this path (`sampled: false`).
Response (the pandas fields, plus `engine`, `n_rows`, and per column `stats` and
`approximate`):

```json
{
  "engine": "duckdb",
  "n_rows": 4000000,
  "version": 0,
  "columns": [{
    "name": "income", "kind": "number",
    "count": 4000000, "missing": 0, "distinct": 3911022,
    "stats": {"min": -999.0, "max": 512000.1, "mean": 26300.4, "std": 17000.2,
              "q1": 14600.0, "median": 22000.5, "q3": 32900.7},
    "histogram": {"edges": [...], "counts": [...]},
    "top_values": null, "iqr_bounds": {"lo": -12850.0, "hi": 60350.7}, "outliers": 103221,
    "sentinel_candidates": [{"value": -999, "count": 40112}],
    "variants": null, "looks_like_dates": false, "numbers_as_text": false,
    "currency_as_text": null, "skewed": true,
    "suggested_params": {"top_k": 10, "bins": 87, "log_scale": true},
    "approximate": ["distinct", "kind", "stats.q1", "stats.median", "stats.q3",
                    "iqr_bounds", "outliers", "skewed", "suggested_params"]
  }]
}
```

`approximate` lists the fields of that column whose value is approximate, or derived from
one. A field not listed is exact. `stats` is present for number columns only.

| field | how | exact? |
|---|---|---|
| `count`, `missing`, `stats.min/max/mean/std` | DuckDB aggregate | exact |
| `histogram` | DuckDB, `np.histogram` bins on exact min / max | exact |
| `distinct` | `approx_count_distinct` (HyperLogLog, ~1-2 % error) | approximate |
| `stats.q1/median/q3` | `approx_quantile` (t-digest) | approximate |
| `iqr_bounds`, `outliers` | fences from the approximate quartiles; count exact for those fences | approximate |
| `skewed`, `suggested_params` | from approximate median / quartiles / distinct | approximate |
| `kind` (and `semantic` in rows) | the pandas rules, evaluated in SQL on approximate `distinct` | approximate |
| `top_values` | exact `GROUP BY` when `distinct` <= 100 000, else `approx_top_k` values with exact counts | exact / approximate |
| `variants` | `distinct` of the raw and of the trimmed lower-case strings | approximate |
| `looks_like_dates`, `numbers_as_text`, `currency_as_text` | the pandas rules ported to SQL regexes, on every row | exact |
| `sentinel_candidates` | the pandas rules in SQL, on every row; text columns parse numbers with `TRY_CAST` (not `pd.to_numeric`) | exact for numbers, approximate for text |
