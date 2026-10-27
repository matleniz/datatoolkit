# Very large CSV: windowed rows and whole-file statistics (engine `duckdb`)

Scope: a workspace role whose `x` is `{"kind": "csv", "path": ..., "engine": "duckdb"}`
(optional extra `large`), with no `y` file, no merge, no `label_last`, and steps that are
only `filter_rows` / `drop_columns`. Such a role is never built as a pandas frame by the
calls below: DuckDB reads the parquet cached for the CSV (`sources/csv_duckdb.py`).
Any other role, step or option takes the unchanged pandas path (same request, same
response, no `engine` field). Code: `workspace/large.py`, `workspace/large_stats.py`.

Same calls, same request bodies at the three doors:

| call | Python (`dtk_engine.`) | HTTP |
|---|---|---|
| windowed rows | `workspace_rows(ws, role, version, offset, limit, columns, filter, sort)` | `POST /api/workspace/rows` |
| statistics | `column_profiles(ws, role, version, columns, sample_rows)` | `POST /api/workspace/profiles` |
| head + shape | `preview_workspace(ws, role, head_rows)` | `POST /api/workspace/preview` |
| columns | `source_columns(spec)` | `POST /api/source/columns` |
| shape | `list_workspace_summaries()` | `GET /api/workspaces/summaries` |
| column names for notes | `column_notes(ws, role, version)` | `POST /api/workspace/column-notes` |

Known limit: `column_notes` reads names from the plan only while every step is pushable
(`filter_rows` / `drop_columns`). A rename step sends the role through pandas (full frame), as
do `preview_step`, `align_report` and formula evaluation.

## Windowed rows

Rows `[offset, offset + limit)` and the row count, read by DuckDB: the time does not
depend on `offset`, the memory depends on `limit` only. Response (the pandas one, plus
`engine` and `columns[].approximate`):

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
- `dtype`: the dtype of the whole column as pandas reads the file (an int column with a
  missing value is `float64`, a bool one `object`), whatever the page holds.
- `filter` / `sort` (view only) run in DuckDB too: stable sort, missing values last;
  a sorted page sorts the sort keys and row numbers only (spilling), then reads the page.
  Both steps stay in the 128 MB limit: the sort runs on one thread when a key is text, and
  the page's rows (scattered over every row group) are read on one thread.
  A filter condition DuckDB cannot reproduce exactly (a value of the wrong type) falls
  back to pandas, which builds the frame.
- `kind` / `semantic` come from the whole-table statistics below, of the role after its
  steps (pandas computes them on the rows the view filter keeps), hence `approximate`.
  The first call computes them for every column (cached per file and steps).

## Statistics

Always computed on the whole table (after the steps), never on the displayed rows nor on
a sample. `sample_rows` is accepted and ignored on this path (`sampled: false`,
`n_sampled_out: 0`). Response (the pandas fields, plus `engine`, `n_rows`, and per column
`stats` and `approximate`):

```json
{
  "engine": "duckdb",
  "n_rows": 4000000,
  "version": 0,
  "columns": [{
    "name": "income", "kind": "number",
    "count": 4000000, "missing": 0, "distinct": 2559817,
    "stats": {"min": -999.0, "max": 512000.1, "mean": 26300.4, "std": 17000.2,
              "q1": 14600.0, "median": 22000.5, "q3": 32900.7},
    "histogram": {"edges": [], "counts": []},
    "top_values": null, "iqr_bounds": {"lo": -12850.0, "hi": 60350.7}, "outliers": 103221,
    "sentinel_candidates": [{"value": "-999", "count": 40112}],
    "variants": null, "looks_like_dates": false, "numbers_as_text": false,
    "currency_as_text": null, "skewed": true,
    "suggested_params": {"top_k": 10, "bins": 87, "log_scale": true},
    "approximate": ["kind", "distinct", "stats.q1", "stats.median", "stats.q3", "iqr_bounds",
                    "outliers", "skewed", "sentinel_candidates", "suggested_params"]
  }]
}
```

`approximate` lists the fields of that column whose value is approximate, or derived from
one. A field not listed is exact. `stats` is present for number columns only.

Up to 1 000 000 rows (after the steps) everything but `kind` is exact. Above, the costly
values use DuckDB's approximate algorithms, on every row:

| field | how | exact? |
|---|---|---|
| `count`, `missing`, `stats.min/max/mean/std`, `histogram` | DuckDB aggregates; `np.histogram` bins on the exact min / max | exact |
| `distinct` | `approx_count_distinct` (HyperLogLog, ~2 % error); exact `count(DISTINCT)` when the estimate is <= 100 000 | approximate above 100 000 |
| `stats.q1/median/q3` | `approx_quantile` (t-digest) | approximate |
| `iqr_bounds`, `outliers`, `skewed` | from the quartiles (the outlier count is exact for those fences) | approximate |
| `top_values` | exact `GROUP BY`; above 100 000 distinct, `approx_top_k` values with exact counts | approximate above 100 000 |
| `variants` | distinct of the raw and of the normalized text; with approximate counts, reported only when they differ by more than 5 % | approximate above 100 000 |
| `suggested_params` | from `distinct` and the quartiles | approximate when they are |
| `kind` (and `semantic` in rows) | the pandas rules, ported to SQL | always listed |
| `sentinel_candidates` | the pandas rules in SQL; text cells read as numbers with `TRY_CAST` (not `pd.to_numeric`) | listed for text columns |
| `looks_like_dates`, `numbers_as_text`, `currency_as_text` | the pandas regexes, rewritten for DuckDB (RE2), on every cell | exact |

Text rules run once per distinct value (weighted by its rows) when there are at most
100 000 of them, else once per cell.

## Memory

DuckDB runs under a 128 MB `memory_limit` with 4 threads (it spills to a temporary
directory beyond it; more threads at that limit make it thrash), for the CSV -> parquet
conversion (retried once at 1 GB if it runs out) and for every query above. A page holds
`limit` rows; statistics hold one column's aggregates. The process peak is the
interpreter's own memory plus DuckDB's limit, whatever the file size.

A front fires several calls at once (rows pages, profiles, shape), all cold on the first
load. At most 3 DuckDB connections are open together (`large.MAX_CONNECTIONS`, the others
wait), the CSV -> parquet conversion runs once (concurrent calls wait for it), and a cached
value (kinds, column facts, shapes) is computed by one call per key, the others wait
(`cache.LRU.memo`). Without this, 7 cold calls at once took 58 s and peaked at 910 MB
(each recomputed the kinds of every column); now 12 s and 488 MB.

Measured on a 618 MB, 4M x 16 CSV (mixed numbers, categories, ids, free text, dates as
text) and on the same file doubled (1.24 GB, 8M rows), through `dtk-api` with a fresh
`DTK_HOME`, peak = `VmHWM` of the process (~270 MB right after start):

| stage | before (main 43ea5ad), 618 MB | after, 618 MB | after, 1.24 GB |
|---|---|---|---|
| source columns (cold: converts the CSV) | 7.9 s, peak 1602 MB | 18.2 s | 30.0 s |
| preview (Sources) | 1.8 s, peak 1907 MB | 0.18 s | 0.19 s |
| rows offset 0 (computes the kinds) | killed at 2.5 GB | 14.6 s | 26.2 s |
| profiles (every column) | not reached | 3.1 s | 7.0 s |
| rows at offset 1.2M / 2M / 4M | not reached | 0.10-0.14 s | 0.15-0.17 s |
| rows sorted (income desc) at offset 2M | not reached | 0.83 s | 1.48 s |
| rows filtered (city = Paris) at offset 100k | not reached | 0.34 s | 0.37 s |
| parquet export | not reached | 13.0 s | 37.7 s |
| **process peak (VmHWM)** | **> 2.5 GB (killed)** | **483 MB** | **463 MB** |

Sorted pages (issue #333), same 618 MB file, 2 GB cgroup (`systemd-run -p MemoryMax=2G`),
one `dtk-api` per run, run-cumulative maxima up to and including each column's sorts (`VmHWM` / cgroup `memory.peak`);
the process was at 458-476 MB / 520-690 MB before the first sort. Pages of 100 rows at
offset 0 and 2M, ascending and descending (before = main 331d789):

| sort column | before | after | peak after (VmHWM / cgroup) |
|---|---|---|---|
| `income`, `score` (float) | 422 OutOfMemory (except `score` desc, offset 0) | ok, 0.3-0.7 s | 489 MB / 811 MB |
| `customer` (text, 4M distinct) | 422 OutOfMemory | ok, 1.2-1.4 s | 482 MB / 811 MB |
| `comment` (long text, 4M distinct) | 422 OutOfMemory | ok, 2-5 s | 478 MB / 811 MB |
| `city`, `age` | ok | ok, 0.1-0.3 s | 499 MB / 597 MB (`city`), 514 MB / 660 MB (`age`, later in the run) |

The cause was the second query, not the sort: 4 threads each decode whole row groups of
every column to pick the page's scattered rows. Peaks are the max over the whole run, not
per case (the cgroup includes the page cache of the parquet).

The pandas engine on the 618 MB file is killed at 2.5 GB during the first call. Times
vary up to 2x on a shared machine.

## Whole-frame calls are refused above a size

Analyses (`/keys/<key>/run` on a `dataset` source), merges, label joins and any other step
than `filter_rows` / `drop_columns` need a pandas frame. Building one from a very large
CSV used to take every byte of the memory cap (the real Studio journey fires five analyses
at once: 4M x 16 CSV, killed at a 2 GB cap, #334). The reader now estimates the frame
(the parquet's uncompressed size x 6) and raises a `SourceError` (HTTP 422) above
`DTK_MAX_FRAME_MB` (default 1024 MB; the 618 MB file is estimated at ~2.4 GB). Rows,
statistics and preview keep working. Under a 2 GB cap, the journey's traffic (profiles + 5
analyses + rows at 1500 / 2100 / 1.2M / 2M / 4M): before, killed at 1.9 GB (4.7 s after
the analyses start); after, peak 582 MB, analyses answer 422, rows 0.1 s.
