# Benchmark History

This directory contains historical benchmark results automatically collected by GitHub Actions.

## Purpose

- Track performance over time across commits on the `main` branch
- Detect performance regressions automatically
- Provide historical data for performance analysis

## Files

- `benchmark-YYYYMMDD-HHMMSS-<commit>.json` - Individual benchmark runs with metadata
- `BENCHMARK-HISTORY.md` - Generated summary report with trends

## How It Works

1. When code is pushed to `main`, the benchmark workflow runs
2. Results are saved with commit metadata
3. Comparison with previous run detects regressions (>50% slower)
4. If regression detected, an issue is automatically created
5. All results are committed back to this directory

## Viewing Results

- **Latest Summary**: See [BENCHMARK-HISTORY.md](./BENCHMARK-HISTORY.md) (generated after first benchmark run)
- **Individual Runs**: Browse the JSON files in this directory
- **Trends**: The summary shows trends over the last 10 runs

## Local Development

Local benchmark results are stored in `benchmarks/results/` (gitignored).
This directory is only used by CI/CD for tracking historical data.
