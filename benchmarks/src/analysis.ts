/**
 * Benchmark results analysis
 */

import type {
  BenchmarkTiming,
  BenchmarkComparison,
  BenchmarkSummary,
  BenchmarkCase
} from './types';

export function compareResults(
  specs: BenchmarkCase[],
  numpyResults: BenchmarkTiming[],
  numpyjsResults: BenchmarkTiming[]
): BenchmarkComparison[] {
  const comparisons: BenchmarkComparison[] = [];

  for (let i = 0; i < specs.length; i++) {
    const spec = specs[i]!;
    const numpy = numpyResults[i]!;
    const numpyjs = numpyjsResults[i]!;

    comparisons.push({
      name: spec.name,
      category: spec.category,
      numpy,
      numpyjs,
      ratio: numpyjs.mean_ms / numpy.mean_ms
    });
  }

  return comparisons;
}

export function calculateSummary(comparisons: BenchmarkComparison[]): BenchmarkSummary {
  const ratios = comparisons.map((c) => c.ratio);

  const sum = ratios.reduce((a, b) => a + b, 0);
  const avg_slowdown = sum / ratios.length;

  const sorted = [...ratios].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median_slowdown =
    sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;

  const best_case = Math.min(...ratios);
  const worst_case = Math.max(...ratios);

  return {
    avg_slowdown,
    median_slowdown,
    best_case,
    worst_case,
    total_benchmarks: comparisons.length
  };
}

export function groupByCategory(
  comparisons: BenchmarkComparison[]
): Map<string, BenchmarkComparison[]> {
  const groups = new Map<string, BenchmarkComparison[]>();

  for (const comparison of comparisons) {
    const existing = groups.get(comparison.category) || [];
    existing.push(comparison);
    groups.set(comparison.category, existing);
  }

  return groups;
}

export function getCategorySummaries(
  comparisons: BenchmarkComparison[]
): Map<string, { avg_slowdown: number; count: number }> {
  const groups = groupByCategory(comparisons);
  const summaries = new Map<string, { avg_slowdown: number; count: number }>();

  for (const [category, items] of groups) {
    const ratios = items.map((item) => item.ratio);
    const avg_slowdown = ratios.reduce((a, b) => a + b, 0) / ratios.length;

    summaries.set(category, {
      avg_slowdown,
      count: items.length
    });
  }

  return summaries;
}

export function formatDuration(ms: number): string {
  if (ms < 1) {
    return `${(ms * 1000).toFixed(2)}μs`;
  } else if (ms < 1000) {
    return `${ms.toFixed(3)}ms`;
  } else {
    return `${(ms / 1000).toFixed(2)}s`;
  }
}

export function formatRatio(ratio: number): string {
  return `${ratio.toFixed(2)}x`;
}

export function printResults(comparisons: BenchmarkComparison[], summary: BenchmarkSummary): void {
  const groups = groupByCategory(comparisons);

  console.log('\n' + '='.repeat(80));
  console.log('BENCHMARK RESULTS');
  console.log('='.repeat(80));

  // Print by category
  for (const [category, items] of groups) {
    console.log(`\n[${category.toUpperCase()}]`);

    for (const item of items) {
      const { name, numpy, numpyjs, ratio } = item;
      const color = ratio < 2 ? '\x1b[32m' : ratio < 5 ? '\x1b[33m' : '\x1b[31m';
      const reset = '\x1b[0m';

      console.log(
        `  ${name.padEnd(45)} ` +
          `NumPy: ${formatDuration(numpy.mean_ms).padStart(10)} | ` +
          `numpy-ts: ${formatDuration(numpyjs.mean_ms).padStart(10)} | ` +
          `${color}${formatRatio(ratio).padStart(8)}${reset}`
      );
    }
  }

  // Print summary
  console.log('\n' + '='.repeat(80));
  console.log('SUMMARY');
  console.log('='.repeat(80));
  console.log(`Total benchmarks: ${summary.total_benchmarks}`);
  console.log(`Average slowdown: ${formatRatio(summary.avg_slowdown)}`);
  console.log(`Median slowdown:  ${formatRatio(summary.median_slowdown)}`);
  console.log(`Best case:        ${formatRatio(summary.best_case)}`);
  console.log(`Worst case:       ${formatRatio(summary.worst_case)}`);

  // Print category summaries
  const categorySummaries = getCategorySummaries(comparisons);
  console.log('\nBy Category:');
  for (const [category, data] of categorySummaries) {
    console.log(`  ${category.padEnd(15)} ${formatRatio(data.avg_slowdown).padStart(8)}`);
  }

  console.log('='.repeat(80) + '\n');
}
