#!/usr/bin/env node
/**
 * Main benchmark orchestrator
 * Runs benchmarks for both Python NumPy and numpy-ts, compares results
 */

import * as fs from 'fs';
import * as path from 'path';
import { getBenchmarkSpecs, filterByCategory } from './specs';
import { runBenchmarks } from './runner';
import { runPythonBenchmarks } from './python-runner';
import { compareResults, calculateSummary, printResults } from './analysis';
import { generateHTMLReport } from './visualization';
import { generatePNGChart } from './chart-generator';
import type { BenchmarkMode, BenchmarkOptions, BenchmarkReport } from './types';

// Read version from root package.json
const packageJson = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '../../package.json'), 'utf-8')
);

async function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  const options: BenchmarkOptions = {
    mode: 'standard',
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg === '--quick') {
      options.mode = 'quick';
    } else if (arg === '--standard') {
      options.mode = 'standard';
    } else if (arg === '--full') {
      options.mode = 'full';
    } else if (arg === '--category' && i + 1 < args.length) {
      options.category = args[++i];
    } else if (arg === '--output' && i + 1 < args.length) {
      options.output = args[++i];
    } else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    }
  }

  console.log('🚀 NumPy vs numpy-ts Benchmark Suite\n');
  console.log(`Mode: ${options.mode}`);

  // Get benchmark specifications
  let specs = getBenchmarkSpecs(options.mode || 'standard');

  if (options.category) {
    console.log(`Category filter: ${options.category}`);
    specs = filterByCategory(specs, options.category);

    if (specs.length === 0) {
      console.error(`❌ No benchmarks found for category: ${options.category}`);
      process.exit(1);
    }
  }

  console.log(`Total benchmarks: ${specs.length}\n`);

  try {
    // Run numpy-ts benchmarks
    console.log('Running numpy-ts benchmarks...');
    const numpyjsResults = await runBenchmarks(specs);

    // Run Python NumPy benchmarks
    console.log('\nRunning Python NumPy benchmarks...');
    const { results: numpyResults, pythonVersion, numpyVersion } = await runPythonBenchmarks(specs);

    // Compare results
    const comparisons = compareResults(specs, numpyResults, numpyjsResults);
    const summary = calculateSummary(comparisons);

    // Print results to console
    printResults(comparisons, summary);

    // Create report
    const report: BenchmarkReport = {
      timestamp: new Date().toISOString(),
      environment: {
        node_version: process.version,
        python_version: pythonVersion,
        numpy_version: numpyVersion,
        numpyjs_version: packageJson.version,
      },
      results: comparisons,
      summary,
    };

    // Save results
    const resultsDir = path.resolve(__dirname, '../results');
    const plotsDir = path.resolve(resultsDir, 'plots');

    // Ensure directories exist
    if (!fs.existsSync(resultsDir)) {
      fs.mkdirSync(resultsDir, { recursive: true });
    }
    if (!fs.existsSync(plotsDir)) {
      fs.mkdirSync(plotsDir, { recursive: true });
    }

    // Save JSON results
    const jsonPath = options.output
      ? path.resolve(options.output)
      : path.join(resultsDir, 'latest.json');
    fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
    console.log(`Results saved to: ${jsonPath}`);

    // Save historical results
    const historyDir = path.join(resultsDir, 'history');
    if (!fs.existsSync(historyDir)) {
      fs.mkdirSync(historyDir, { recursive: true });
    }
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const historyPath = path.join(historyDir, `benchmark-${timestamp}.json`);
    fs.writeFileSync(historyPath, JSON.stringify(report, null, 2));

    // Generate HTML report
    const htmlPath = path.join(plotsDir, 'latest.html');
    generateHTMLReport(report, htmlPath);
    console.log(`HTML report saved to: ${htmlPath}`);

    // Generate PNG chart
    const pngPath = path.join(plotsDir, 'latest.png');
    await generatePNGChart(report, pngPath);
    console.log(`PNG chart saved to: ${pngPath}`);

    console.log(`\nView report: open ${htmlPath}`);
  } catch (error) {
    console.error('❌ Benchmark failed:', error);
    process.exit(1);
  }
}

function printHelp() {
  console.log(`
NumPy vs numpy-ts Benchmark Suite

Usage:
  npm run bench [options]

Options:
  --quick              Run quick benchmarks (small sizes, few iterations)
  --standard           Run standard benchmarks (default)
  --full               Run full benchmarks (large sizes, many iterations)
  --category <name>    Run only benchmarks in specified category
  --output <path>      Save JSON results to specified path
  --help, -h           Show this help message

Categories:
  creation             Array creation (zeros, ones, arange, etc.)
  arithmetic           Arithmetic operations (add, multiply, etc.)
  linalg               Linear algebra (matmul, transpose)
  reductions           Reductions (sum, mean, max, min)
  reshape              Reshape operations (reshape, flatten, ravel)

Examples:
  npm run bench                           # Run standard benchmarks
  npm run bench -- --quick                # Run quick benchmarks
  npm run bench -- --category linalg      # Run only linalg benchmarks
  npm run bench -- --full --output out.json  # Full benchmarks, save to out.json
`);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
