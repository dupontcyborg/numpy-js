/**
 * TypeScript/numpy-ts benchmark runner
 */

import { performance } from 'perf_hooks';
import * as np from '../../src/index';
import type { BenchmarkCase, BenchmarkTiming, BenchmarkSetup } from './types';
import type { NDArray } from '../../src/core/ndarray';

function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1]! + sorted[mid]!) / 2
    : sorted[mid]!;
}

function std(arr: number[]): number {
  const m = mean(arr);
  const variance = arr.reduce((sum, x) => sum + (x - m) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

function setupArrays(setup: BenchmarkSetup): Record<string, any> {
  const arrays: Record<string, any> = {};

  for (const [key, spec] of Object.entries(setup)) {
    const { shape, dtype = 'float64', fill = 'zeros', value } = spec;

    // Handle scalar values
    if (key === 'n' || key === 'axis' || key === 'new_shape' || key === 'shape' || key === 'fill_value') {
      arrays[key] = shape[0];
      if (key === 'new_shape' || key === 'shape') {
        arrays[key] = shape;
      }
      continue;
    }

    // Create arrays
    if (fill === 'zeros') {
      arrays[key] = np.zeros(shape, dtype);
    } else if (fill === 'ones') {
      arrays[key] = np.ones(shape, dtype);
    } else if (fill === 'random') {
      // Create random-like data using arange for consistency
      const size = shape.reduce((a, b) => a * b, 1);
      const flat = np.arange(0, size, 1, dtype);
      arrays[key] = flat.reshape(...shape);
    } else if (fill === 'arange') {
      const size = shape.reduce((a, b) => a * b, 1);
      const flat = np.arange(0, size, 1, dtype);
      arrays[key] = flat.reshape(...shape);
    } else if (value !== undefined) {
      arrays[key] = np.ones(shape, dtype).multiply(value);
    }
  }

  return arrays;
}

function executeOperation(operation: string, arrays: Record<string, any>): any {
  // Array creation
  if (operation === 'zeros') {
    return np.zeros(arrays['shape']);
  } else if (operation === 'ones') {
    return np.ones(arrays['shape']);
  } else if (operation === 'empty') {
    return np.empty(arrays['shape']);
  } else if (operation === 'full') {
    return np.full(arrays['shape'], arrays['fill_value']);
  } else if (operation === 'arange') {
    return np.arange(0, arrays['n'], 1);
  } else if (operation === 'linspace') {
    return np.linspace(0, 100, arrays['n']);
  } else if (operation === 'logspace') {
    return np.logspace(0, 3, arrays['n']);
  } else if (operation === 'geomspace') {
    return np.geomspace(1, 1000, arrays['n']);
  } else if (operation === 'eye') {
    return np.eye(arrays['n']);
  } else if (operation === 'identity') {
    return np.identity(arrays['n']);
  } else if (operation === 'copy') {
    return np.copy(arrays['a']);
  } else if (operation === 'zeros_like') {
    return np.zeros_like(arrays['a']);
  } else if (operation === 'ones_like') {
    return np.ones_like(arrays['a']);
  } else if (operation === 'empty_like') {
    return np.empty_like(arrays['a']);
  } else if (operation === 'full_like') {
    return np.full_like(arrays['a'], 7);
  }

  // Arithmetic
  else if (operation === 'add') {
    return arrays['a'].add(arrays['b']);
  } else if (operation === 'subtract') {
    return arrays['a'].subtract(arrays['b']);
  } else if (operation === 'multiply') {
    return arrays['a'].multiply(arrays['b']);
  } else if (operation === 'divide') {
    return arrays['a'].divide(arrays['b']);
  }

  // Linear algebra
  else if (operation === 'matmul') {
    return arrays['a'].matmul(arrays['b']);
  } else if (operation === 'transpose') {
    return arrays['a'].transpose();
  }

  // Reductions
  else if (operation === 'sum') {
    const axis = arrays['axis'];
    return arrays['a'].sum(axis);
  } else if (operation === 'mean') {
    const axis = arrays['axis'];
    return arrays['a'].mean(axis);
  } else if (operation === 'max') {
    const axis = arrays['axis'];
    return arrays['a'].max(axis);
  } else if (operation === 'min') {
    const axis = arrays['axis'];
    return arrays['a'].min(axis);
  } else if (operation === 'prod') {
    const axis = arrays['axis'];
    return arrays['a'].prod(axis);
  } else if (operation === 'argmin') {
    const axis = arrays['axis'];
    return arrays['a'].argmin(axis);
  } else if (operation === 'argmax') {
    const axis = arrays['axis'];
    return arrays['a'].argmax(axis);
  } else if (operation === 'var') {
    const axis = arrays['axis'];
    return arrays['a'].var(axis);
  } else if (operation === 'std') {
    const axis = arrays['axis'];
    return arrays['a'].std(axis);
  } else if (operation === 'all') {
    const axis = arrays['axis'];
    return arrays['a'].all(axis);
  } else if (operation === 'any') {
    const axis = arrays['axis'];
    return arrays['a'].any(axis);
  }

  // Reshape
  else if (operation === 'reshape') {
    return arrays['a'].reshape(...arrays['new_shape']);
  } else if (operation === 'flatten') {
    return arrays['a'].flatten();
  } else if (operation === 'ravel') {
    return arrays['a'].ravel();
  } else if (operation === 'squeeze') {
    return arrays['a'].squeeze();
  }

  // Slicing
  else if (operation === 'slice') {
    return arrays['a'].slice('0:100', '0:100');
  }

  throw new Error(`Unknown operation: ${operation}`);
}

export async function runBenchmark(spec: BenchmarkCase): Promise<BenchmarkTiming> {
  const { name, operation, setup, iterations, warmup } = spec;

  // Setup arrays
  const arrays = setupArrays(setup);

  // Warmup
  for (let i = 0; i < warmup; i++) {
    executeOperation(operation, arrays);
  }

  // Benchmark
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    const result = executeOperation(operation, arrays);
    const end = performance.now();
    times.push(end - start);

    // Keep reference to prevent optimization
    void result;
  }

  return {
    name,
    mean_ms: mean(times),
    median_ms: median(times),
    min_ms: Math.min(...times),
    max_ms: Math.max(...times),
    std_ms: std(times)
  };
}

export async function runBenchmarks(specs: BenchmarkCase[]): Promise<BenchmarkTiming[]> {
  const results: BenchmarkTiming[] = [];

  console.error(`Node ${process.version}`);
  console.error(`Running ${specs.length} benchmarks...`);

  for (let i = 0; i < specs.length; i++) {
    const spec = specs[i]!;
    const result = await runBenchmark(spec);
    results.push(result);

    console.error(`  [${i + 1}/${specs.length}] ${spec.name}: ${result.mean_ms.toFixed(3)}ms`);
  }

  return results;
}
