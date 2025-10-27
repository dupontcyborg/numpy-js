/**
 * Benchmark type definitions
 */

import type { DType } from '../../src/core/dtype';

export interface BenchmarkSetup {
  [key: string]: {
    shape: number[];
    dtype?: DType;
    fill?: 'zeros' | 'ones' | 'random' | 'arange';
    value?: number;
  };
}

export interface BenchmarkCase {
  name: string;
  category: string;
  operation: string;
  setup: BenchmarkSetup;
  iterations: number;
  warmup: number;
}

export interface BenchmarkTiming {
  name: string;
  mean_ms: number;
  median_ms: number;
  min_ms: number;
  max_ms: number;
  std_ms: number;
  ops_per_sec: number; // Operations per second
  total_ops: number; // Total operations executed
  total_samples: number; // Number of timing samples taken
}

export interface BenchmarkComparison {
  name: string;
  category: string;
  numpy: BenchmarkTiming;
  numpyjs: BenchmarkTiming;
  ratio: number; // numpyjs / numpy (how many times slower)
}

export interface BenchmarkSummary {
  avg_slowdown: number;
  median_slowdown: number;
  best_case: number;
  worst_case: number;
  total_benchmarks: number;
}

export interface BenchmarkReport {
  timestamp: string;
  environment: {
    node_version: string;
    python_version?: string;
    numpy_version?: string;
    numpyjs_version: string;
  };
  results: BenchmarkComparison[];
  summary: BenchmarkSummary;
}

export type BenchmarkMode = 'quick' | 'standard';

export interface BenchmarkOptions {
  mode?: BenchmarkMode;
  category?: string;
  output?: string;
}
