/**
 * Benchmark specifications
 * Defines all benchmarks to run
 */

import type { BenchmarkCase, BenchmarkMode } from './types';

export function getBenchmarkSpecs(mode: BenchmarkMode = 'standard'): BenchmarkCase[] {
  // Array sizes vary by mode
  // Auto-calibration handles the iteration count
  const sizeConfig = {
    quick: {
      small: 1000,
      medium: [100, 100] as [number, number],
      large: [500, 500] as [number, number],
    },
    standard: {
      small: 1000,
      medium: [100, 100] as [number, number],
      large: [500, 500] as [number, number],
    },
    large: {
      small: 10000,
      medium: [316, 316] as [number, number], // ~100K elements
      large: [1000, 1000] as [number, number], // 1M elements
    },
  };

  const sizes = sizeConfig[mode] || sizeConfig.standard;

  const warmupConfig = {
    quick: {
      iterations: 1, // Not used with auto-calibration, kept for compatibility
      warmup: 3, // Less warmup for faster feedback
    },
    standard: {
      iterations: 1, // Not used with auto-calibration, kept for compatibility
      warmup: 10, // More warmup for stable results
    },
    large: {
      iterations: 1, // Not used with auto-calibration, kept for compatibility
      warmup: 5, // Moderate warmup for large arrays
    },
  };

  const config = warmupConfig[mode] || warmupConfig.standard;

  const { iterations, warmup } = config;
  const specs: BenchmarkCase[] = [];

  // ========================================
  // Array Creation Benchmarks
  // ========================================

  specs.push({
    name: `zeros [${sizes.small}]`,
    category: 'creation',
    operation: 'zeros',
    setup: {
      shape: { shape: [sizes.small] },
    },
    iterations,
    warmup,
  });

  if (Array.isArray(sizes.medium)) {
    specs.push({
      name: `zeros [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'zeros',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `ones [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'ones',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });
  }

  specs.push({
    name: `arange(${sizes.small})`,
    category: 'creation',
    operation: 'arange',
    setup: {
      n: { shape: [sizes.small] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `linspace(0, 100, ${sizes.small})`,
    category: 'creation',
    operation: 'linspace',
    setup: {
      n: { shape: [sizes.small] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `logspace(0, 3, ${sizes.small})`,
    category: 'creation',
    operation: 'logspace',
    setup: {
      n: { shape: [sizes.small] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `geomspace(1, 1000, ${sizes.small})`,
    category: 'creation',
    operation: 'geomspace',
    setup: {
      n: { shape: [sizes.small] },
    },
    iterations,
    warmup,
  });

  if (Array.isArray(sizes.medium)) {
    const eyeSize = sizes.medium[0]!;
    specs.push({
      name: `eye(${eyeSize})`,
      category: 'creation',
      operation: 'eye',
      setup: {
        n: { shape: [eyeSize] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `identity(${eyeSize})`,
      category: 'creation',
      operation: 'identity',
      setup: {
        n: { shape: [eyeSize] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `empty [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'empty',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `full [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'full',
      setup: {
        shape: { shape: sizes.medium },
        fill_value: { shape: [7] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `copy [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'copy',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `zeros_like [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'zeros_like',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // Arithmetic Benchmarks
  // ========================================

  if (Array.isArray(sizes.medium)) {
    specs.push({
      name: `add [${sizes.medium.join('x')}] + scalar`,
      category: 'arithmetic',
      operation: 'add',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
        b: { shape: [1], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `add [${sizes.medium.join('x')}] + [${sizes.medium.join('x')}]`,
      category: 'arithmetic',
      operation: 'add',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
        b: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `multiply [${sizes.medium.join('x')}] * scalar`,
      category: 'arithmetic',
      operation: 'multiply',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
        b: { shape: [1], value: 2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `multiply [${sizes.medium.join('x')}] * [${sizes.medium.join('x')}]`,
      category: 'arithmetic',
      operation: 'multiply',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        b: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `mod [${sizes.medium.join('x')}] % scalar`,
      category: 'arithmetic',
      operation: 'mod',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' }, // arange data
        b: { shape: [1], value: 7 }, // Scalar 7
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `floor_divide [${sizes.medium.join('x')}] // scalar`,
      category: 'arithmetic',
      operation: 'floor_divide',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' }, // arange data
        b: { shape: [1], value: 3 }, // Scalar 3 (avoid zeros)
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `reciprocal [${sizes.medium.join('x')}]`,
      category: 'arithmetic',
      operation: 'reciprocal',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' }, // Use ones to avoid 1/0
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // Mathematical Operations Benchmarks
  // ========================================

  if (Array.isArray(sizes.medium)) {
    specs.push({
      name: `sqrt [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'sqrt',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: 1 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `power [${sizes.medium.join('x')}] ** 2`,
      category: 'math',
      operation: 'power',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: 1 },
        b: { shape: [1], value: 2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `absolute [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'absolute',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: -100 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `negative [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'negative',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `sign [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'sign',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: -100 },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // Linear Algebra Benchmarks
  // ========================================

  if (Array.isArray(sizes.medium)) {
    const [m, n] = sizes.medium;
    specs.push({
      name: `matmul [${m}x${n}] @ [${n}x${m}]`,
      category: 'linalg',
      operation: 'matmul',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
        b: { shape: [n!, m!], fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `transpose [${m}x${n}]`,
      category: 'linalg',
      operation: 'transpose',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });
  }

  // Larger matmul if not in quick mode
  if (mode !== 'quick' && Array.isArray(sizes.large)) {
    const [m, n] = sizes.large;
    specs.push({
      name: `matmul [${m}x${n}] @ [${n}x${m}]`,
      category: 'linalg',
      operation: 'matmul',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
        b: { shape: [n!, m!], fill: 'arange', dtype: 'float64' },
      },
      iterations: Math.floor(iterations / 2), // Fewer iterations for large
      warmup: Math.floor(warmup / 2),
    });
  }

  // ========================================
  // Reduction Benchmarks
  // ========================================

  if (Array.isArray(sizes.medium)) {
    specs.push({
      name: `sum [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'sum',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `sum [${sizes.medium.join('x')}] axis=0`,
      category: 'reductions',
      operation: 'sum',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        axis: { shape: [0] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `mean [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'mean',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `max [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'max',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `min [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'min',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `prod [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'prod',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `argmin [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'argmin',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `argmax [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'argmax',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `var [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'var',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `std [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'std',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `all [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'all',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `any [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'any',
      setup: {
        a: { shape: sizes.medium, fill: 'zeros' },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // Reshape Benchmarks
  // ========================================

  if (Array.isArray(sizes.medium)) {
    const [m, n] = sizes.medium;

    specs.push({
      name: `reshape [${m}x${n}] -> [${n}x${m}] (contiguous)`,
      category: 'reshape',
      operation: 'reshape',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        new_shape: { shape: [n!, m!] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `flatten [${m}x${n}]`,
      category: 'reshape',
      operation: 'flatten',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `ravel [${m}x${n}]`,
      category: 'reshape',
      operation: 'ravel',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // BigInt (64-bit) Benchmarks
  // ========================================
  // Tests representative operations with int64/uint64 dtypes
  // to compare BigInt performance vs standard numeric types

  if (Array.isArray(sizes.medium)) {
    // Creation with BigInt
    specs.push({
      name: `zeros [${sizes.medium.join('x')}] (int64)`,
      category: 'bigint',
      operation: 'zeros',
      setup: {
        shape: { shape: sizes.medium, dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    // Arithmetic with BigInt
    specs.push({
      name: `add [${sizes.medium.join('x')}] + [${sizes.medium.join('x')}] (int64)`,
      category: 'bigint',
      operation: 'add',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int64' },
        b: { shape: sizes.medium, fill: 'ones', dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `multiply [${sizes.medium.join('x')}] * scalar (int64)`,
      category: 'bigint',
      operation: 'multiply',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int64' },
        b: { shape: [1], value: 2, dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    // Math operations with BigInt
    specs.push({
      name: `absolute [${sizes.medium.join('x')}] (int64)`,
      category: 'bigint',
      operation: 'absolute',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: -100, dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    // Reductions with BigInt
    specs.push({
      name: `sum [${sizes.medium.join('x')}] (int64)`,
      category: 'bigint',
      operation: 'sum',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `max [${sizes.medium.join('x')}] (int64)`,
      category: 'bigint',
      operation: 'max',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    // Unsigned BigInt test
    specs.push({
      name: `add [${sizes.medium.join('x')}] + [${sizes.medium.join('x')}] (uint64)`,
      category: 'bigint',
      operation: 'add',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'uint64' },
        b: { shape: sizes.medium, fill: 'ones', dtype: 'uint64' },
      },
      iterations,
      warmup,
    });
  }

  return specs;
}

export function filterByCategory(specs: BenchmarkCase[], category: string): BenchmarkCase[] {
  return specs.filter((spec) => spec.category === category);
}

export function getCategories(specs: BenchmarkCase[]): string[] {
  const categories = new Set(specs.map((spec) => spec.category));
  return Array.from(categories).sort();
}
