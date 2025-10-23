/**
 * Benchmark specifications
 * Defines all benchmarks to run
 */

import type { BenchmarkCase, BenchmarkMode } from './types';

export function getBenchmarkSpecs(mode: BenchmarkMode = 'standard'): BenchmarkCase[] {
  // Determine sizes and iterations based on mode
  const config = {
    quick: {
      sizes: { small: 100, medium: [50, 50] },
      iterations: 10,
      warmup: 3
    },
    standard: {
      sizes: { small: 1000, medium: [100, 100], large: [500, 500] },
      iterations: 50,
      warmup: 10
    },
    full: {
      sizes: { small: 10000, medium: [1000, 1000], large: [2000, 2000] },
      iterations: 100,
      warmup: 20
    }
  }[mode];

  const { sizes, iterations, warmup } = config;
  const specs: BenchmarkCase[] = [];

  // ========================================
  // Array Creation Benchmarks
  // ========================================

  specs.push({
    name: `zeros [${sizes.small}]`,
    category: 'creation',
    operation: 'zeros',
    setup: {
      shape: { shape: [sizes.small] }
    },
    iterations,
    warmup
  });

  if (Array.isArray(sizes.medium)) {
    specs.push({
      name: `zeros [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'zeros',
      setup: {
        shape: { shape: sizes.medium }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `ones [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'ones',
      setup: {
        shape: { shape: sizes.medium }
      },
      iterations,
      warmup
    });
  }

  specs.push({
    name: `arange(${sizes.small})`,
    category: 'creation',
    operation: 'arange',
    setup: {
      n: { shape: [sizes.small] }
    },
    iterations,
    warmup
  });

  specs.push({
    name: `linspace(0, 100, ${sizes.small})`,
    category: 'creation',
    operation: 'linspace',
    setup: {
      n: { shape: [sizes.small] }
    },
    iterations,
    warmup
  });

  if (Array.isArray(sizes.medium)) {
    const eyeSize = sizes.medium[0]!;
    specs.push({
      name: `eye(${eyeSize})`,
      category: 'creation',
      operation: 'eye',
      setup: {
        n: { shape: [eyeSize] }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `identity(${eyeSize})`,
      category: 'creation',
      operation: 'identity',
      setup: {
        n: { shape: [eyeSize] }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `empty [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'empty',
      setup: {
        shape: { shape: sizes.medium }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `full [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'full',
      setup: {
        shape: { shape: sizes.medium },
        fill_value: { shape: [7] }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `copy [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'copy',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `zeros_like [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'zeros_like',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' }
      },
      iterations,
      warmup
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
        b: { shape: [1], fill: 'ones' }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `add [${sizes.medium.join('x')}] + [${sizes.medium.join('x')}]`,
      category: 'arithmetic',
      operation: 'add',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
        b: { shape: sizes.medium, fill: 'ones' }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `multiply [${sizes.medium.join('x')}] * scalar`,
      category: 'arithmetic',
      operation: 'multiply',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
        b: { shape: [1], value: 2 }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `multiply [${sizes.medium.join('x')}] * [${sizes.medium.join('x')}]`,
      category: 'arithmetic',
      operation: 'multiply',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        b: { shape: sizes.medium, fill: 'arange' }
      },
      iterations,
      warmup
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
        b: { shape: [n!, m!], fill: 'arange', dtype: 'float64' }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `transpose [${m}x${n}]`,
      category: 'linalg',
      operation: 'transpose',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' }
      },
      iterations,
      warmup
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
        b: { shape: [n!, m!], fill: 'arange', dtype: 'float64' }
      },
      iterations: Math.floor(iterations / 2), // Fewer iterations for large
      warmup: Math.floor(warmup / 2)
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
        a: { shape: sizes.medium, fill: 'arange' }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `sum [${sizes.medium.join('x')}] axis=0`,
      category: 'reductions',
      operation: 'sum',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        axis: { shape: [0] }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `mean [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'mean',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `max [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'max',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' }
      },
      iterations,
      warmup
    });
  }

  // ========================================
  // Reshape Benchmarks
  // ========================================

  if (Array.isArray(sizes.medium)) {
    const [m, n] = sizes.medium;
    const total = m! * n!;

    specs.push({
      name: `reshape [${m}x${n}] -> [${n}x${m}] (contiguous)`,
      category: 'reshape',
      operation: 'reshape',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        new_shape: { shape: [n!, m!] }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `flatten [${m}x${n}]`,
      category: 'reshape',
      operation: 'flatten',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' }
      },
      iterations,
      warmup
    });

    specs.push({
      name: `ravel [${m}x${n}]`,
      category: 'reshape',
      operation: 'ravel',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' }
      },
      iterations,
      warmup
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
