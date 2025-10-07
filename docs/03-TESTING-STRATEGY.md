# Testing Strategy

## Overview

Our testing strategy focuses on three primary goals:
1. **Functional Correctness**: Validate that our implementation produces identical results to NumPy
2. **Comprehensive Coverage**: Test all public API functions
3. **Performance Baseline**: Establish performance characteristics (optimize later)

## Testing Pyramid

```
                    ┌─────────────────┐
                    │   Performance   │  Benchmarks against Python NumPy
                    │   Benchmarks    │
                    └─────────────────┘
                  ┌───────────────────────┐
                  │  Cross-Validation     │  TypeScript ↔ Python comparison
                  │  Integration Tests    │
                  └───────────────────────┘
              ┌─────────────────────────────────┐
              │         Unit Tests              │  Individual function tests
              └─────────────────────────────────┘
```

---

## 1. Unit Tests

### Purpose
Test individual functions and methods in isolation with known inputs and expected outputs.

### Framework
- **Test Runner**: Vitest (fast, TypeScript-native) or Jest
- **Assertion Library**: Built-in assertions + custom numerical comparisons
- **Coverage Tool**: c8 or Istanbul

### Structure
```
tests/unit/
├── core/
│   ├── ndarray.test.ts          # NDArray class tests
│   ├── dtype.test.ts            # DType system tests
│   ├── indexing.test.ts         # Indexing and slicing
│   └── broadcasting.test.ts     # Broadcasting logic
├── creation/
│   ├── zeros.test.ts
│   ├── ones.test.ts
│   ├── arange.test.ts
│   └── ...
├── manipulation/
│   ├── reshape.test.ts
│   ├── transpose.test.ts
│   └── ...
├── math/
│   ├── arithmetic.test.ts
│   ├── trigonometric.test.ts
│   ├── exponential.test.ts
│   └── ...
├── linalg/
│   ├── dot.test.ts
│   ├── matmul.test.ts
│   └── ...
└── ...
```

### Test Cases
Each function should have tests for:
- **Happy Path**: Standard expected usage
- **Edge Cases**: Empty arrays, single elements, zero dimensions
- **Boundary Conditions**: Maximum/minimum values, overflow/underflow
- **Type Variations**: All supported dtypes (int8, int16, int32, float32, float64, etc.)
- **Shape Variations**: 0-D, 1-D, 2-D, N-D arrays
- **Error Cases**: Invalid inputs, type mismatches, shape incompatibilities

### Example Test Structure
```typescript
describe('numpy.zeros', () => {
  describe('basic functionality', () => {
    it('should create 1-D array of zeros', () => {
      const arr = np.zeros(5);
      expect(arr.shape).toEqual([5]);
      expect(arr.tolist()).toEqual([0, 0, 0, 0, 0]);
    });

    it('should create 2-D array of zeros', () => {
      const arr = np.zeros([2, 3]);
      expect(arr.shape).toEqual([2, 3]);
      expect(arr.tolist()).toEqual([[0, 0, 0], [0, 0, 0]]);
    });
  });

  describe('dtype support', () => {
    it('should create int32 array', () => {
      const arr = np.zeros(5, { dtype: 'int32' });
      expect(arr.dtype.name).toBe('int32');
    });

    it('should create float64 array', () => {
      const arr = np.zeros(5, { dtype: 'float64' });
      expect(arr.dtype.name).toBe('float64');
    });
  });

  describe('edge cases', () => {
    it('should handle empty array', () => {
      const arr = np.zeros(0);
      expect(arr.shape).toEqual([0]);
      expect(arr.size).toBe(0);
    });

    it('should handle 0-D array', () => {
      const arr = np.zeros([]);
      expect(arr.shape).toEqual([]);
      expect(arr.ndim).toBe(0);
    });
  });
});
```

---

## 2. Cross-Validation Tests (Python ↔ TypeScript)

### Purpose
The **most critical testing layer**: Validate that NumPy.js produces identical results to Python NumPy for every function.

### Architecture

#### Test Oracle Pattern
Python NumPy serves as the "oracle" (source of truth). For each test:
1. Generate test inputs
2. Execute function in Python NumPy
3. Execute function in NumPy.js
4. Compare outputs

#### Implementation Approaches

**Option A: Python Subprocess (Recommended)**
```typescript
// tests/validation/harness.ts
import { spawn } from 'child_process';

async function validateAgainstNumPy(
  operation: string,
  inputs: any[],
  expectedShape?: number[]
): Promise<ValidationResult> {
  // 1. Serialize inputs to JSON
  const inputJson = JSON.stringify(inputs);

  // 2. Call Python script
  const python = spawn('python3', ['scripts/test-harness/oracle.py']);
  python.stdin.write(JSON.stringify({ operation, inputs }));
  python.stdin.end();

  // 3. Get Python NumPy result
  const pythonResult = await readPythonOutput(python);

  // 4. Execute in NumPy.js
  const jsResult = executeOperation(operation, inputs);

  // 5. Compare results
  return compareResults(pythonResult, jsResult);
}
```

```python
# scripts/test-harness/oracle.py
import numpy as np
import json
import sys

def execute_operation(operation, inputs):
    """Execute NumPy operation and return result."""
    # Parse operation (e.g., "np.zeros([2, 3])")
    result = eval(operation)

    return {
        'data': result.tolist() if isinstance(result, np.ndarray) else result,
        'dtype': str(result.dtype) if isinstance(result, np.ndarray) else None,
        'shape': result.shape if isinstance(result, np.ndarray) else None,
    }

if __name__ == '__main__':
    request = json.load(sys.stdin)
    result = execute_operation(request['operation'], request['inputs'])
    print(json.dumps(result))
```

**Option B: Python HTTP Server**
- More overhead but better for interactive development
- Can keep Python process running between tests
- Easier debugging

**Option C: WASI/PyScript (Future)**
- Run Python in WebAssembly
- Eliminates subprocess overhead
- More complex setup

### Test Structure
```
tests/validation/
├── harness/
│   ├── oracle.py              # Python test oracle
│   ├── client.ts              # TypeScript client for oracle
│   ├── comparisons.ts         # Numerical comparison utilities
│   └── serialization.ts       # Data serialization between Python/JS
├── suites/
│   ├── array-creation.test.ts
│   ├── array-manipulation.test.ts
│   ├── mathematical.test.ts
│   ├── linalg.test.ts
│   └── ...
└── fixtures/
    ├── arrays.json            # Pre-generated test arrays
    └── edge-cases.json        # Edge case datasets
```

### Comparison Strategies

#### Numerical Tolerance
Floating-point arithmetic is non-deterministic across implementations. Use tolerances:

```typescript
function assertArraysClose(
  actual: NDArray,
  expected: NDArray,
  rtol: number = 1e-7,  // Relative tolerance
  atol: number = 1e-9   // Absolute tolerance
): void {
  // Check: |actual - expected| <= atol + rtol * |expected|
  const diff = np.abs(np.subtract(actual, expected));
  const threshold = np.add(atol, np.multiply(rtol, np.abs(expected)));

  if (!np.all(np.less_equal(diff, threshold))) {
    throw new AssertionError(`Arrays not close enough`);
  }
}
```

#### Exact Integer Comparison
Integer operations should be exact:
```typescript
function assertArraysExact(actual: NDArray, expected: NDArray): void {
  if (!np.array_equal(actual, expected)) {
    throw new AssertionError(`Arrays not equal`);
  }
}
```

#### Shape and Dtype Comparison
Always verify:
```typescript
function assertArrayProperties(
  actual: NDArray,
  expected: { shape: number[]; dtype: string }
): void {
  expect(actual.shape).toEqual(expected.shape);
  expect(actual.dtype.name).toBe(expected.dtype);
}
```

### Test Generation

#### Property-Based Testing
Use property-based testing to generate diverse inputs:

```typescript
import { fc } from 'fast-check';

describe('np.add - property-based', () => {
  it('should be commutative', () => {
    fc.assert(
      fc.property(
        fc.array(fc.float(), { minLength: 1, maxLength: 100 }),
        fc.array(fc.float(), { minLength: 1, maxLength: 100 }),
        (a, b) => {
          const arr1 = np.array(a);
          const arr2 = np.array(b);

          // a + b should equal b + a
          const result1 = np.add(arr1, arr2);
          const result2 = np.add(arr2, arr1);

          return np.allclose(result1, result2);
        }
      )
    );
  });
});
```

#### Systematic Input Generation
Generate comprehensive test matrices:

```typescript
// Generate test cases for all dtype combinations
const dtypes = ['int8', 'int16', 'int32', 'float32', 'float64'];
const shapes = [[], [5], [3, 4], [2, 3, 4]];

for (const dtype of dtypes) {
  for (const shape of shapes) {
    it(`should work with dtype=${dtype} shape=${JSON.stringify(shape)}`, async () => {
      await validateAgainstNumPy(`np.zeros(${JSON.stringify(shape)}, dtype='${dtype}')`);
    });
  }
}
```

### Test Execution Strategy

#### Parallel Execution
Run validation tests in parallel for speed:
```typescript
describe.concurrent('array creation validation', () => {
  // Tests run in parallel
});
```

#### Test Batching
Batch multiple operations into single Python subprocess call:
```python
# Process multiple operations in one call
def batch_execute(operations):
    results = []
    for op in operations:
        results.append(execute_operation(op['operation'], op['inputs']))
    return results
```

---

## 3. Integration Tests

### Purpose
Test combinations of functions working together, mimicking real-world usage patterns.

### Structure
```
tests/integration/
├── workflows/
│   ├── data-processing.test.ts    # Load → transform → analyze
│   ├── linear-algebra.test.ts     # Matrix operations pipeline
│   ├── statistics.test.ts         # Statistical analysis workflow
│   └── ml-inference.test.ts       # Basic ML operations
├── io/
│   ├── npy-roundtrip.test.ts      # Save/load .npy files
│   └── npz-roundtrip.test.ts      # Save/load .npz files
└── compatibility/
    ├── broadcasting.test.ts       # Complex broadcasting scenarios
    └── dtype-promotion.test.ts    # Type promotion rules
```

### Example Integration Test
```typescript
describe('data processing workflow', () => {
  it('should load, normalize, and compute statistics', async () => {
    // 1. Create sample data
    const data = np.random.randn([100, 10]);

    // 2. Normalize (mean=0, std=1)
    const mean = np.mean(data, { axis: 0 });
    const std = np.std(data, { axis: 0 });
    const normalized = np.divide(np.subtract(data, mean), std);

    // 3. Compute correlation matrix
    const correlation = np.corrcoef(normalized, { rowvar: false });

    // 4. Validate against Python
    await validateWorkflow('normalize_and_correlate', { data });

    // 5. Check properties
    expect(correlation.shape).toEqual([10, 10]);
    expect(np.allclose(np.mean(normalized, { axis: 0 }), np.zeros(10))).toBe(true);
  });
});
```

---

## 4. Performance Benchmarks

### Purpose
Establish baseline performance characteristics and identify optimization opportunities.

### Framework
- **Benchmark.js** or **Tinybench**: For microbenchmarks
- **Custom harness**: For comparing against Python NumPy

### Structure
```
tests/benchmarks/
├── micro/
│   ├── array-creation.bench.ts
│   ├── element-wise.bench.ts
│   ├── reductions.bench.ts
│   └── linalg.bench.ts
├── macro/
│   ├── data-processing.bench.ts
│   └── ml-workflows.bench.ts
├── python-comparison/
│   ├── benchmark-runner.py    # Python benchmark harness
│   └── results/               # Benchmark results
└── reports/
    └── performance-report.md
```

### Benchmark Example
```typescript
import { bench, describe } from 'vitest';

describe('array creation benchmarks', () => {
  bench('np.zeros(1000)', () => {
    np.zeros(1000);
  });

  bench('np.ones([100, 100])', () => {
    np.ones([100, 100]);
  });

  bench('np.arange(10000)', () => {
    np.arange(10000);
  });
});
```

### Python Comparison Benchmark
```python
# tests/benchmarks/python-comparison/benchmark-runner.py
import numpy as np
import time
import json

def benchmark_operation(operation, iterations=1000):
    """Benchmark NumPy operation."""
    start = time.perf_counter()
    for _ in range(iterations):
        eval(operation)
    end = time.perf_counter()

    return {
        'operation': operation,
        'iterations': iterations,
        'total_time': end - start,
        'avg_time': (end - start) / iterations,
    }

# Run benchmarks
benchmarks = [
    'np.zeros(1000)',
    'np.ones([100, 100])',
    'np.arange(10000)',
    # ... more operations
]

results = [benchmark_operation(op) for op in benchmarks]
print(json.dumps(results, indent=2))
```

```typescript
// Compare results
import { spawn } from 'child_process';

async function compareBenchmarks() {
  // Run Python benchmarks
  const pythonResults = await runPythonBenchmarks();

  // Run JS benchmarks
  const jsResults = await runJSBenchmarks();

  // Generate comparison report
  generateReport(pythonResults, jsResults);
}
```

### Performance Metrics
Track:
- **Absolute Time**: Raw execution time
- **Relative Performance**: JS/Python ratio
- **Memory Usage**: Heap size before/after operations
- **Throughput**: Operations per second

### Performance Targets (Initial)
We're prioritizing correctness, so initial performance targets are modest:
- Simple operations (zeros, ones): 10-100x slower than Python acceptable
- Mathematical functions: 50-500x slower acceptable
- Linear algebra: 100-1000x slower acceptable initially

These will improve dramatically with:
1. TypedArrays optimization
2. WASM backend
3. SIMD operations
4. Lazy evaluation

---

## 5. Test Data Management

### Fixtures
Pre-generate test arrays for consistency:

```typescript
// tests/fixtures/generators.ts
export function generateTestArrays() {
  return {
    // Basic arrays
    empty: np.array([]),
    scalar: np.array(5),
    vector: np.array([1, 2, 3, 4, 5]),
    matrix: np.array([[1, 2], [3, 4]]),

    // Special values
    withNaN: np.array([1, NaN, 3]),
    withInf: np.array([1, Infinity, -Infinity]),

    // Different dtypes
    int8Array: np.array([1, 2, 3], { dtype: 'int8' }),
    float32Array: np.array([1.5, 2.5], { dtype: 'float32' }),

    // Large arrays
    large1D: np.arange(10000),
    large2D: np.random.rand([1000, 1000]),
  };
}
```

### Golden Files
Store expected outputs for complex operations:

```json
// tests/fixtures/golden/fft-outputs.json
{
  "fft_real_1d": {
    "input": [1, 2, 3, 4],
    "output": [10, -2+2i, -2, -2-2i],
    "shape": [4],
    "dtype": "complex128"
  }
}
```

---

## 6. Continuous Integration

### CI Pipeline
```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: npm run test:unit
      - run: npm run test:coverage

  validation-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install numpy>=2.0
      - run: npm install
      - run: npm run test:validation

  benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: npm run benchmark
      - run: npm run benchmark:report
```

### Test Commands
```json
{
  "scripts": {
    "test": "npm run test:unit && npm run test:validation",
    "test:unit": "vitest run tests/unit",
    "test:validation": "vitest run tests/validation",
    "test:integration": "vitest run tests/integration",
    "test:watch": "vitest watch",
    "test:coverage": "vitest run --coverage",
    "benchmark": "vitest bench",
    "benchmark:python": "python tests/benchmarks/python-comparison/benchmark-runner.py"
  }
}
```

---

## 7. Test Coverage Goals

### Coverage Targets
- **Unit Tests**: 90%+ line coverage
- **Validation Tests**: 100% API coverage (every public function validated against Python)
- **Integration Tests**: Cover common usage patterns

### Coverage Tracking
```typescript
// Track which APIs have validation tests
const API_COVERAGE = {
  'numpy.zeros': { validated: true, unit: true, integration: true },
  'numpy.ones': { validated: true, unit: true, integration: false },
  // ... all 800+ functions
};

// Generate coverage report
function generateAPICoverageReport() {
  const total = Object.keys(API_COVERAGE).length;
  const validated = Object.values(API_COVERAGE).filter(v => v.validated).length;

  console.log(`API Validation Coverage: ${validated}/${total} (${(validated/total*100).toFixed(1)}%)`);
}
```

---

## 8. Error Testing

### Error Cases to Test
Every function should have tests for:
- Invalid argument types
- Invalid shapes
- Out-of-bounds indices
- Division by zero
- Numerical overflow/underflow
- NaN propagation
- Memory allocation failures (large arrays)

### Example
```typescript
describe('error handling', () => {
  it('should throw on invalid shape', () => {
    expect(() => np.zeros([-1])).toThrow('Invalid shape');
  });

  it('should throw on invalid dtype', () => {
    expect(() => np.zeros(5, { dtype: 'invalid' })).toThrow('Unknown dtype');
  });

  it('should throw on shape mismatch', () => {
    const a = np.zeros([2, 3]);
    const b = np.zeros([3, 4]);
    expect(() => np.add(a, b)).toThrow('Shape mismatch');
  });
});
```

---

## 9. Documentation Testing

### Doctest-Style Examples
Include runnable examples in documentation:

```typescript
/**
 * Creates an array of zeros.
 *
 * @example
 * ```typescript
 * const arr = np.zeros(5);
 * console.log(arr.tolist()); // [0, 0, 0, 0, 0]
 * ```
 *
 * @example
 * ```typescript
 * const arr = np.zeros([2, 3]);
 * console.log(arr.shape); // [2, 3]
 * ```
 */
export function zeros(shape: number | number[], options?: ZerosOptions): NDArray {
  // implementation
}
```

Extract and run these examples as tests:
```typescript
// Extract code from JSDoc comments and execute
function extractAndTestDocExamples() {
  const examples = extractCodeFromDocs('src/**/*.ts');
  examples.forEach(example => {
    it(`should run doc example: ${example.title}`, () => {
      eval(example.code); // Run the example code
    });
  });
}
```

---

## 10. Test Utilities

### Custom Assertions
```typescript
// tests/utils/assertions.ts
export const npAssert = {
  arrayEqual(actual: NDArray, expected: NDArray) {
    expect(np.array_equal(actual, expected)).toBe(true);
  },

  arrayClose(actual: NDArray, expected: NDArray, rtol = 1e-7, atol = 1e-9) {
    expect(np.allclose(actual, expected, { rtol, atol })).toBe(true);
  },

  shapeEqual(actual: NDArray, expected: number[]) {
    expect(actual.shape).toEqual(expected);
  },

  dtypeEqual(actual: NDArray, expected: string) {
    expect(actual.dtype.name).toBe(expected);
  },
};
```

### Test Helpers
```typescript
// tests/utils/helpers.ts
export function randomShape(maxDim = 5, maxSize = 10): number[] {
  const ndim = Math.floor(Math.random() * maxDim) + 1;
  return Array.from({ length: ndim }, () => Math.floor(Math.random() * maxSize) + 1);
}

export function randomDtype(): string {
  const dtypes = ['int8', 'int16', 'int32', 'float32', 'float64'];
  return dtypes[Math.floor(Math.random() * dtypes.length)];
}
```

---

## Summary

Our testing strategy ensures:
1. ✅ **Correctness**: Every function validated against Python NumPy
2. ✅ **Comprehensive**: 800+ functions with multiple test cases each
3. ✅ **Automated**: Full CI/CD integration
4. ✅ **Performance-Aware**: Benchmark suite tracks performance over time
5. ✅ **Maintainable**: Clear structure and reusable utilities

This rigorous testing approach will give us confidence that NumPy.js is a faithful implementation of NumPy for the JavaScript ecosystem.

---

**Last Updated**: 2025-10-07
**Status**: Complete testing strategy documentation
