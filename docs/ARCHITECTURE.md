# Architecture

## Core Philosophy

**"Correctness first, then optimize"**

1. Build correct NumPy-compatible API
2. Use @stdlib for proven computations
3. Validate everything against Python NumPy
4. Optimize bottlenecks later (WASM, SIMD, etc.)

---

## High-Level Design

```
┌─────────────────────────────────────────┐
│    NumPy-Compatible API (Thin Wrapper)  │
│  np.zeros(), arr.matmul(), np.linalg.*  │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│    @stdlib/ndarray (Array Foundation)   │
│  Memory, broadcasting, slicing, views   │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│    @stdlib/blas/base/* (Computation)    │
│  dgemm, dgemv, ddot, daxpy + WASM      │
└─────────────────────────────────────────┘
```

**We build**: Thin NumPy-compatible API layer only
**We use**: @stdlib/ndarray for array structure, @stdlib/blas for all computations

---

## Core Classes

### NDArray

```typescript
class NDArray {
  // Core properties
  private data: TypedArray;           // Float64Array, Int32Array, etc.
  readonly shape: readonly number[];  // [rows, cols, ...]
  readonly strides: readonly number[]; // Byte strides for each dimension
  readonly dtype: DType;              // Data type
  readonly offset: number;            // Offset into data buffer

  // Views share data
  readonly flags: {
    owndata: boolean;      // Owns the underlying buffer?
    writeable: boolean;    // Can modify data?
    c_contiguous: boolean; // C-order contiguous?
  };

  // Properties
  get ndim(): number;     // Number of dimensions
  get size(): number;     // Total elements
  get nbytes(): number;   // Total bytes
  get T(): NDArray;       // Transpose view

  // Creation
  static zeros(shape: number[], dtype?: DType): NDArray;
  static ones(shape: number[], dtype?: DType): NDArray;
  static array(data: NestedArray, dtype?: DType): NDArray;

  // Slicing (string-based)
  slice(...indices: SliceIndex[]): NDArray;
  row(i: number): NDArray;           // Convenience: arr[i, :]
  col(j: number): NDArray;           // Convenience: arr[:, j]

  // Element access
  get(indices: number[]): number | NDArray;
  set(indices: number[], value: number): void;

  // Operations (delegate to @stdlib where possible)
  matmul(other: NDArray): NDArray;   // Matrix multiply using dgemm
  dot(other: NDArray): number | NDArray;  // Dot product
  add(other: NDArray | number): NDArray;
  multiply(other: NDArray | number): NDArray;

  // Reductions
  sum(axis?: number): NDArray | number;
  mean(axis?: number): NDArray | number;
  max(axis?: number): NDArray | number;

  // Utilities
  copy(): NDArray;
  reshape(shape: number[]): NDArray;
  transpose(axes?: number[]): NDArray;
}
```

### DType System

Support all NumPy types:

**Numeric:**
- `int8`, `int16`, `int32`, `int64` (BigInt)
- `uint8`, `uint16`, `uint32`, `uint64` (BigInt)
- `float32`, `float64`
- `complex64`, `complex128` (interleaved: [real, imag, real, imag, ...])

**Others:**
- `bool` (stored as uint8)
- `datetime64`, `timedelta64`
- Structured dtypes
- `object` (JavaScript objects, no pickle)

```typescript
interface DType {
  name: string;           // 'float64', 'int32', etc.
  itemsize: number;       // Bytes per element
  kind: string;           // 'f', 'i', 'u', 'c', 'b', 'M', 'm', 'O'
  arrayType: TypedArrayConstructor;
}
```

---

## Memory Layout

### Strides and Contiguity

```
Array: [[1, 2, 3],
        [4, 5, 6]]

Shape: [2, 3]
C-order strides: [3*8, 8] = [24, 8] bytes
Data: [1, 2, 3, 4, 5, 6] (row-major)

Element [i, j] at offset: i * stride[0] + j * stride[1]
```

### Views vs Copies

**Views** (share data):
- `transpose()` - just swap strides
- `slice()` - adjust offset and strides
- `reshape()` - if possible without copy

**Copies** (new data):
- `copy()` - explicit
- `astype()` - dtype conversion
- Non-contiguous reshape

---

## Slicing

### String-Based Syntax

```typescript
// NumPy: arr[0:5, :, ::2]
// Ours:  arr.slice('0:5', ':', '::2')

arr.slice('0:5')        // First 5 elements
arr.slice(':', '2')     // All rows, column 2
arr.slice('1:-1')       // Exclude first and last
arr.slice('::2')        // Every other element
arr.slice('-1')         // Last element

// Convenience helpers
arr.row(0)              // arr[0, :]
arr.col(2)              // arr[:, 2]
arr.rows(0, 5)          // arr[0:5, :]
arr.cols(1, 3)          // arr[:, 1:3]
```

**Parser:**
```typescript
function parseSlice(spec: string): SliceSpec {
  // '0:5'    → {start: 0, stop: 5, step: 1}
  // ':'      → {start: null, stop: null, step: 1}
  // '::2'    → {start: null, stop: null, step: 2}
  // '-1'     → {start: -1, stop: null, step: 1}
}
```

---

## Broadcasting

Follow NumPy rules exactly:

```typescript
// Shapes aligned from right
(3, 4) + (4,)     → (3, 4)  ✓
(3, 1) + (1, 4)   → (3, 4)  ✓
(3, 4) + (3,)     → Error   ✗

function broadcastable(shape1: number[], shape2: number[]): boolean {
  const maxLen = Math.max(shape1.length, shape2.length);
  for (let i = 0; i < maxLen; i++) {
    const dim1 = shape1[shape1.length - 1 - i] ?? 1;
    const dim2 = shape2[shape2.length - 1 - i] ?? 1;
    if (dim1 !== dim2 && dim1 !== 1 && dim2 !== 1) {
      return false;
    }
  }
  return true;
}
```

---

## Using @stdlib

### How We Use @stdlib

**@stdlib/ndarray** - Array foundation:
- ✅ Memory management (TypedArrays, strides, views)
- ✅ Broadcasting logic built-in
- ✅ Slicing support
- ✅ DType system

**@stdlib/blas/base/** - Computational kernels:
- ✅ Level 1 BLAS: `ddot`, `daxpy`, `dcopy`, `dscal`, `dnrm2`
- ✅ Level 2 BLAS: `dgemv`, `dsymv`, `dtrmv`, `dtrsv`
- ✅ Level 3 BLAS: `dgemm`, `dger`
- ✅ WASM versions available: `dgemm-wasm`, `ddot-wasm`, etc.

**We implement:**
- Thin NumPy-compatible API wrapper
- String-based slicing syntax (`arr.slice('0:5', ':')`)
- High-level functions that compose @stdlib operations

### Example: Matrix Multiply

```typescript
import dgemm from '@stdlib/blas/base/dgemm';

class NDArray {
  matmul(other: NDArray): NDArray {
    // Validate
    if (this.ndim !== 2 || other.ndim !== 2) {
      throw new Error('matmul requires 2D arrays');
    }
    if (this.shape[1] !== other.shape[0]) {
      throw new Error(`Shape mismatch: ${this.shape} vs ${other.shape}`);
    }

    // Allocate result
    const result = NDArray.zeros([this.shape[0], other.shape[1]]);

    // Delegate to @stdlib's dgemm
    // C = alpha * A * B + beta * C
    dgemm(
      'row-major',      // Layout
      'no-transpose',   // TransA
      'no-transpose',   // TransB
      this.shape[0],    // M (rows of A)
      other.shape[1],   // N (cols of B)
      this.shape[1],    // K (cols of A, rows of B)
      1.0,              // alpha
      this.data,        // A
      this.shape[1],    // lda (leading dimension)
      other.data,       // B
      other.shape[1],   // ldb
      0.0,              // beta
      result.data,      // C
      result.shape[1]   // ldc
    );

    return result;
  }
}
```

### Example: Solve Linear System

```typescript
import dgesv from '@stdlib/lapack/base/dgesv';

function solve(A: NDArray, b: NDArray): NDArray {
  // Validate shapes
  if (A.shape[0] !== A.shape[1]) {
    throw new Error('A must be square');
  }
  if (b.shape[0] !== A.shape[0]) {
    throw new Error('Incompatible dimensions');
  }

  // dgesv modifies input, so copy
  const ACopy = A.copy();
  const x = b.copy();

  const n = A.shape[0];
  const ipiv = new Int32Array(n);

  const info = dgesv(
    'row-major',
    n,              // Matrix size
    1,              // Number of RHS (1 for vector)
    ACopy.data,
    n,              // lda
    ipiv,           // Pivot indices (output)
    x.data,
    1               // ldb
  );

  if (info < 0) {
    throw new Error(`Invalid argument at position ${-info}`);
  }
  if (info > 0) {
    throw new Error('Matrix is singular');
  }

  return x;
}
```

---

## Testing Strategy

### 1. Unit Tests (Vitest)

Test individual functions in isolation:

```typescript
describe('NDArray.zeros', () => {
  it('creates 1D array', () => {
    const arr = NDArray.zeros([5]);
    expect(arr.shape).toEqual([5]);
    expect(arr.data).toEqual(new Float64Array(5));
  });

  it('creates 2D array', () => {
    const arr = NDArray.zeros([2, 3]);
    expect(arr.shape).toEqual([2, 3]);
    expect(arr.size).toBe(6);
  });
});
```

### 2. Python Comparison Tests

Validate against Python NumPy:

```typescript
import { execSync } from 'child_process';

function validateAgainstNumPy(code: string): any {
  // Run Python code, return JSON output
  const result = execSync(`python3 -c "
import numpy as np
import json
${code}
print(json.dumps(result.tolist()))
  "`).toString();
  return JSON.parse(result);
}

it('matmul matches NumPy', () => {
  const A = NDArray.array([[1, 2], [3, 4]]);
  const B = NDArray.array([[5, 6], [7, 8]]);
  const result = A.matmul(B);

  const expected = validateAgainstNumPy(`
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = A @ B
  `);

  expect(result.toArray()).toEqual(expected);
});
```

---

## Module Structure

```
src/
├── core/
│   ├── ndarray.ts          # Main NDArray class
│   ├── dtype.ts            # DType system
│   ├── creation.ts         # zeros, ones, array, arange, linspace
│   ├── indexing.ts         # Slicing implementation
│   ├── broadcasting.ts     # Broadcasting rules
│   └── strides.ts          # Stride calculations
│
├── lib/
│   ├── arithmetic.ts       # add, subtract, multiply, divide
│   ├── comparison.ts       # greater, less, equal, etc.
│   ├── math.ts             # sin, cos, exp, log, sqrt, etc.
│   ├── reductions.ts       # sum, mean, std, min, max
│   └── manipulation.ts     # reshape, transpose, concatenate
│
├── linalg/
│   ├── basic.ts            # dot, matmul, norm
│   ├── solve.ts            # solve, inv, det
│   └── decomposition.ts    # svd, eig, qr, cholesky
│
├── random/
│   └── index.ts            # random, randn, randint
│
├── backends/
│   └── stdlib.ts           # @stdlib wrappers and utilities
│
└── index.ts                # Public API exports
```

---

## Key Design Decisions

### 1. BigInt for int64/uint64
- **Decision**: Use BigInt for exact representation
- **Tradeoff**: Different type, slightly slower
- **Alternative**: Float64 with precision loss

### 2. Complex Number Storage
- **Decision**: Interleaved storage `[real, imag, real, imag, ...]`
- **Input format**: Nested arrays `[[real, imag], ...]`
- **Rationale**: Matches C layout, efficient memory

### 3. Slicing Syntax
- **Decision**: String-based `arr.slice('0:5', ':')`
- **Rationale**: No build step, type-safe, familiar to Python devs
- **Helpers**: `row()`, `col()`, `rows()`, `cols()`

### 4. @stdlib Integration
- **Decision**: Use @stdlib/ndarray as foundation + @stdlib/blas for computations
- **Rationale**: Battle-tested implementation, we focus purely on NumPy API compatibility

### 5. Performance Strategy
- **Phase 1**: Correctness with @stdlib (pure JS)
- **Phase 2**: Profile and identify bottlenecks
- **Phase 3**: Add WASM for proven bottlenecks only

---

## Future Optimizations

**v1.0**: Pure TypeScript + @stdlib
- Acceptable: 10-100x slower than NumPy
- Focus: Correctness and API completeness

**v2.0**: Selective WASM
- Target: Large matrix operations, FFT
- Keep JS fallback for compatibility

**v3.0**: Advanced
- SIMD for element-wise ops
- WebGPU for specific operations
- Lazy evaluation

---

**Last Updated**: 2025-10-07
