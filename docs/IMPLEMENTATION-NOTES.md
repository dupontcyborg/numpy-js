# Implementation Notes

Running log of implementation decisions, gotchas, and lessons learned as we build.

---

## Setup (2025-10-07)

### Project Structure
- Using esbuild for fast builds
- Vitest for testing (fast, TypeScript-native)
- TypeScript strict mode enabled
- Targeting ES2020+

### Dependencies
- Using @stdlib for BLAS/LAPACK operations
- fft.js for FFT (when we get there)
- Pure JS fallback for everything

---

## @stdlib Investigation Results (2025-10-10)

### ✅ DECISION: Use @stdlib/ndarray as Foundation

**Findings:**
- `@stdlib/ndarray` has everything we need: shape, strides, TypedArrays, views
- Broadcasting: `broadcastArray()`, `broadcastArrays()`
- Slicing: `slice()` function available
- DTypes: Full system with 20+ types
- **All BLAS functions available via `@stdlib/blas/base/*`**:
  - Level 1: `ddot`, `daxpy`, `dcopy`, `dscal`
  - Level 2: `dgemv`, `dsymv`, `dtrmv`
  - Level 3: `dgemm`, `dger` ✅ (Matrix multiply!)
  - WASM versions: `dgemm-wasm`, `ddot-wasm` for performance

**Strategy:**
```typescript
import ndarray from '@stdlib/ndarray';
import dgemm from '@stdlib/blas/base/dgemm';

// Use @stdlib/ndarray + thin wrapper for NumPy API
```

---

## Design Patterns

### Memory Management

```typescript
// Views share data, copies don't
const original = np.array([[1, 2], [3, 4]]);
const view = original.transpose();      // Shares data
const copy = original.copy();           // New data

view.set([0, 0], 99);
console.log(original.get([0, 0]));     // 99 (shared)
console.log(copy.get([0, 0]));         // 1 (independent)
```

### Error Handling

```typescript
// Always validate shapes before operations
function add(a: NDArray, b: NDArray): NDArray {
  if (!broadcastable(a.shape, b.shape)) {
    throw new ValueError(
      `operands could not be broadcast together with shapes ${a.shape} ${b.shape}`
    );
  }
  // ...
}
```

---

## Testing Patterns

### Unit Test Template

```typescript
import { describe, it, expect } from 'vitest';
import { NDArray } from '../src/core/ndarray';

describe('NDArray.function', () => {
  it('handles basic case', () => {
    const arr = NDArray.zeros([2, 3]);
    // test...
  });

  it('handles edge case', () => {
    // test...
  });

  it('throws on invalid input', () => {
    expect(() => {
      // invalid operation
    }).toThrow();
  });
});
```

### Python Comparison Template

```typescript
import { execSync } from 'child_process';

function runNumPy(code: string): any {
  const result = execSync(`python3 -c "
import numpy as np
import json
${code}
print(json.dumps(result.tolist() if hasattr(result, 'tolist') else result))
  "`).toString();
  return JSON.parse(result);
}

it('matches NumPy output', () => {
  const A = NDArray.array([[1, 2], [3, 4]]);
  const result = A.sum();

  const expected = runNumPy(`
A = np.array([[1, 2], [3, 4]])
result = A.sum()
  `);

  expect(result).toBe(expected);
});
```

---

## Common Gotchas

### 1. Stride Calculation
```typescript
// C-order (row-major): last dimension varies fastest
// Shape: [2, 3, 4]
// Strides: [3*4*itemsize, 4*itemsize, itemsize]

function computeStrides(shape: number[], itemsize: number): number[] {
  const strides = new Array(shape.length);
  let stride = itemsize;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}
```

### 2. Negative Indices
```typescript
// Python: arr[-1] is last element
function normalizeIndex(idx: number, size: number): number {
  if (idx < 0) {
    idx += size;
  }
  if (idx < 0 || idx >= size) {
    throw new IndexError(`index ${idx} out of bounds for size ${size}`);
  }
  return idx;
}
```

### 3. Broadcasting Edge Cases
```typescript
// (3,) can broadcast with (3, 1) or (1, 3)?
// Answer: Both!
// (3,) → (1, 3) can broadcast with (3, 1) → (3, 3)

// But (3,) cannot broadcast directly with (3, 1)
// Must align from the right
```

---

## Performance Notes

### When to Use @stdlib

**Use @stdlib** when:
- Matrix operations (BLAS: dgemm, dgemv, etc.)
- Linear algebra (LAPACK: dgesv, dgesvd, etc.)
- Proven numerical algorithms

**Implement ourselves** when:
- Broadcasting required
- Working with views (stride manipulation)
- Element-wise operations on arbitrary shapes
- Custom logic needed

### Example: Why Not @stdlib for Element-wise?

```typescript
// @stdlib functions expect flat arrays
// Our arrays may be views with complex strides

// BAD: Copy to flat array just to add
function add_bad(a: NDArray, b: NDArray): NDArray {
  const aFlat = a.ravel();  // Copy!
  const bFlat = b.ravel();  // Copy!
  // ... use @stdlib
  // ... reshape back
}

// GOOD: Direct iteration respecting strides
function add_good(a: NDArray, b: NDArray): NDArray {
  const [aB, bB] = broadcast(a, b);
  const result = NDArray.empty(aB.shape);
  // Element-wise operation with proper indexing
  for (let i = 0; i < result.size; i++) {
    result.data[i] = aB.getFlat(i) + bB.getFlat(i);
  }
  return result;
}
```

---

## Implementation Checklist

### Phase 1: Core Foundation
- [ ] NDArray class
  - [ ] Constructor with TypedArray + shape
  - [ ] Properties: shape, strides, dtype, ndim, size
  - [ ] get/set methods
  - [ ] copy() method
- [ ] DType system
  - [ ] Basic types: float64, float32, int32
  - [ ] TypedArray mapping
- [ ] Array creation
  - [ ] zeros, ones, empty
  - [ ] array() from nested arrays
  - [ ] arange, linspace
- [ ] Slicing
  - [ ] String parser: "0:5", ":", "::2"
  - [ ] Apply slicing to create views
  - [ ] Convenience: row(), col()
- [ ] Broadcasting
  - [ ] Check if shapes broadcastable
  - [ ] Compute output shape
  - [ ] Create broadcast views

### Phase 2: Basic Operations
- [ ] Arithmetic
  - [ ] add, subtract, multiply, divide
  - [ ] Scalar operations
  - [ ] Broadcasting
- [ ] Reductions
  - [ ] sum, mean, min, max
  - [ ] Support axis parameter
  - [ ] Support keepdims
- [ ] Comparison
  - [ ] greater, less, equal
  - [ ] Return boolean arrays
- [ ] Matrix operations
  - [ ] matmul using @stdlib dgemm
  - [ ] dot product using @stdlib ddot

### Phase 3: Linear Algebra
- [ ] solve using @stdlib dgesv
- [ ] inv, det
- [ ] svd using @stdlib dgesvd
- [ ] eig using @stdlib dgeev
- [ ] qr, cholesky

### Phase 4: Extended
- [ ] More math: sin, cos, exp, log
- [ ] Random number generation
- [ ] FFT using fft.js
- [ ] I/O: .npy/.npz files

---

## Known Issues

_(None yet - will document as we discover)_

---

## Questions Answered

1. **@stdlib ndarray**: ✅ ANSWERED - Use it!
   - Has everything: views, strides, broadcasting
   - Works perfectly with @stdlib/blas functions
   - Decision: Use @stdlib/ndarray, build thin NumPy wrapper on top

2. **Complex numbers**: Implement early or later?
   - Interleaved storage decided
   - When to actually implement?

3. **int64/uint64**: BigInt strategy solid?
   - Works but different type
   - Document clearly

4. **Testing**: How much Python comparison vs unit tests?
   - Start with unit tests
   - Add Python comparison for validation
   - Balance speed vs thoroughness

---

## Resources

- [@stdlib documentation](https://stdlib.io/)
- [NumPy documentation](https://numpy.org/doc/stable/)
- [BLAS/LAPACK reference](http://www.netlib.org/lapack/)

---

**Last Updated**: 2025-10-07
