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
- [x] NDArray class
  - [x] Constructor with TypedArray + shape
  - [x] Properties: shape, strides, dtype, ndim, size
  - [ ] get/set methods
  - [ ] copy() method
- [ ] DType system
  - [x] Basic types: float64 (default)
  - [ ] More types: float32, int32, etc.
  - [ ] TypedArray mapping
- [x] Array creation
  - [x] zeros, ones, empty
  - [x] array() from nested arrays
  - [x] arange, linspace, eye
- [x] Slicing ✅ **COMPLETE (2025-10-12)**
  - [x] String parser: "0:5", ":", "::2"
  - [x] Apply slicing to create views
  - [x] Convenience: row(), col(), rows(), cols()
- [x] Broadcasting ✅ **COMPLETE (2025-10-12)**
  - [x] Check if shapes broadcastable
  - [x] Compute output shape
  - [x] Create broadcast views
  - [x] Integrated into all arithmetic operations

### Phase 2: Basic Operations
- [x] Arithmetic ✅ **WITH BROADCASTING**
  - [x] add, subtract, multiply, divide
  - [x] Scalar operations
  - [x] Broadcasting (fully integrated)
- [x] Reductions ✅ **WITH AXIS SUPPORT (2025-10-12)**
  - [x] sum, mean, min, max
  - [x] Support axis parameter
  - [x] Support keepdims
  - [x] Negative axis support
- [x] Matrix operations ✅ **COMPLETE**
  - [x] matmul using @stdlib dgemm
- [ ] Comparison
  - [ ] greater, less, equal
  - [ ] Return boolean arrays
- [ ] More reductions
  - [ ] prod, std, var, median
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
   - Implement benchmarks between Python & JS
   - Balance speed vs thoroughness

---

## Resources

- [@stdlib documentation](https://stdlib.io/)
- [NumPy documentation](https://numpy.org/doc/stable/)
- [BLAS/LAPACK reference](http://www.netlib.org/lapack/)

---

## Broadcasting Implementation (2025-10-12)

### ✅ Successfully Implemented

Broadcasting is now fully functional across all arithmetic operations!

**What Was Done:**
1. Created `src/core/broadcasting.ts` module with:
   - `computeBroadcastShape()` - Check compatibility and compute output shape
   - `areBroadcastable()` - Quick compatibility check
   - `broadcastStdlibArrays()` - Broadcast arrays using @stdlib
   - `broadcastErrorMessage()` - Generate descriptive error messages

2. Updated all arithmetic operations in `NDArray`:
   - `add()`, `subtract()`, `multiply()`, `divide()` now support broadcasting
   - Uses @stdlib's `broadcastArrays()` under the hood
   - Efficient: Creates read-only views, no data copying

3. Comprehensive testing:
   - **25 new unit tests** for broadcasting utilities and operations
   - **15 new Python validation tests** confirming NumPy compatibility
   - **All 169 tests pass** (100 unit + 69 validation)

**Implementation Details:**

The approach leverages @stdlib's battle-tested broadcasting:

```typescript
private _elementwiseOp(other: NDArray, op: (a, b) => number, opName: string): NDArray {
  // 1. Check compatibility and compute output shape
  const outputShape = computeBroadcastShape([this.shape, other.shape]);

  // 2. Broadcast both arrays (creates views, no copying)
  const [broadcastThis, broadcastOther] = broadcastStdlibArrays([this._data, other._data]);

  // 3. Create result array and perform element-wise operation
  const result = zeros(outputShape);
  for (let i = 0; i < size; i++) {
    resultData[i] = op(broadcastThis.iget(i), broadcastOther.iget(i));
  }

  return result;
}
```

**Examples Now Supported:**

```typescript
// (3, 4) + (4,) → (3, 4)
array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).add(array([10, 20, 30, 40]))

// (3, 1) * (1, 4) → (3, 4)
array([[2], [3], [4]]).multiply(array([[10, 20, 30, 40]]))

// (1,) + (3, 4) → (3, 4)
array([5]).add(array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))
```

**Performance Considerations:**

- @stdlib's broadcasting creates memory-efficient views (no data duplication)
- Element-wise iteration uses `iget()` for correct strided access
- Current implementation prioritizes correctness over speed
- Future optimization: Could use SIMD or @stdlib's element-wise ops for specific cases

**What's Next:**

Broadcasting is foundational - it unblocks many other operations. Natural next steps:
1. **Slicing** - Another core feature
2. **Axis support for reductions** - `sum(axis=0)`, `mean(axis=1)`, etc.
3. **More arithmetic ops** - `power()`, `mod()`, etc. (will automatically have broadcasting)
4. **Comparison operations** - `greater()`, `less()`, `equal()` with broadcasting

---

## Slicing Implementation (2025-10-12)

### ✅ Successfully Implemented

Array slicing is now fully functional with NumPy-compatible string syntax!

**What Was Done:**
1. Created `src/core/slicing.ts` module with:
   - `parseSlice()` - Parse slice strings like "0:5", ":", "::2", "-1"
   - `normalizeSlice()` - Handle negative indices and defaults
   - `computeSliceLength()` - Calculate slice result length
   - Full support for NumPy slice syntax

2. Implemented slicing in `NDArray` class:
   - `slice(...sliceStrs)` - Main slicing method accepting string specs per dimension
   - `row(i)`, `col(j)` - Convenience methods for single row/column
   - `rows(start, stop)`, `cols(start, stop)` - Range convenience methods
   - Integrates with @stdlib's `slice()` function for view semantics

3. Added 0-dimensional array support:
   - Updated `toArray()` to properly handle scalars (0-d arrays)
   - Single index slicing returns 0-d arrays matching NumPy behavior

4. Comprehensive testing:
   - **45 unit tests** for slice parser (parseSlice, normalizeSlice, etc.)
   - **52 unit tests** for actual slice operations on NDArray objects
   - **27 Python validation tests** confirming NumPy compatibility
   - **All 214 tests pass** (187 unit + 27 validation)

**Implementation Details:**

String-based slice syntax was chosen because TypeScript doesn't support Python's native slicing syntax:

```typescript
// NumPy/Python:
arr[0:5, ::2]

// NumPy.js equivalent:
arr.slice('0:5', '::2')
```

The implementation follows this pipeline:

```typescript
slice(...sliceStrs: string[]): NDArray {
  // 1. Parse each slice string
  const sliceSpecs = sliceStrs.map((str, i) => {
    const spec = parseSlice(str);           // Parse "0:5" → {start: 0, stop: 5, step: 1}
    return normalizeSlice(spec, this.shape[i]);  // Handle negatives, defaults
  });

  // 2. Convert to @stdlib Slice objects
  const stdlibSlices = sliceSpecs.map(spec => {
    if (spec.isIndex) return spec.start;    // Single index: 5
    return new Slice(start, stop, step);    // Range: Slice(0, 5, 1)
  });

  // 3. Use @stdlib's slice function (creates view, no copying)
  const result = stdlib_slice(this._data, ...stdlibSlices);
  return new NDArray(result);
}
```

**Supported Syntax:**

All NumPy slice syntax variants are supported:

```typescript
// Single index
arr.slice('2')           // arr[2]  → 0-d array (scalar)
arr.slice('-1')          // arr[-1] → last element

// Start:Stop
arr.slice('2:7')         // arr[2:7]   → indices 2,3,4,5,6
arr.slice(':5')          // arr[:5]    → first 5 elements
arr.slice('5:')          // arr[5:]    → from index 5 to end
arr.slice(':-3')         // arr[:-3]   → all but last 3

// Start:Stop:Step
arr.slice('::2')         // arr[::2]   → every 2nd element
arr.slice('1:8:2')       // arr[1:8:2] → indices 1,3,5,7
arr.slice('::-1')        // arr[::-1]  → reverse array
arr.slice('7:2:-1')      // arr[7:2:-1]→ indices 7,6,5,4,3

// Multi-dimensional
arr.slice('0:2', '1:3')  // arr[0:2, 1:3] → submatrix
arr.slice(':', '::2')    // arr[:, ::2]   → every 2nd column

// Convenience methods
arr.row(1)               // arr[1, :]     → single row
arr.col(2)               // arr[:, 2]     → single column
arr.rows(1, 3)           // arr[1:3, :]   → rows 1-2
arr.cols(1, 3)           // arr[:, 1:3]   → columns 1-2
```

**Examples:**

```typescript
// 1D slicing
const arr1d = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
arr1d.slice('2:7');      // [2, 3, 4, 5, 6]
arr1d.slice('-3:');      // [7, 8, 9]
arr1d.slice('::2');      // [0, 2, 4, 6, 8]
arr1d.slice('::-1');     // [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

// 2D slicing
const arr2d = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
arr2d.slice('0', ':');   // [1, 2, 3]  (first row)
arr2d.slice(':', '1');   // [2, 5, 8]  (second column)
arr2d.slice('0:2', '1:3'); // [[2, 3], [5, 6]]  (submatrix)

// Convenience methods
arr2d.row(1);            // [4, 5, 6]
arr2d.col(2);            // [3, 6, 9]
arr2d.rows(0, 2);        // [[1, 2, 3], [4, 5, 6]]
arr2d.cols(1, 3);        // [[2, 3], [5, 6], [8, 9]]
```

**View Semantics:**

Like NumPy, slices return views (not copies) when possible:

```typescript
const original = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
const view = original.slice('1:', '1:');  // [[5, 6], [8, 9]]

// Views share underlying data with @stdlib
// (Modifying views affects original - when we add set() method)
```

**Design Trade-offs:**

1. **String-based syntax**: Required due to TypeScript limitations
   - Pro: Clean, readable, familiar to NumPy users
   - Pro: Type-safe (strings are validated at runtime)
   - Con: No compile-time validation (but comprehensive runtime errors)

2. **@stdlib integration**: Leverages robust, tested implementation
   - Pro: Correct handling of all edge cases (negative indices, steps, etc.)
   - Pro: Efficient view creation (no data copying)
   - Pro: Consistent with broadcasting approach

3. **0-dimensional arrays**: Return scalars via `.toArray()`
   - Matches NumPy: `arr[0]` on 1-D array returns scalar
   - Uses `._data.get()` for 0-d array values

**Performance Considerations:**

- Slice parsing is fast (simple string operations)
- @stdlib's `slice()` creates views with adjusted strides (O(1), no data copy)
- Step-based slicing handled efficiently by @stdlib
- Negative indices normalized once during parsing

**What's Next:**

With slicing complete, these operations become more useful:
1. **Advanced indexing** - Boolean and integer array indexing
2. **Reshape operations** - `reshape()`, `flatten()`, `ravel()`
3. **Set operations** - `arr.slice('0:2').set(value)` for assignment
4. **More convenience methods** - `diagonal()`, `take()`, `compress()`

---

## Axis-Based Reductions Implementation (2025-10-12)

### ✅ Successfully Implemented

Reduction operations now support axis and keepdims parameters, matching NumPy behavior!

**What Was Done:**
1. Updated four reduction methods with axis support:
   - `sum(axis?, keepdims?)` - Sum along specified axis
   - `mean(axis?, keepdims?)` - Mean along specified axis
   - `max(axis?, keepdims?)` - Maximum along specified axis
   - `min(axis?, keepdims?)` - Minimum along specified axis

2. Implemented helper methods for multi-dimensional indexing:
   - `_outerIndexToMultiIndex()` - Convert output index to input multi-index
   - `_multiIndexToLinear()` - Convert multi-index to linear index
   - `_computeStrides()` - Compute strides for row-major order

3. Comprehensive testing:
   - **31 new unit tests** for all reduction operations with axis support
   - **29 new Python validation tests** confirming NumPy compatibility
   - **All 247 tests pass** (218 unit + 29 validation)

**Implementation Details:**

The axis parameter allows reducing along specific dimensions:

```typescript
sum(axis?: number, keepdims: boolean = false): NDArray | number {
  if (axis === undefined) {
    // Sum all elements → scalar
    return this.data.reduce((a, b) => a + b, 0);
  }

  // Normalize negative axis (-1 → ndim-1)
  if (axis < 0) {
    axis = this.ndim + axis;
  }

  // Compute output shape (remove axis dimension)
  const outputShape = Array.from(this.shape).filter((_, i) => i !== axis);

  // Perform reduction
  const result = zeros(outputShape);
  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let sum = 0;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = this._outerIndexToMultiIndex(outerIdx, axis, axisIdx);
      const linearIdx = this._multiIndexToLinear(inputIndices);
      sum += this.data[linearIdx];
    }
    resultData[outerIdx] = sum;
  }

  // Handle keepdims (preserve reduced dimension as size-1)
  if (keepdims) {
    const keepdimsShape = [...this.shape];
    keepdimsShape[axis] = 1;
    return new NDArray(stdlib_ndarray.ndarray(
      'float64', resultData, keepdimsShape,
      this._computeStrides(keepdimsShape), 0, 'row-major'
    ));
  }

  return result;
}
```

**Supported Syntax:**

All NumPy reduction syntax variants are supported:

```typescript
const arr = array([[1, 2, 3], [4, 5, 6]]);  // shape: (2, 3)

// No axis - reduce all elements → scalar
arr.sum()            // 21
arr.mean()           // 3.5
arr.max()            // 6
arr.min()            // 1

// axis=0 - reduce along rows → shape: (3,)
arr.sum(0)           // [5, 7, 9]
arr.mean(0)          // [2.5, 3.5, 4.5]
arr.max(0)           // [4, 5, 6]

// axis=1 - reduce along columns → shape: (2,)
arr.sum(1)           // [6, 15]
arr.mean(1)          // [2, 5]
arr.max(1)           // [3, 6]

// Negative axis - count from end
arr.sum(-1)          // Same as axis=1 for 2D
arr.sum(-2)          // Same as axis=0 for 2D

// keepdims=true - preserve reduced dimension
arr.sum(0, true)     // [[5, 7, 9]]  shape: (1, 3)
arr.sum(1, true)     // [[6], [15]]  shape: (2, 1)
```

**3D Array Examples:**

```typescript
const arr3d = array([
  [[1, 2], [3, 4]],
  [[5, 6], [7, 8]]
]);  // shape: (2, 2, 2)

// axis=0 - reduce along first dimension → shape: (2, 2)
arr3d.sum(0)         // [[6, 8], [10, 12]]

// axis=1 - reduce along second dimension → shape: (2, 2)
arr3d.sum(1)         // [[4, 6], [12, 14]]

// axis=2 - reduce along third dimension → shape: (2, 2)
arr3d.sum(2)         // [[3, 7], [11, 15]]

// axis=-1 - last axis → shape: (2, 2)
arr3d.mean(-1)       // [[1.5, 3.5], [5.5, 7.5]]
```

**Implementation Challenges & Solutions:**

1. **Negative Axis Handling in mean()**:
   - Bug: Using `this.shape[axis]` with negative axis (e.g., -1) returned `undefined`
   - Fix: Normalize axis at the start of the method before using it to access shape

2. **Multi-dimensional Index Conversion**:
   - Challenge: Converting between linear indices and multi-dimensional indices
   - Solution: Implemented helper methods to handle row-major stride calculations

3. **keepdims Support**:
   - Challenge: Preserving shape with size-1 dimensions
   - Solution: Used @stdlib's ndarray constructor to reshape result data

**Performance Considerations:**

- **Current implementation**: Iterates through all elements for axis reduction
  - Time complexity: O(size) where size = total number of elements
  - Simple and correct, prioritizing correctness over optimization

- **Future optimizations**:
  - Could use @stdlib's element-wise operations for specific cases
  - SIMD vectorization for large arrays
  - Multi-threading for very large reductions

**What's Next:**

With axis-based reductions complete, natural next steps include:
1. **More reduction operations** - `prod()`, `std()`, `var()`, `median()`
2. **Cumulative operations** - `cumsum()`, `cumprod()`
3. **Boolean reductions** - `all()`, `any()`
4. **Arg reductions** - `argmin()`, `argmax()` to find indices of min/max values

---

## Comparison Operations Implementation (2025-10-12)

### ✅ Successfully Implemented

Comparison operations now support broadcasting and return boolean arrays (as uint8)!

**What Was Done:**
1. Implemented six comparison methods:
   - `greater(other)` - Element-wise greater than (>)
   - `greater_equal(other)` - Element-wise greater than or equal (>=)
   - `less(other)` - Element-wise less than (<)
   - `less_equal(other)` - Element-wise less than or equal (<=)
   - `equal(other)` - Element-wise equality (==)
   - `not_equal(other)` - Element-wise inequality (!=)

2. All operations support:
   - Scalar comparisons
   - Array-to-array comparisons
   - Broadcasting for incompatible shapes

3. Comprehensive testing:
   - **26 new unit tests** for all comparison operations
   - **18 new Python validation tests** confirming NumPy compatibility
   - **All 262 tests pass** (244 unit + 18 validation)

**Boolean Array Representation:**

JavaScript lacks native boolean TypedArrays, so we use `uint8` (0=false, 1=true):

```typescript
greater(other: NDArray | number): NDArray {
  if (typeof other === 'number') {
    const data = new Uint8Array(this.size);
    for (let i = 0; i < this.size; i++) {
      data[i] = this.data[i]! > other ? 1 : 0;
    }
    return new NDArray(stdlib_ndarray.ndarray('uint8', data, ...));
  } else {
    return this._comparisonOp(other, (a, b) => a > b, 'greater');
  }
}
```

**Why uint8 instead of bool?**
1. ✅ TypedArrays don't have a boolean type
2. ✅ uint8 is memory-efficient (1 byte per element)
3. ✅ Standard practice in numerical libraries
4. ✅ Easy conversion for boolean operations (0 = false, 1 = true)
5. ✅ Compatible with NumPy via `.astype(np.uint8)`

**Supported Syntax:**

```typescript
const a = array([1, 2, 3, 4, 5]);
const b = array([2, 2, 2, 6, 1]);

// Scalar comparisons
a.greater(3);         // [0, 0, 0, 1, 1]  (uint8)
a.less_equal(3);      // [1, 1, 1, 0, 0]
a.equal(2);           // [0, 1, 0, 0, 0]

// Array comparisons
a.greater(b);         // [0, 0, 1, 0, 1]
a.equal(b);           // [0, 1, 0, 0, 0]

// With broadcasting
const matrix = array([[1, 2, 3], [4, 5, 6]]);
const row = array([2, 4, 5]);
matrix.greater(row);  // [[0, 0, 0], [1, 1, 1]]  shape: (2, 3)
```

**What's Next:**

With comparison operations complete, natural next steps include:
1. **Logical operations** - `logical_and()`, `logical_or()`, `logical_not()` for boolean arrays
2. **Boolean indexing** - Use comparison results to filter arrays: `arr[arr > 5]`
3. **Conditional functions** - `where(condition, x, y)` for conditional selection
4. **Reshape operations** - `reshape()`, `flatten()`, `ravel()`, `transpose()`

---

**Last Updated**: 2025-10-12
