# Performance Optimization Tasks

**Status**: Deferred for future optimization pass
**Priority**: High impact but not blocking release

---

## Overview

These optimizations can provide 10-100x speedup for common operations by adding fast paths for contiguous arrays. Currently, all operations use slow generic code that works with any memory layout.

---

## 🚀 Optimization Tasks

### **Task P1**: Add fast path for contiguous arrays in elementwise operations

**Location**: `src/internal/compute.ts:31-106` (`elementwiseBinaryOp`)

**Current Issue**:
- Always uses `iget()` which is slow
- Works with any stride pattern
- No special handling for contiguous arrays (the common case)

**Optimization**:
```typescript
export function elementwiseBinaryOp(...) {
  // Fast path: both arrays C-contiguous with same shape (no broadcasting)
  if (a.isCContiguous && b.isCContiguous &&
      shapesEqual(a.shape, b.shape)) {
    // Direct TypedArray access - 10-100x faster
    const aData = a.data;
    const bData = b.data;
    const resultData = result.data;
    for (let i = 0; i < size; i++) {
      resultData[i] = op(aData[i], bData[i]);
    }
    return result;
  }

  // Slow path: use broadcasting and iget() (current implementation)
  // ...
}
```

**Estimated speedup**: 10-100x for contiguous arrays
**Estimated time**: 2 hours
**Test**: Add benchmarks comparing fast vs slow path

---

### **Task P2**: Add fast path for contiguous arrays in reductions

**Location**: `src/ops/reduction.ts` (all reduction functions)

**Current Issue**:
- Uses `outerIndexToMultiIndex()` and `multiIndexToLinear()`
- Very slow for large arrays
- No special handling for contiguous arrays

**Optimization for `sum()`**:
```typescript
export function sum(storage: ArrayStorage, axis?: number) {
  if (axis === undefined) {
    // Already fast - scalar reduction
    // ...
  }

  // Fast path: C-contiguous array
  if (storage.isCContiguous && normalizedAxis === storage.ndim - 1) {
    // Reduction along last axis - just stride through linearly
    const data = storage.data;
    const axisSize = shape[normalizedAxis];
    const outerSize = size / axisSize;

    for (let outer = 0; outer < outerSize; outer++) {
      let sum = 0;
      const offset = outer * axisSize;
      for (let inner = 0; inner < axisSize; inner++) {
        sum += data[offset + inner];
      }
      resultData[outer] = sum;
    }
    return result;
  }

  // Slow path: use multi-index calculations (current implementation)
  // ...
}
```

**Apply same pattern to**:
- `mean()` - calls sum, so benefits automatically
- `max()` - similar optimization
- `min()` - similar optimization

**Estimated speedup**: 5-50x for contiguous arrays
**Estimated time**: 2-3 hours
**Test**: Add benchmarks comparing performance

---

## 📊 Impact Analysis

### Before Optimization
```typescript
// 1M element array, simple addition
const a = ones([1000, 1000]);
const b = ones([1000, 1000]);
const c = a.add(b);  // ~100ms with iget()
```

### After Optimization
```typescript
// Same operation, fast path
const c = a.add(b);  // ~1-2ms with direct access
```

**Expected improvements**:
- Arithmetic operations: 10-100x faster
- Reductions: 5-50x faster
- Memory bandwidth limited (best case)

---

## 🔧 Implementation Strategy

### Phase 1: Measure
1. Add benchmarks for current performance
2. Profile to confirm iget() is the bottleneck
3. Set performance targets

### Phase 2: Implement
1. Add `isCContiguous` checks (already done!)
2. Add fast path for common cases
3. Keep slow path for edge cases
4. Ensure correctness with existing tests

### Phase 3: Validate
1. Verify performance improvements
2. Ensure no regressions in edge cases
3. Test with different dtypes
4. Benchmark vs NumPy (for context)

---

## 📝 Notes

- **Already done**: Added `isCContiguous` and `isFContiguous` checks to `ArrayStorage`
- **Already optimized**: `reshape()` and `ravel()` use fast paths for contiguous arrays
- **Safe to do**: These are pure performance optimizations - correctness unchanged
- **Low risk**: Slow path always available as fallback

---

## 🎯 Success Criteria

- [ ] Arithmetic operations 10x+ faster for contiguous arrays
- [ ] Reductions 5x+ faster for contiguous arrays
- [ ] All existing tests still pass
- [ ] No performance regression for non-contiguous arrays
- [ ] Benchmarks demonstrate improvements

---

## 🚀 Future Optimizations (Beyond these tasks)

**WASM Backend**:
- Replace JavaScript loops with WASM
- Use SIMD instructions
- Estimated: 2-4x additional speedup

**GPU Backend**:
- WebGPU for large arrays (>100k elements)
- Especially beneficial for matrix operations
- Estimated: 10-100x for large operations

**Parallelization**:
- Web Workers for multi-threaded operations
- Good for large arrays on multi-core systems
- Estimated: Near-linear scaling with cores

---

*Deferred on: October 20, 2025*
*Reason: Focus on correctness and completeness first*
