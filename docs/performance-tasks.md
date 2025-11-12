# Performance Optimization Tasks

**Status**: Arithmetic optimizations completed (2025-11-12)
**Priority**: High impact but not blocking release

---

## Overview

These optimizations provide 8-10x speedup for common operations by adding fast paths for contiguous arrays. Arithmetic operations now use direct TypedArray access instead of slow generic code.

---

## 🚀 Optimization Tasks

### **Task P1**: ✅ COMPLETED - Add fast path for contiguous arrays in arithmetic operations

**Location**: `src/ops/arithmetic.ts` (add, subtract, multiply)

**Implementation** (2025-11-12):
- Added `canUseFastPath()` helper to detect contiguous arrays with same shape
- Each operation dispatches: scalar → fast array → slow broadcasting
- Fast paths use direct TypedArray access (no `iget()`, no function pointers)
- Clean architecture: separate helper functions per operation

**Code structure**:
```typescript
export function add(a, b) {
  if (typeof b === 'number') return addScalar(a, b);      // Already fast
  if (canUseFastPath(a, b)) return addArraysFast(a, b);   // NEW fast path
  return elementwiseBinaryOp(a, b, ...);                  // Broadcasting
}

function addArraysFast(a, b) {
  // Direct TypedArray access - fully optimizable by V8
  for (let i = 0; i < size; i++) {
    resultData[i] = aData[i] + bData[i];
  }
}
```

**Measured performance**:
```
add [100x100] + [100x100]:      93.44μs → 11.98μs (7.8x faster) ✨
multiply [100x100] * [100x100]: 114.79μs → 12.43μs (9.2x faster) ✨
```

**Key insight**: The bottleneck was `iget()` overhead (function call + stride calculation per element), not function pointers or loop branches. V8's JIT handles simple branches efficiently.

**Remaining operations** (can add fast paths when needed):
- divide, mod, floor_divide, power, reciprocal

---

### **Task P2**: ⏸️ TESTED BUT DEFERRED - Add fast path for contiguous arrays in reductions

**Location**: `src/ops/reduction.ts` (all reduction functions)

**Current Issue**:
- Uses `outerIndexToMultiIndex()` and `multiIndexToLinear()`
- Slow for large arrays with axis parameter
- No special handling for contiguous arrays

**Status**: Implemented and tested (2025-11-12), then reverted

**Why deferred**:
- Fast path only works for **last axis** (`axis=-1` or `axis=ndim-1`)
- Current benchmarks use `axis=0` (first axis), so no improvement visible
- Adds 180 LOC of complexity for an unmeasured case
- Decision: Implement after stdlib refactor when we have cleaner base

**Tested performance** (axis=-1, last axis):
```
sum [100x100] axis=-1:  10.3ms (fast path)
sum [100x100] axis=0:   237.4ms (slow path)
→ 23x faster for last-axis reductions!
```

**Implementation approach** (for future):
```typescript
export function sum(storage, axis) {
  // Fast path: C-contiguous array reducing along last axis
  if (storage.isCContiguous && normalizedAxis === ndim - 1) {
    for (let outer = 0; outer < outerSize; outer++) {
      let sum = 0;
      const offset = outer * axisSize;
      for (let inner = 0; inner < axisSize; inner++) {
        sum += data[offset + inner];  // Linear access!
      }
      resultData[outer] = sum;
    }
  }
  // Slow path: multi-index calculations
}
```

**When to implement**:
- After stdlib removal (simpler codebase)
- When we can optimize for both axis=0 and axis=-1
- Add benchmarks for both axes to measure impact

**Estimated speedup**: 20-50x for last-axis reductions
**Estimated time**: 2-3 hours (when revisited)

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

- **Completed**: Arithmetic fast paths (add, subtract, multiply) - 8x faster
- **Already done**: Added `isCContiguous` and `isFContiguous` checks to `ArrayStorage`
- **Already optimized**: `reshape()` and `ravel()` use fast paths for contiguous arrays
- **Deferred**: Reduction optimizations (tested but not measured by current benchmarks)
- **Safe pattern**: Slow path always available as fallback for edge cases

---

## 🎯 Success Criteria

- [x] Arithmetic operations 8x+ faster for contiguous arrays
- [ ] Reductions 20x+ faster for contiguous arrays (deferred)
- [x] All existing tests still pass (1183/1183)
- [x] No performance regression for non-contiguous arrays
- [x] Benchmarks demonstrate improvements

---

## 🔍 Bundle Size Analysis (2025-11-12)

**Discovery**: Tree-shaking is ineffective for @stdlib packages

Current bundle size: **583KB** (minified)
- 160KB: complex64/complex128 arrays (unused, can't tree-shake!)
- 200KB: stdlib utilities and dtype detection
- 48.9KB: buffer polyfill
- 15KB: dgemm (BLAS - actually needed)
- ~100KB: Your actual code

**Recommendation**: Remove stdlib ndarray wrapper, keep only @stdlib/blas
**Estimated size after removal**: ~250KB (57% reduction!)

**Action**: Tracked as separate task - remove stdlib after arithmetic optimizations merge

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
