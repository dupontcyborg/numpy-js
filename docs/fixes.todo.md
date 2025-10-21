# NumPy.js Remaining Tasks

**Status**: Updated October 20, 2025
**Progress**: 22/33 original tasks complete (67%)
**Remaining**: 7 tasks

---

## 📊 Summary

✅ **COMPLETED**:
- Dtype system (7 tasks)
- Core usability - get(), set(), copy() (3 tasks)
- DType preservation - matmul(), flatten(), ravel() (3 tasks)
- Complex number support (removed - 5 tasks N/A)
- Performance optimizations for reshape/ravel (2 tasks partial)
- Dtype retention tests (3 tasks)
- Flags property (C_CONTIGUOUS, F_CONTIGUOUS) (1 task)

🚀 **DEFERRED**: Performance optimizations → See `docs/performance-tasks.md`

📋 **REMAINING**: 7 tasks (listed below)

---

## 🔧 Remaining Tasks

### Validation & Error Handling (3 tasks)

#### **Task #1**: Add division by zero handling

**Location**: `src/ops/arithmetic.ts` (divide functions)
**Priority**: Medium
**Estimated time**: 30 minutes

**NumPy behavior**:
- Float types: Returns `Infinity` or `-Infinity`
- Integer types: Raises error
- Special case: `0 / 0` returns `NaN` for floats

**Implementation**:
```typescript
function divideScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  if (scalar === 0) {
    if (dtype.startsWith('int') || dtype.startsWith('uint')) {
      throw new Error('integer division by zero');
    }
    // For float types, JavaScript naturally produces Inf/NaN
  }
  // ... rest of implementation
}
```

**Test**: Add division by zero tests
```typescript
expect(() => ones([2], 'int32').divide(0)).toThrow('division by zero');
expect(ones([2], 'float64').divide(0).get([0])).toBe(Infinity);
expect(zeros([2], 'float64').divide(0).get([0])).toBe(NaN);
```

---

#### **Task #2**: Add overflow detection for integer operations

**Location**: All arithmetic operations
**Priority**: Low (but good for correctness)
**Estimated time**: 2 hours

**Issue**: JavaScript integers silently overflow
```typescript
const arr = array([127], 'int8');
arr.add(1);  // Silently wraps to -128
```

**Options**:
1. **Wrap** (current behavior) - C-style, fast
2. **Saturate** - Clamp to min/max
3. **Raise error** - Throw on overflow
4. **Promote** - Auto-promote to larger type (NumPy 2.0 behavior)

**Recommendation**: Document current behavior (wrapping), add optional strict mode later

**Test**: Add overflow tests for all integer dtypes

---

#### **Task #3**: Add validation for mathematical operations

**Location**: Future math operations (sqrt, log, etc.)
**Priority**: Low (no math ops yet)
**Estimated time**: 1 hour when needed

**Examples**:
- `sqrt(-1)` for real dtypes → Error or NaN
- `log(0)` → Error or -Infinity
- `log(-1)` → Error or NaN

**Note**: We don't have these operations yet, so defer until implemented

---

### Design Improvements (1 task)


#### **Task #6**: Add comprehensive dtype promotion matrix tests

**Location**: Expand `tests/unit/dtype-promotion-matrix.test.ts` or add to existing tests
**Priority**: Medium
**Estimated time**: 1-2 hours

**Coverage**: Test all 11×11 dtype combinations (complex removed)

**Current**: ~34 promotion tests
**Target**: ~121 combinations (11 dtypes × 11 dtypes)

**Implementation**:
```typescript
describe('Complete dtype promotion matrix', () => {
  const allDTypes: DType[] = [
    'float64', 'float32',
    'int64', 'int32', 'int16', 'int8',
    'uint64', 'uint32', 'uint16', 'uint8',
    'bool'
  ];

  for (const dtype1 of allDTypes) {
    for (const dtype2 of allDTypes) {
      it(`promotes ${dtype1} + ${dtype2}`, () => {
        const a = ones([2], dtype1);
        const b = ones([2], dtype2);
        const result = a.add(b);
        // Verify result dtype follows NumPy rules
        expect(result.dtype).toBe(expectedPromotion[dtype1][dtype2]);
      });
    }
  }
});
```

**Note**: Can generate expected promotions from NumPy for validation

---

#### **Task #7**: Expand NumPy validation for all dtypes

**Location**: Expand existing `tests/validation/*.numpy.test.ts`
**Priority**: Low (nice to have)
**Estimated time**: 2-3 hours

**Current coverage**: Mostly float64, some int32/float32
**Target coverage**: All dtypes for all operations

**Specific additions**:
1. **int8/16/32** validation
   - Overflow behavior
   - Type promotion

2. **uint8/16/32** validation
   - Unsigned arithmetic
   - Mixed signed/unsigned

3. **float32** precision validation
   - Precision loss vs float64
   - Special values (NaN, Inf)

**Implementation**: Add dtype loops to existing test files