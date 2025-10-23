# NumPy.js Remaining Tasks

**Status**: Updated October 22, 2025
**Progress**: 24/33 original tasks complete (73%)
**Remaining**: 5 tasks

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
- Division by zero handling with NumPy-compliant promotion (1 task)

🚀 **DEFERRED**: Performance optimizations → See `docs/performance-tasks.md`

📋 **REMAINING**: 5 tasks (listed below)

---

## 🔧 Remaining Tasks

### Validation & Error Handling (2 tasks)

✅ **COMPLETED**: Task #1 - Division by zero handling (October 22, 2025)
- **Implemented NumPy-compliant division behavior**:
  - All integer division now promotes to float64 (NumPy behavior)
  - Division by zero returns Infinity/NaN (not errors)
  - float32 / integer → float32, float64 / anything → float64
- **Created comprehensive test coverage**:
  - 42 unit tests in `tests/unit/division.test.ts`
  - 29 NumPy validation tests in `tests/validation/division.numpy.test.ts`
- **Updated existing tests** to match NumPy behavior (6 tests in bigint-arithmetic, dtype-retention, dtype-promotion-matrix)
- All 900 unit tests pass, all 29 NumPy validation tests pass

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

### Design Improvements (0 tasks)

✅ **COMPLETED**: Task #6 - Comprehensive dtype promotion matrix tests (October 20, 2025)
- Created `tests/unit/dtype-promotion-matrix.test.ts` with all 121 combinations (385 tests)
- Created `tests/validation/dtype-promotion-matrix.numpy.test.ts` with NumPy validation (165 tests)
- Fixed `promoteDTypes()` implementation to match NumPy behavior exactly
- Optimized validation tests with batched Python calls (9s → 3.4s, 2.6x faster)
- All tests pass: 385 unit tests + 165 NumPy validation tests

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