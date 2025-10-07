# Architecture Design Document

## Philosophy: Correctness First, Performance Later

Our architecture prioritizes:
1. **Functional Correctness**: Match NumPy's behavior exactly
2. **Clean Abstractions**: Maintainable, understandable code
3. **Testability**: Easy to validate against Python NumPy
4. **Extensibility**: Support for optimization backends later

Performance optimization comes AFTER we have a correct, validated implementation.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Public API Layer                          │
│  (numpy.js - matches NumPy 2.0+ Python API)                     │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────┐
│                     Core Array Layer                             │
│  • NDArray class                                                 │
│  • DType system                                                  │
│  • Broadcasting engine                                           │
│  • Indexing and slicing                                          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────┐
│                   Operation Layer                                │
│  • Universal functions (ufuncs)                                  │
│  • Reduction operations                                          │
│  • Linear algebra operations                                     │
│  • FFT operations                                                │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────┐
│                    Backend Layer                                 │
│  Phase 1: Pure TypeScript (TypedArrays)                         │
│  Phase 2: Optional WASM backend (future)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
src/
├── index.ts                    # Main entry point, exports public API
├── core/
│   ├── ndarray.ts             # NDArray class
│   ├── dtype.ts               # DType system
│   ├── broadcasting.ts        # Broadcasting rules
│   ├── indexing.ts            # Indexing and slicing
│   ├── strides.ts             # Stride calculation
│   └── memory.ts              # Memory layout utilities
├── dtypes/
│   ├── scalar-types.ts        # int8, float64, etc.
│   ├── complex.ts             # Complex number support
│   ├── structured.ts          # Structured dtypes
│   └── datetime.ts            # Datetime types
├── ufuncs/
│   ├── ufunc.ts               # Universal function base class
│   ├── arithmetic.ts          # add, multiply, etc.
│   ├── trigonometric.ts       # sin, cos, etc.
│   ├── comparison.ts          # greater, less, equal, etc.
│   ├── logical.ts             # and, or, not, etc.
│   └── bitwise.ts             # bitwise operations
├── lib/
│   ├── creation.ts            # zeros, ones, arange, etc.
│   ├── manipulation.ts        # reshape, transpose, etc.
│   ├── reduction.ts           # sum, mean, std, etc.
│   ├── statistics.ts          # median, percentile, etc.
│   ├── sorting.ts             # sort, argsort, etc.
│   └── set-operations.ts      # unique, union, etc.
├── linalg/
│   ├── index.ts               # Linear algebra module
│   ├── products.ts            # dot, matmul, etc.
│   ├── decomposition.ts       # qr, svd, etc.
│   ├── solve.ts               # solve, lstsq, inv, etc.
│   └── norms.ts               # norm, det, etc.
├── fft/
│   ├── index.ts               # FFT module
│   ├── standard.ts            # fft, ifft, etc.
│   └── real.ts                # rfft, irfft, etc.
├── random/
│   ├── generator.ts           # Generator class
│   ├── bit-generators.ts      # PCG64, MT19937, etc.
│   └── distributions.ts       # uniform, normal, etc.
├── polynomial/
│   ├── polynomial.ts          # Power series
│   ├── chebyshev.ts           # Chebyshev polynomials
│   └── legendre.ts            # Legendre polynomials
├── io/
│   ├── npy.ts                 # .npy format read/write
│   ├── npz.ts                 # .npz format read/write
│   └── text.ts                # loadtxt, savetxt
├── ma/
│   └── index.ts               # Masked array support
├── testing/
│   └── index.ts               # Testing utilities
└── types/
    ├── index.d.ts             # TypeScript type definitions
    └── numpy.d.ts             # Public API types
```

---

## Core Classes

### 1. NDArray Class

The fundamental n-dimensional array class.

```typescript
class NDArray<T extends DType = DType> {
  // Core properties
  readonly data: TypedArray;      // Underlying data buffer
  readonly dtype: T;               // Data type
  readonly shape: readonly number[]; // Array dimensions
  readonly strides: readonly number[]; // Strides for each dimension
  readonly offset: number;         // Offset into data buffer

  // Computed properties
  get ndim(): number;              // Number of dimensions
  get size(): number;              // Total number of elements
  get nbytes(): number;            // Total bytes
  get itemsize(): number;          // Bytes per element

  // Views and copies
  readonly T: NDArray<T>;          // Transposed view
  readonly real: NDArray<RealType<T>>; // Real part
  readonly imag: NDArray<RealType<T>>; // Imaginary part
  readonly flat: FlatIterator<T>;  // Flat iterator

  // Flags
  readonly flags: {
    c_contiguous: boolean;
    f_contiguous: boolean;
    owndata: boolean;
    writeable: boolean;
  };

  // Methods
  reshape(shape: number[]): NDArray<T>;
  transpose(axes?: number[]): NDArray<T>;
  astype<U extends DType>(dtype: U): NDArray<U>;
  copy(order?: 'C' | 'F'): NDArray<T>;
  tolist(): NestedArray<ScalarType<T>>;

  // Element access
  item(...indices: number[]): ScalarType<T>;
  get(index: Index): NDArray<T> | ScalarType<T>;
  set(index: Index, value: NDArray<T> | ScalarType<T>): void;

  // Reduction operations
  sum(axis?: Axis, keepdims?: boolean): NDArray<T>;
  mean(axis?: Axis, keepdims?: boolean): NDArray<FloatType<T>>;
  max(axis?: Axis, keepdims?: boolean): NDArray<T>;
  // ... more reductions

  // Operators (implement via methods)
  add(other: NDArray<T> | number): NDArray<T>;
  multiply(other: NDArray<T> | number): NDArray<T>;
  // ... more operators
}
```

### Key Design Decisions

#### Memory Layout
- **Data Storage**: Use TypedArrays (Float64Array, Int32Array, etc.)
- **Views**: Multiple NDArrays can share the same underlying buffer
- **Strides**: Calculate element access using strides (matches NumPy)

#### Indexing
Support NumPy indexing semantics:
```typescript
// Integer indexing
arr.get([0, 1, 2])

// Slicing
arr.get([slice(null), slice(0, 5)])

// Boolean indexing
arr.get([arr.greater(0)])

// Fancy indexing
arr.get([np.array([0, 2, 4])])
```

---

### 2. DType System

```typescript
abstract class DType {
  abstract readonly name: string;
  abstract readonly kind: DTypeKind; // 'i', 'f', 'c', etc.
  abstract readonly itemsize: number;
  abstract readonly byteorder: '<' | '>' | '=' | '|';

  // Type conversion
  abstract cast(value: any): any;
  abstract arrayType: TypedArrayConstructor;

  // String representation
  get str(): string; // e.g., '<f8'

  // Comparison
  equals(other: DType): boolean;
  canCastTo(other: DType, casting?: CastingMode): boolean;
}

// Scalar types
class Int8DType extends DType { ... }
class Int16DType extends DType { ... }
class Int32DType extends DType { ... }
class Float32DType extends DType { ... }
class Float64DType extends DType { ... }
class Complex64DType extends DType { ... }
class Complex128DType extends DType { ... }

// Structured types
class StructuredDType extends DType {
  readonly fields: Map<string, { dtype: DType; offset: number }>;
}

// Datetime types
class DatetimeDType extends DType {
  readonly unit: 'Y' | 'M' | 'D' | 'h' | 'm' | 's' | 'ms' | 'us' | 'ns';
}
```

### DType Registry
```typescript
const dtypes = {
  int8: new Int8DType(),
  int16: new Int16DType(),
  int32: new Int32DType(),
  int64: new Int64DType(), // Special handling needed
  float32: new Float32DType(),
  float64: new Float64DType(),
  complex64: new Complex64DType(),
  complex128: new Complex128DType(),
  // ...
};

function dtype(spec: string | DType | Type): DType {
  // Parse dtype specification
  if (typeof spec === 'string') {
    return parseDTypeString(spec);
  }
  // ...
}
```

---

### 3. Broadcasting Engine

```typescript
class BroadcastEngine {
  /**
   * Determine if shapes are broadcastable and compute output shape
   */
  static broadcast(shapes: number[][]): {
    broadcastable: boolean;
    outputShape: number[];
  } {
    // Implement NumPy broadcasting rules
  }

  /**
   * Create broadcast iterator for operands
   */
  static createIterator(arrays: NDArray[]): BroadcastIterator {
    // ...
  }

  /**
   * Broadcast array to target shape (create view, no copy)
   */
  static broadcastTo(array: NDArray, shape: number[]): NDArray {
    // ...
  }
}
```

### Broadcasting Rules (NumPy-compatible)
1. Align shapes from right
2. Dimensions are compatible if:
   - They are equal, OR
   - One of them is 1
3. Missing dimensions treated as 1

```typescript
// Examples:
// (3, 4) + (4,)     -> (3, 4)  ✓
// (3, 4) + (3, 4)   -> (3, 4)  ✓
// (3, 1) + (1, 4)   -> (3, 4)  ✓
// (3, 4) + (3,)     -> Error   ✗
```

---

### 4. Universal Functions (ufuncs)

```typescript
abstract class UniversalFunction {
  readonly name: string;
  readonly nin: number;  // Number of inputs
  readonly nout: number; // Number of outputs

  /**
   * Apply operation element-wise
   */
  abstract apply(...inputs: NDArray[]): NDArray | NDArray[];

  /**
   * Reduce along axis
   */
  reduce(array: NDArray, axis?: Axis): NDArray {
    // e.g., np.add.reduce(arr) === np.sum(arr)
  }

  /**
   * Accumulate along axis
   */
  accumulate(array: NDArray, axis?: number): NDArray {
    // e.g., np.add.accumulate(arr) === np.cumsum(arr)
  }

  /**
   * Outer product
   */
  outer(a: NDArray, b: NDArray): NDArray {
    // e.g., np.add.outer(a, b) -> a[:, None] + b[None, :]
  }

  /**
   * At specific indices
   */
  at(array: NDArray, indices: NDArray, values?: NDArray): void {
    // In-place operation at indices
  }
}
```

### Ufunc Implementation Strategy

**Phase 1: Explicit loops (correctness)**
```typescript
class AddUfunc extends UniversalFunction {
  apply(a: NDArray, b: NDArray): NDArray {
    // 1. Broadcast inputs
    const [aB, bB, outShape] = broadcast([a, b]);

    // 2. Allocate output
    const out = new NDArray({
      shape: outShape,
      dtype: promoteTypes(a.dtype, b.dtype),
    });

    // 3. Iterate and compute
    for (let i = 0; i < out.size; i++) {
      out.data[i] = aB.data[i] + bB.data[i];
    }

    return out;
  }
}
```

**Phase 2: Optimized kernels (performance)**
- Type-specialized implementations
- SIMD operations
- WASM kernels
- GPU acceleration (WebGPU)

---

##Implementation Strategy

### Phase 1: Foundation (Pure TypeScript)

**Goals:**
- Correct implementation using TypedArrays
- Complete API surface matching NumPy
- 100% test coverage via Python validation

**Not Goals:**
- Performance optimization
- Special-cased kernels
- Memory efficiency

**Benefits:**
- Fast development
- Easy debugging
- Perfect correctness validation
- Establishes test infrastructure

### Phase 2: Optimization (Selective)

After Phase 1 complete and validated:

**Hot Path Identification:**
```typescript
// Use profiling to find bottlenecks
const profiler = new Profiler();

// Run representative workloads
profiler.start();
np.dot(largeMatrix1, largeMatrix2);
np.fft.fft(largeSignal);
np.linalg.solve(A, b);
profiler.stop();

// Identify functions consuming >5% of runtime
const hotPaths = profiler.getHotPaths();
```

**Optimization Approaches:**
1. **Type Specialization**: Fast paths for common dtypes
2. **SIMD**: Leverage JavaScript SIMD (when available)
3. **WASM**: Compile critical kernels to WASM
4. **Lazy Evaluation**: Defer computation until needed
5. **Expression Templates**: Fuse operations (e.g., `a + b * c`)

---

## Memory Management

### TypedArray as Foundation

All array data stored in TypedArrays:
```typescript
type TypedArray =
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array
  | BigInt64Array
  | BigUint64Array;
```

### Views vs Copies

**Views** (no data copy):
- `reshape()` - where possible
- `transpose()` - always
- `slice()` - always
- `broadcast_to()` - always

**Copies** (data duplication):
- `copy()` - explicit
- `astype()` - always
- `flatten()` - always

### Stride Calculation

```typescript
function computeStrides(shape: number[], order: 'C' | 'F', itemsize: number): number[] {
  const ndim = shape.length;
  const strides = new Array(ndim);

  if (order === 'C') {
    // C-contiguous (row-major)
    let stride = itemsize;
    for (let i = ndim - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
  } else {
    // F-contiguous (column-major)
    let stride = itemsize;
    for (let i = 0; i < ndim; i++) {
      strides[i] = stride;
      stride *= shape[i];
    }
  }

  return strides;
}
```

### Element Access via Strides

```typescript
function getElementOffset(indices: number[], strides: number[], offset: number): number {
  let position = offset;
  for (let i = 0; i < indices.length; i++) {
    position += indices[i] * strides[i];
  }
  return position;
}
```

---

## Special Considerations

### 1. 64-bit Integers

JavaScript numbers are 64-bit floats, can't represent all int64 values exactly.

**Solutions:**
- **Option A**: Use BigInt for int64/uint64
  - Pros: Exact representation
  - Cons: Slower, different type system
- **Option B**: Use Float64, warn on precision loss
  - Pros: Simple, consistent
  - Cons: Not exact for large integers
- **Option C**: Hybrid - BigInt64Array for storage, convert on access
  - Pros: Exact storage, reasonable performance
  - Cons: Complex

**Recommendation**: Start with Option A (BigInt), provide Option B as fallback

### 2. Complex Numbers

JavaScript has no native complex type.

**Solutions:**
- **Interleaved storage**: [real0, imag0, real1, imag1, ...]
  ```typescript
  class Complex64DType extends DType {
    readonly itemsize = 8; // 2 * float32
    readonly arrayType = Float32Array;

    getReal(array: NDArray, index: number): number {
      return array.data[index * 2];
    }

    getImag(array: NDArray, index: number): number {
      return array.data[index * 2 + 1];
    }
  }
  ```

### 3. Object Arrays

NumPy supports dtype `object` for arbitrary Python objects.

**Implementation:**
- Store JavaScript objects/values in regular array
- Pickle compatibility not required (different language)
- Support JSON-serializable objects only

---

## Error Handling

### Exception Types
```typescript
class ValueError extends Error {}
class IndexError extends Error {}
class TypeError extends Error {}
class LinAlgError extends Error {}
```

### Validation Strategy
- **Always validate**: Shape compatibility, dtype compatibility
- **Fail fast**: Error immediately on invalid input
- **Clear messages**: Include shapes, dtypes in error messages

Example:
```typescript
function add(a: NDArray, b: NDArray): NDArray {
  if (!broadcastable(a.shape, b.shape)) {
    throw new ValueError(
      `Shapes ${a.shape} and ${b.shape} are not broadcastable`
    );
  }
  // ...
}
```

---

## Extensibility Points

### Custom DTypes
```typescript
interface DTypePlugin {
  name: string;
  create(spec: any): DType;
}

function registerDType(plugin: DTypePlugin): void {
  // Allow users to add custom dtypes
}
```

### Custom Backends
```typescript
interface Backend {
  name: string;
  add(a: NDArray, b: NDArray): NDArray;
  matmul(a: NDArray, b: NDArray): NDArray;
  // ...
}

function setBackend(backend: Backend): void {
  // Switch computational backend
}
```

---

## Summary

Our architecture emphasizes:
1. ✅ **Correctness**: Exact NumPy semantics
2. ✅ **Simplicity**: Straightforward implementations
3. ✅ **Testability**: Easy to validate
4. ✅ **Extensibility**: Optimization points identified
5. ✅ **Type Safety**: Full TypeScript support

Performance comes later, after we have a correct, tested implementation.

---

**Last Updated**: 2025-10-07
**Status**: Complete architecture specification
