# Design Decisions

This document records all major design decisions made for NumPy.js, including rationale and alternatives considered.

---

## 1. 64-bit Integer Handling

**Decision**: Use `BigInt` for int64/uint64

**Rationale**:
- Provides exact representation of all 64-bit integer values
- Maintains correctness over convenience
- JavaScript numbers (float64) can't represent all integers beyond 2^53

**Implementation**:
```typescript
const arr = np.array([1n, 2n, 3n], { dtype: 'int64' });
// Stored in BigInt64Array
```

**Alternative Considered**: Use Float64 with precision warnings
- **Rejected**: Sacrifices correctness, leads to subtle bugs

**Configuration Option**: May provide config flag to fallback to Float64 for performance-critical code that doesn't need full range

---

## 2. Complex Number Representation

**Decision**: Support nested arrays (Option C) initially, with future extensions

**Python Equivalent**:
```python
# NumPy accepts this
np.array([[1, 2], [3, 4]], dtype=complex)  # → [1+2j, 3+4j]
```

**TypeScript API**:
```typescript
// Option C: Nested arrays (RECOMMENDED - matches NumPy)
np.array([[1, 2], [3, 4]], { dtype: 'complex128' });  // [1+2j, 3+4j]

// Future extensions:
// - String parsing: np.array(['1+2j', '3+4j'])
// - Objects: np.array([{real: 1, imag: 2}])
```

**Rationale**:
- Nested array format is standard NumPy
- Easy to type in TypeScript
- Natural for users coming from Python

**Storage**: Interleaved Float64Array: `[real0, imag0, real1, imag1, ...]`

**Roadmap Note**: Add string parsing ('1+2j') and object format in Phase 3 if user demand exists

---

## 3. Linear Algebra Backend

**Decision**: Pure TypeScript initially, WASM optimization in post-v1.0

**Phase 1 (v1.0)**:
- Implement all operations in pure TypeScript
- Focus on correctness
- Acceptable performance: 100-1000x slower than NumPy

**Phase 2 (post-v1.0)**:
- Integrate WASM backend (compiled LAPACK/BLAS)
- Performance target: 10x slower than NumPy
- Fallback to pure TS if WASM unavailable

**Rationale**:
- Correctness first, performance later
- Pure TS easier to debug and validate
- WASM can be swapped in without API changes

**Alternative Considered**: Integrate ml-matrix or mathjs
- **Rejected**: Different APIs, harder to ensure NumPy compatibility

---

## 4. FFT Implementation

**Decision**: Use fft.js library

**Rationale**:
- Battle-tested FFT implementation
- Good performance
- Focus our effort on NumPy API compatibility, not DSP algorithms

**Integration Strategy**:
```typescript
import FFT from 'fft.js';

export function fft(a: NDArray, n?: number, axis?: number, norm?: string): NDArray {
  // Validate inputs (NumPy compatibility)
  // Call fft.js
  // Wrap output in NDArray
  // Match NumPy's normalization conventions
}
```

**Alternative Considered**: Implement Cooley-Tukey from scratch
- **Rejected**: Time-consuming, fft.js already optimized

---

## 5. Browser vs Node.js Support

**Decision**: Support both with separate I/O APIs

### File I/O

**Node.js API**:
```typescript
import * as np from 'numpy-js/node';

np.save('array.npy', arr);           // fs.writeFileSync
const arr = np.load('array.npy');    // fs.readFileSync
np.savez('archive.npz', { a, b });   // Write ZIP file
```

**Browser API**:
```typescript
import * as np from 'numpy-js/browser';

const blob = np.save(arr);           // Returns Blob
const arr = np.load(blob);           // Parse Blob
const arr = np.load(file);           // Parse File object

// Trigger download
np.saveAndDownload('array.npy', arr);
```

**Universal API** (both environments):
```typescript
import * as np from 'numpy-js';

const buffer = np.saveToBuffer(arr);  // ArrayBuffer (works everywhere)
const arr = np.loadFromBuffer(buffer);
```

**Package Structure**:
```
numpy-js
├── index.ts           # Core (universal)
├── node.ts            # Node.js-specific (fs, memmap)
└── browser.ts         # Browser-specific (Blob, File API)
```

**Rationale**:
- Clear separation of concerns
- Explicit about environment capabilities
- Universal API for computation (core)

**Alternative Considered**: Single unified API with environment detection
- **Rejected**: Confusing when API silently does different things in different environments

---

## 6. Performance Strategy

**Decision**: Correctness first, never compromise correctness for speed

**Principles**:
1. **v1.0 Goal**: 100% functional correctness
2. **Performance**: Secondary concern initially
3. **Optimization**: Only after correctness validated
4. **No Fast Mode**: No flag that sacrifices correctness

**Performance Targets**:
- **v1.0**: 100-1000x slower acceptable
- **v2.0**: 10-100x slower (with WASM)
- **v3.0**: 1-10x slower (with SIMD/GPU)

**Rationale**:
- Fixing wrong fast code is harder than optimizing correct slow code
- Users need reliable results
- Python NumPy took years to optimize

**Optimization Strategy** (post-v1.0):
1. Profile to find hot paths
2. Add type-specialized fast paths
3. Integrate WASM for linalg/FFT
4. SIMD for element-wise operations
5. GPU acceleration (WebGPU) for specific operations

---

## 7. Object Arrays

**Decision**: Support JavaScript objects only (no Python pickle compatibility)

**Implementation**:
```typescript
// Supported
const arr = np.array([
  { name: 'Alice', age: 30 },
  { name: 'Bob', age: 25 }
], { dtype: 'object' });

// Storage: Regular JavaScript Array
```

**Limitations**:
- ✅ JavaScript objects supported
- ✅ JSON-serializable types
- ❌ No Python pickle compatibility
- ❌ Can't round-trip with Python NumPy object arrays

**Phase 1**: Support native types only (numbers, strings, booleans, null)

**Phase 2**: Support arbitrary JSON-serializable objects

**Rationale**:
- Python pickle is Python-specific
- JavaScript ecosystem uses JSON
- Clear documentation prevents confusion

**Documentation**:
```markdown
**Note**: Object dtype in NumPy.js stores JavaScript objects, not Python pickles.
Files with object arrays are not compatible between Python NumPy and NumPy.js.
For cross-language compatibility, use structured dtypes instead.
```

---

## 8. Random Number Generation

**Decision**: PCG64 (deterministic, seedable) with WebCrypto for unseeded initialization

**Implementation**:
```typescript
class Generator {
  private bitGen: PCG64;

  constructor(seed?: number | bigint) {
    if (seed !== undefined) {
      // Deterministic: same seed = same sequence
      this.bitGen = new PCG64(seed);
    } else {
      // Use WebCrypto to generate cryptographically random seed
      const randomSeed = crypto.getRandomValues(new BigUint64Array(2));
      this.bitGen = new PCG64(randomSeed);
    }
  }

  random(size?: number | number[]): NDArray {
    // Generate using PCG64
  }
}

// Usage
const rng1 = np.random.default_rng(42);  // Deterministic
const rng2 = np.random.default_rng();     // Random seed from WebCrypto
```

**Rationale**:
- **PCG64**: NumPy's default generator (since 1.17)
- **Seedable**: Required for reproducible results
- **WebCrypto**: Better than Math.random() for unseeded case
- **Statistical equivalence**: Good enough (not bit-for-bit identical)

**Not cryptographically secure**:
- Like NumPy, this is for statistical simulation, not security
- Document clearly: "Not suitable for cryptographic purposes"

**Alternative Considered**: Use WebCrypto exclusively
- **Rejected**: Can't be seeded, breaks reproducibility

---

## 9. Memory Management

**Decision**: Hybrid approach - automatic GC + optional explicit disposal

### Automatic (Default)
```typescript
const arr = np.zeros([10000, 10000]);
// ... use arr
// Automatically garbage collected when no longer referenced
```

### Explicit Disposal (Optional)
```typescript
const huge = np.zeros([10000, 10000]);  // 800MB
// ... use it
huge.dispose();  // Free immediately, don't wait for GC

// Trying to use after disposal throws error
huge.sum();  // Error: Array has been disposed
```

### Using Statement (TypeScript 5.2+)
```typescript
{
  using arr = np.zeros([10000, 10000]);
  // ... use arr
  // Automatically disposed at end of block
}
```

**Implementation**:
```typescript
class NDArray implements Disposable {
  private _disposed = false;

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    // Null out references to help GC
    this.data = null;
  }

  [Symbol.dispose](): void {
    this.dispose();
  }

  private checkDisposed(): void {
    if (this._disposed) {
      throw new Error('Array has been disposed');
    }
  }

  // All methods call checkDisposed()
  sum(): number {
    this.checkDisposed();
    // ...
  }
}
```

**Rationale**:
- **Default**: Familiar JavaScript behavior
- **Optional disposal**: For performance-critical code
- **Safety**: Error on use-after-dispose
- **Best of both worlds**: Simple for beginners, control for experts

**When to use `.dispose()`**:
- Very large arrays (>100MB)
- Tight loops creating many arrays
- WebGL texture backed arrays
- Memory-constrained environments

---

## 10. JavaScript-Friendly API Extensions

**Decision**: Support both NumPy API and JavaScript idioms where possible

### Core Principle
- **Primary API**: Strict NumPy compatibility
- **Extensions**: JavaScript-friendly additions that don't break NumPy compatibility

### Dual API Support

#### 1. Iteration
```typescript
// NumPy way (always supported)
for (let i = 0; i < arr.shape[0]; i++) {
  console.log(arr.get([i]));
}

// JavaScript way (extension)
for (const row of arr) {
  console.log(row);  // Iterate over first axis
}

// Even more JS
arr[Symbol.iterator]();
Array.from(arr);
```

#### 2. Functional Operations
```typescript
// NumPy way
const positive = arr.get([arr.greater(0)]);
const doubled = np.multiply(arr, 2);

// JavaScript way (extension - if feasible)
const positive = arr.filter(x => x > 0);
const doubled = arr.map(x => x * 2);
```

#### 3. Async Operations
```typescript
// NumPy way (synchronous)
const arr = np.load('huge.npy');

// Extension (Node.js/Browser)
const arr = await np.loadAsync('huge.npy');
```

#### 4. Method Chaining
```typescript
// NumPy way (functional)
np.sum(np.abs(np.subtract(arr, mean)))

// Extension: fluent API
arr.subtract(mean).abs().sum()

// Both styles supported!
```

#### 5. Serialization
```typescript
// NumPy way
np.save('arr.npy', arr);

// Extension: JSON (web-friendly)
const json = arr.toJSON();
localStorage.setItem('array', JSON.stringify(json));
const arr2 = np.fromJSON(JSON.parse(localStorage.getItem('array')));
```

### Implementation Strategy

**Option A**: Extensions on main API
```typescript
class NDArray {
  // NumPy methods
  sum(axis?: Axis): NDArray { }

  // Extensions (clearly documented)
  [Symbol.iterator](): Iterator<NDArray> { }
  map(fn: (x: number) => number): NDArray { }
  toJSON(): SerializedArray { }
}
```

**Option B**: Separate namespace
```typescript
// NumPy (strict)
arr.sum()

// Extensions
arr.js.forEach(x => ...)
np.js.loadAsync('file.npy')
```

**Decision**: **Option A** - Extensions on main API with clear documentation

**Rationale**:
- More ergonomic for JavaScript developers
- No conflicting APIs (extensions are additive)
- TypeScript types make usage clear
- Documentation clearly marks extensions

**Documentation Strategy**:
```typescript
/**
 * Returns sum of array elements over given axis.
 *
 * @param axis - Axis or axes along which to sum
 * @returns Sum of elements
 *
 * @example
 * ```typescript
 * const arr = np.array([[1, 2], [3, 4]]);
 * arr.sum();        // 10
 * arr.sum({axis: 0}); // [4, 6]
 * ```
 *
 * **NumPy Equivalent**: `numpy.ndarray.sum`
 */
sum(axis?: Axis): NDArray;

/**
 * Iterate over the first axis of the array.
 *
 * @example
 * ```typescript
 * const arr = np.array([[1, 2], [3, 4]]);
 * for (const row of arr) {
 *   console.log(row); // [1, 2], then [3, 4]
 * }
 * ```
 *
 * **Extension**: This is a NumPy.js extension for JavaScript ergonomics.
 * Not available in Python NumPy.
 */
[Symbol.iterator](): Iterator<NDArray>;
```

### Extensions Priority

**Phase 1** (v1.0):
- ✅ `[Symbol.iterator]` for iteration
- ✅ `.toJSON()` / `fromJSON()` for serialization
- ✅ `.dispose()` for memory management

**Phase 2** (v1.1):
- ✅ `.map()`, `.filter()`, `.reduce()` functional methods
- ✅ `async` versions of I/O operations
- ✅ Fluent API / method chaining

**Phase 3** (v1.2):
- ✅ Integration with Observable, D3.js
- ✅ Streaming operations for large datasets
- ✅ WebGPU tensor operations

---

## Summary Table

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 64-bit integers | BigInt | Correctness over convenience |
| Complex numbers | Nested arrays [[r,i]] | Matches NumPy, extensible |
| Linear algebra | Pure TS → WASM | Correctness first |
| FFT | fft.js library | Focus on API, not algorithms |
| Browser/Node | Separate APIs | Explicit capabilities |
| Performance | Correctness first | Never compromise correctness |
| Object arrays | JS objects only | No pickle compatibility |
| Random | PCG64 + WebCrypto | Seedable + secure unseeded |
| Memory | Auto GC + optional dispose | Best of both worlds |
| JS extensions | Support both | Additive, not conflicting |

---

## Future Decisions

These will be decided during implementation:

### 1. Build System
- esbuild vs rollup vs webpack
- Separate bundles for node/browser?

### 2. TypeScript Configuration
- Target ES2020+ or ES2015?
- Module system: ESM only or CommonJS too?

### 3. Testing Framework
- Vitest vs Jest
- Browser testing strategy (Playwright?)

### 4. Documentation
- TypeDoc vs Docusaurus
- Live examples (Observable? StackBlitz?)

### 5. Package Distribution
- npm only or other registries?
- CDN for browser (unpkg, jsdelivr)?

---

**Last Updated**: 2025-10-07
**Status**: Core design decisions finalized
