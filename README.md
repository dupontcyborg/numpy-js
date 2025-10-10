# NumPy.js

> Complete NumPy implementation for TypeScript and JavaScript

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Status**: 🚧 In Development - Phase 0 (Project Setup)

---

## What is NumPy.js?

A complete, functionally-equivalent implementation of NumPy 2.0+ for the JavaScript ecosystem.

### Goals

- ✅ **100% NumPy 2.0+ API** - All 800+ functions
- ✅ **Full Type Safety** - Complete TypeScript definitions
- ✅ **Cross-Platform** - Node.js and browsers
- ✅ **File Compatibility** - Read/write .npy and .npz files
- ✅ **Correctness First** - Validated against Python NumPy

### Not Goals

- ❌ Matching Python NumPy's exact performance (initially)
- ❌ C API compatibility
- ❌ Legacy NumPy 1.x deprecated functions

---

## Quick Example

```typescript
import * as np from 'numpy-js';

// Create arrays
const A = np.array([[1, 2], [3, 4]]);
const B = np.zeros([2, 2]);

// Matrix operations
const C = A.matmul(B);
const eigenvalues = np.linalg.eig(A);

// Slicing (string-based syntax)
const row = A.slice('0', ':');  // First row
const col = A.col(1);            // Second column

// Broadcasting
const scaled = A.add(5).multiply(2);

// Reductions
const total = A.sum();
const columnMeans = A.mean({ axis: 0 });

// Random
const random = np.random.randn([100, 100]);

// I/O (Node.js)
np.save('matrix.npy', A);
const loaded = np.load('matrix.npy');
```

---

## Architecture

```
┌────────────────────────────────┐
│    NumPy-Compatible API        │
└────────────┬───────────────────┘
             │
┌────────────┴───────────────────┐
│  NDArray (Memory & Views)      │
│  Broadcasting, Slicing, DTypes │
└────────────┬───────────────────┘
             │
┌────────────┴───────────────────┐
│  Computational Backend         │
│  @stdlib (BLAS/LAPACK)         │
└────────────────────────────────┘
```

**We build**: NumPy API, NDArray class, broadcasting, slicing
**We use**: @stdlib for proven numerical computations

---

## Key Features

### Comprehensive NumPy API
- Array creation: `zeros`, `ones`, `arange`, `linspace`
- Math operations: `add`, `multiply`, `sin`, `cos`, `exp`, `log`
- Linear algebra: `matmul`, `solve`, `inv`, `svd`, `eig`
- Reductions: `sum`, `mean`, `std`, `min`, `max`
- Random: `rand`, `randn`, distributions
- FFT operations
- I/O: .npy/.npz files

### TypeScript Native
```typescript
// Full type inference
const arr = np.zeros([3, 4]);  // Type: NDArray<Float64>
arr.shape;  // Type: readonly [3, 4]
arr.sum();  // Type: number

// Type-safe slicing
arr.slice('0:2', ':');  // Returns NDArray
arr.get([0, 1]);        // Returns number
```

### Slicing Syntax

Since TypeScript doesn't support Python's `arr[0:5, :]` syntax, we use strings:

```typescript
// String-based (primary)
arr.slice('0:5', '1:3');     // arr[0:5, 1:3]
arr.slice(':', '-1');        // arr[:, -1]
arr.slice('::2');            // arr[::2]

// Convenience helpers
arr.row(0);                  // arr[0, :]
arr.col(2);                  // arr[:, 2]
arr.rows(0, 5);              // arr[0:5, :]
arr.cols(1, 3);              // arr[:, 1:3]
```

### Broadcasting

Automatic NumPy-style broadcasting:

```typescript
const a = np.ones([3, 4]);
const b = np.arange(4);
const c = a.add(b);  // (3, 4) + (4,) → (3, 4)
```

---

## Installation (Future)

```bash
npm install numpy-js
```

### Node.js
```typescript
import * as np from 'numpy-js/node';
```

### Browser
```typescript
import * as np from 'numpy-js/browser';
```

---

## Development Status

### Phase 0: Project Setup ✅ (Current)
- [x] Package configuration
- [x] TypeScript setup
- [x] Build system (esbuild)
- [x] Test framework (Vitest)
- [x] Documentation consolidated
- [ ] Linting (ESLint + Prettier)
- [ ] CI/CD (GitHub Actions)

### Phase 1: Core Foundation (Next)
- [ ] NDArray class with memory management
- [ ] DType system (20+ types)
- [ ] Broadcasting engine
- [ ] String-based slicing
- [ ] Array creation functions
- [ ] Basic arithmetic operations

### Phase 2: Essential Operations
- [ ] Matrix operations (using @stdlib BLAS)
- [ ] Linear algebra (using @stdlib LAPACK)
- [ ] Reductions with axis support
- [ ] Mathematical functions
- [ ] Comparison operations

### Phase 3: Extended Features
- [ ] Random number generation
- [ ] FFT operations (using fft.js)
- [ ] I/O operations (.npy/.npz)
- [ ] Advanced indexing
- [ ] Complex numbers, datetime

See [API-REFERENCE.md](./docs/API-REFERENCE.md) for complete function checklist.

---

## Documentation

- [ARCHITECTURE.md](./docs/ARCHITECTURE.md) - Design and implementation details
- [API-REFERENCE.md](./docs/API-REFERENCE.md) - Complete API checklist
- [IMPLEMENTATION-NOTES.md](./docs/IMPLEMENTATION-NOTES.md) - Development notes

---

## Testing

Two-tier testing strategy:

1. **Unit Tests** - Test our implementation
2. **Python Comparison** - Validate against NumPy

```typescript
// Unit test
it('creates 2D array of zeros', () => {
  const arr = np.zeros([2, 3]);
  expect(arr.shape).toEqual([2, 3]);
  expect(arr.sum()).toBe(0);
});

// Python comparison
it('matmul matches NumPy', async () => {
  const A = np.array([[1, 2], [3, 4]]);
  const B = np.array([[5, 6], [7, 8]]);
  const result = A.matmul(B);

  await validateAgainstNumPy(result, `
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    result = A @ B
  `);
});
```

---

## Design Decisions

### 1. BigInt for int64/uint64
Exact representation over convenience. Different type but no precision loss.

### 2. String-Based Slicing
`arr.slice('0:5', ':')` instead of `arr[0:5, :]` - TypeScript limitation, Pythonic compromise.

### 3. @stdlib for Computations
Use battle-tested BLAS/LAPACK implementations. Focus on API, not reimplementing algorithms.

### 4. Correctness First
Validate everything against Python NumPy before optimizing. WASM/SIMD later.

### 5. Complex Numbers
Interleaved storage: `[real, imag, real, imag, ...]`. Input as nested arrays: `[[1, 2], [3, 4]]`.

See [ARCHITECTURE.md](./docs/ARCHITECTURE.md) for full rationale.

---

## Contributing

Project is in early development. Setup:

```bash
git clone https://github.com/nicolasdupont/numpy-js.git
cd numpy-js
npm install
npm test
```

Pick a function from [API-REFERENCE.md](./docs/API-REFERENCE.md) and implement it!

---

## Comparison with Alternatives

| Feature | NumPy.js | numjs | ndarray | TensorFlow.js |
|---------|----------|-------|---------|---------------|
| API Coverage | 100% NumPy | ~20% | Different | ML-focused |
| TypeScript | Native | Partial | No | Yes |
| .npy files | Yes | No | No | No |
| Python-compatible | Yes | Mostly | No | No |
| Size | TBD | Small | Tiny | Large |

---

## Performance Expectations

**v1.0** (Pure JS + @stdlib):
- 10-100x slower than NumPy - acceptable for correctness focus

**v2.0** (Selective WASM):
- 2-20x slower - optimized bottlenecks only

**v3.0** (Advanced):
- 1-10x slower - SIMD, GPU for specific operations

Focus is correctness and completeness first, then performance.

---

## License

[MIT License](./LICENSE) - Copyright (c) 2025 Nicolas Dupont

---

## Links

- **Documentation**: [`docs/`](./docs)
- **NumPy**: https://numpy.org/
- **@stdlib**: https://stdlib.io/
- **Issues**: https://github.com/nicolasdupont/numpy-js/issues

---

**Ready to bring NumPy to JavaScript!** ⭐
