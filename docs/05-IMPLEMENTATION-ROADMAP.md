# Implementation Roadmap

## Overview

This roadmap breaks down the implementation of NumPy.js into manageable phases, with specific milestones and deliverables. Each phase builds on the previous, ensuring we maintain correctness throughout.

**Estimated Timeline**: 12-18 months to v1.0 (full API coverage)

---

## Phase 0: Project Setup (Weeks 1-2)

### Goals
- Establish project infrastructure
- Set up development environment
- Create initial test framework

### Tasks

#### Project Structure
- [x] Create project repository
- [ ] Initialize npm/package.json
- [ ] Configure TypeScript (tsconfig.json)
  - Strict mode enabled
  - ES2020+ target
  - Module: ESNext
- [ ] Set up build system (esbuild or rollup)
- [ ] Configure linter (ESLint) and formatter (Prettier)
- [ ] Set up Git hooks (husky + lint-staged)

#### Testing Infrastructure
- [ ] Install test framework (Vitest or Jest)
- [ ] Set up Python test oracle
  - Python environment with NumPy 2.0+
  - Test harness scripts
  - Data serialization utilities
- [ ] Create test utilities
  - `assertArrayEqual`
  - `assertArrayClose`
  - Python subprocess runner
- [ ] Write first end-to-end test (validate setup)

#### Documentation
- [ ] README.md with project goals
- [ ] CONTRIBUTING.md guidelines
- [ ] Development setup instructions

#### CI/CD
- [ ] GitHub Actions workflow
  - Run tests on push
  - TypeScript type checking
  - Linting
- [ ] Code coverage reporting (Codecov)

### Deliverables
- ✅ Working project structure
- ✅ Test harness validating against Python NumPy
- ✅ CI pipeline running

---

## Phase 1: Core Foundation (Months 1-3)

### Goals
- Implement core NDArray class
- Basic dtype system
- Fundamental array operations
- Array creation routines

### Milestone 1.1: DType System (Weeks 3-4)

#### Tasks
- [ ] Design DType class hierarchy
- [ ] Implement scalar dtypes:
  - [ ] `int8`, `int16`, `int32`
  - [ ] `uint8`, `uint16`, `uint32`
  - [ ] `float32`, `float64`
  - [ ] `bool`
- [ ] Implement dtype utilities:
  - [ ] `dtype()` constructor
  - [ ] `result_type()` - type promotion
  - [ ] `can_cast()` - casting rules
  - [ ] `promote_types()` - common type
- [ ] Create dtype registry
- [ ] Write comprehensive dtype tests

#### Validation
- [ ] All dtypes round-trip with Python NumPy
- [ ] Type promotion matches NumPy exactly
- [ ] Casting rules match NumPy

### Milestone 1.2: NDArray Core (Weeks 5-7)

#### Tasks
- [ ] Implement NDArray class:
  - [ ] Core properties (data, shape, strides, dtype)
  - [ ] Memory layout (C-order only initially)
  - [ ] Stride calculation
  - [ ] Element access via strides
- [ ] Implement array properties:
  - [ ] `ndim`, `size`, `nbytes`, `itemsize`
  - [ ] `flags` (c_contiguous, etc.)
  - [ ] `T` (transpose view)
  - [ ] `shape`, `strides` accessors
- [ ] Basic methods:
  - [ ] `item()` - get scalar
  - [ ] `tolist()` - convert to nested arrays
  - [ ] `copy()` - deep copy
- [ ] Write NDArray tests

#### Validation
- [ ] Arrays created in JS match Python NumPy exactly
- [ ] Strides computed correctly
- [ ] Element access is accurate

### Milestone 1.3: Array Creation (Weeks 8-9)

#### Tasks
- [ ] Implement creation functions:
  - [ ] `array()` - from nested arrays
  - [ ] `zeros()`, `ones()`, `empty()`
  - [ ] `zeros_like()`, `ones_like()`, `empty_like()`
  - [ ] `full()`, `full_like()`
  - [ ] `eye()`, `identity()`
  - [ ] `arange()`, `linspace()`, `logspace()`
- [ ] Write creation tests
- [ ] Cross-validate ALL functions against Python

#### Validation
- [ ] 100% API compatibility for creation routines
- [ ] All dtypes supported
- [ ] All shape variations work

### Milestone 1.4: Basic Indexing (Weeks 10-11)

#### Tasks
- [ ] Implement integer indexing:
  - [ ] Single element `arr[0]`
  - [ ] Multi-dimensional `arr[0, 1, 2]`
- [ ] Implement slicing:
  - [ ] Basic slices `arr[1:5]`
  - [ ] Step slices `arr[::2]`
  - [ ] Negative indices `arr[-1]`
  - [ ] Ellipsis `arr[..., 0]`
- [ ] Implement assignment:
  - [ ] `arr[0] = 5`
  - [ ] `arr[1:5] = [1, 2, 3, 4]`
- [ ] Write indexing tests

#### Validation
- [ ] Indexing matches NumPy exactly
- [ ] Views share memory correctly
- [ ] Assignment works for all cases

### Milestone 1.5: Shape Manipulation (Weeks 12)

#### Tasks
- [ ] Implement shape operations:
  - [ ] `reshape()` - with views where possible
  - [ ] `ravel()`, `flatten()`
  - [ ] `transpose()`, `swapaxes()`
  - [ ] `squeeze()`, `expand_dims()`
- [ ] Write shape manipulation tests

#### Validation
- [ ] All operations match NumPy
- [ ] Views vs copies match NumPy behavior

---

## Phase 2: Core Functionality (Months 4-6)

### Goals
- Broadcasting engine
- Universal functions (ufuncs)
- Mathematical operations
- Reduction operations

### Milestone 2.1: Broadcasting (Weeks 13-14)

#### Tasks
- [ ] Implement broadcasting engine:
  - [ ] `broadcastable()` - check compatibility
  - [ ] `broadcast_shapes()` - compute output shape
  - [ ] `broadcast_to()` - broadcast array to shape
  - [ ] `broadcast_arrays()` - broadcast multiple arrays
- [ ] Create broadcast iterator
- [ ] Write broadcasting tests
- [ ] Test edge cases (0-D arrays, size-1 dimensions)

#### Validation
- [ ] Broadcasting rules match NumPy exactly
- [ ] All edge cases handled

### Milestone 2.2: Arithmetic Ufuncs (Weeks 15-16)

#### Tasks
- [ ] Create UniversalFunction base class
- [ ] Implement arithmetic ufuncs:
  - [ ] `add`, `subtract`, `multiply`, `divide`
  - [ ] `power`, `mod`, `floor_divide`
  - [ ] `negative`, `positive`, `absolute`
  - [ ] `sign`, `reciprocal`
- [ ] Implement ufunc methods:
  - [ ] `.reduce()` - e.g., `np.add.reduce(arr)`
  - [ ] `.accumulate()` - e.g., `np.add.accumulate(arr)`
  - [ ] `.outer()` - outer product
- [ ] Write arithmetic ufunc tests

#### Validation
- [ ] All ufuncs produce identical results to NumPy
- [ ] Type promotion works correctly
- [ ] Broadcasting integrated

### Milestone 2.3: Comparison and Logical Ufuncs (Week 17)

#### Tasks
- [ ] Implement comparison ufuncs:
  - [ ] `greater`, `greater_equal`
  - [ ] `less`, `less_equal`
  - [ ] `equal`, `not_equal`
- [ ] Implement logical ufuncs:
  - [ ] `logical_and`, `logical_or`, `logical_not`, `logical_xor`
- [ ] Write comparison/logical tests

#### Validation
- [ ] All comparisons match NumPy
- [ ] Boolean dtype handled correctly

### Milestone 2.4: Mathematical Functions (Weeks 18-20)

#### Tasks
- [ ] Implement trigonometric functions:
  - [ ] `sin`, `cos`, `tan`
  - [ ] `arcsin`, `arccos`, `arctan`, `arctan2`
  - [ ] `hypot`, `degrees`, `radians`
- [ ] Implement hyperbolic functions:
  - [ ] `sinh`, `cosh`, `tanh`
  - [ ] `arcsinh`, `arccosh`, `arctanh`
- [ ] Implement exponential/logarithm:
  - [ ] `exp`, `exp2`, `expm1`
  - [ ] `log`, `log2`, `log10`, `log1p`
  - [ ] `logaddexp`, `logaddexp2`
- [ ] Implement rounding:
  - [ ] `round`, `rint`, `floor`, `ceil`, `trunc`, `fix`
- [ ] Implement floating point:
  - [ ] `signbit`, `copysign`, `frexp`, `ldexp`
  - [ ] `nextafter`, `spacing`
- [ ] Write mathematical function tests

#### Validation
- [ ] Numerical results match within tolerance
- [ ] Edge cases (NaN, Inf) handled correctly

### Milestone 2.5: Reduction Operations (Weeks 21-22)

#### Tasks
- [ ] Implement reductions:
  - [ ] `sum`, `prod`
  - [ ] `mean`, `std`, `var`
  - [ ] `min`, `max`
  - [ ] `argmin`, `argmax`
  - [ ] `any`, `all`
  - [ ] `cumsum`, `cumprod`
- [ ] Support `axis` parameter
- [ ] Support `keepdims` parameter
- [ ] Implement NaN-safe versions:
  - [ ] `nansum`, `nanmean`, `nanstd`, `nanvar`
  - [ ] `nanmin`, `nanmax`
- [ ] Write reduction tests

#### Validation
- [ ] All reductions match NumPy
- [ ] Axis handling correct
- [ ] NaN handling matches NumPy

### Milestone 2.6: Array Manipulation (Weeks 23-24)

#### Tasks
- [ ] Implement joining:
  - [ ] `concatenate`, `stack`, `vstack`, `hstack`, `dstack`
  - [ ] `block`, `column_stack`
- [ ] Implement splitting:
  - [ ] `split`, `array_split`
  - [ ] `vsplit`, `hsplit`, `dsplit`
- [ ] Implement tiling:
  - [ ] `tile`, `repeat`
- [ ] Implement adding/removing:
  - [ ] `append`, `insert`, `delete`
  - [ ] `unique`, `pad`
- [ ] Implement rearranging:
  - [ ] `flip`, `fliplr`, `flipud`
  - [ ] `roll`, `rot90`
- [ ] Write manipulation tests

#### Validation
- [ ] All operations match NumPy
- [ ] Memory sharing correct

---

## Phase 3: Advanced Features (Months 7-9)

### Goals
- Linear algebra
- FFT
- Random number generation
- Statistics
- Sorting and searching

### Milestone 3.1: Linear Algebra Basics (Weeks 25-27)

#### Tasks
- [ ] Implement products:
  - [ ] `dot`, `vdot`, `inner`, `outer`
  - [ ] `matmul` (matrix multiplication)
  - [ ] `tensordot`, `einsum`
  - [ ] `kron` (Kronecker product)
- [ ] Implement norms:
  - [ ] `norm`, `matrix_norm`, `vector_norm`
- [ ] Write linalg basic tests

#### Validation
- [ ] Matrix operations match NumPy
- [ ] Einstein summation works

### Milestone 3.2: Linear Algebra Advanced (Weeks 28-30)

#### Tasks
- [ ] Implement solving:
  - [ ] `solve` - solve linear system
  - [ ] `lstsq` - least squares
  - [ ] `inv` - matrix inverse
  - [ ] `pinv` - pseudo-inverse
- [ ] Implement decompositions:
  - [ ] `qr` - QR decomposition
  - [ ] `svd` - Singular Value Decomposition
  - [ ] `cholesky` - Cholesky decomposition
- [ ] Implement eigenvalues:
  - [ ] `eig`, `eigh`
  - [ ] `eigvals`, `eigvalsh`
- [ ] Implement other:
  - [ ] `det` - determinant
  - [ ] `matrix_rank`
  - [ ] `slogdet`
- [ ] Write linalg advanced tests

#### Validation
- [ ] All decompositions match NumPy
- [ ] Numerical stability tested

**Note**: May need to integrate with existing JS linear algebra library (e.g., ml-matrix) or use WASM (e.g., compiled LAPACK)

### Milestone 3.3: FFT Operations (Weeks 31-32)

#### Tasks
- [ ] Implement standard FFTs:
  - [ ] `fft`, `ifft` - 1-D
  - [ ] `fft2`, `ifft2` - 2-D
  - [ ] `fftn`, `ifftn` - N-D
- [ ] Implement real FFTs:
  - [ ] `rfft`, `irfft` - 1-D real
  - [ ] `rfft2`, `irfft2` - 2-D real
  - [ ] `rfftn`, `irfftn` - N-D real
- [ ] Implement Hermitian FFTs:
  - [ ] `hfft`, `ihfft`
- [ ] Implement helpers:
  - [ ] `fftfreq`, `rfftfreq`
  - [ ] `fftshift`, `ifftshift`
- [ ] Write FFT tests

#### Validation
- [ ] FFT results match NumPy within tolerance
- [ ] Inverse operations work correctly

**Note**: Use existing JS FFT library (e.g., fft.js) or implement Cooley-Tukey algorithm

### Milestone 3.4: Random Number Generation (Weeks 33-35)

#### Tasks
- [ ] Implement BitGenerators:
  - [ ] `PCG64` (default)
  - [ ] `MT19937` (Mersenne Twister)
- [ ] Implement Generator class
- [ ] Implement simple random:
  - [ ] `random`, `integers`, `choice`, `bytes`
- [ ] Implement permutations:
  - [ ] `shuffle`, `permutation`
- [ ] Implement distributions:
  - [ ] `uniform`, `normal`, `standard_normal`
  - [ ] `exponential`, `gamma`, `beta`
  - [ ] `binomial`, `poisson`, `multinomial`
  - [ ] And ~30 more distributions...
- [ ] Write random tests

#### Validation
- [ ] Statistical properties match NumPy
- [ ] Seeding produces identical sequences (where possible)

**Note**: Exact bit-for-bit compatibility may not be achievable, but statistical properties should match

### Milestone 3.5: Statistics (Weeks 36-37)

#### Tasks
- [ ] Implement order statistics:
  - [ ] `ptp`, `percentile`, `quantile`
  - [ ] `nanpercentile`, `nanquantile`
- [ ] Implement averages/variances:
  - [ ] `median`, `average`
  - [ ] `nanmedian`
- [ ] Implement correlating:
  - [ ] `corrcoef`, `correlate`, `cov`
- [ ] Implement histograms:
  - [ ] `histogram`, `histogram2d`, `histogramdd`
  - [ ] `bincount`, `digitize`
- [ ] Write statistics tests

#### Validation
- [ ] All statistics match NumPy

### Milestone 3.6: Sorting and Searching (Weeks 38-39)

#### Tasks
- [ ] Implement sorting:
  - [ ] `sort`, `argsort`, `lexsort`
  - [ ] `partition`, `argpartition`
  - [ ] `sort_complex`
- [ ] Implement searching:
  - [ ] `argmax`, `argmin`, `nanargmax`, `nanargmin`
  - [ ] `argwhere`, `nonzero`, `flatnonzero`
  - [ ] `where`, `searchsorted`, `extract`
- [ ] Implement counting:
  - [ ] `count_nonzero`
- [ ] Write sorting/searching tests

#### Validation
- [ ] Sorting algorithms stable where required
- [ ] Search results match NumPy

---

## Phase 4: I/O and Specialized Features (Months 10-12)

### Goals
- File I/O (.npy/.npz support)
- Polynomials
- Set operations
- Bitwise operations
- String operations
- Datetime support
- Masked arrays

### Milestone 4.1: File I/O (Weeks 40-42)

#### Tasks
- [ ] Implement .npy format:
  - [ ] Reader (version 1.0, 2.0, 3.0)
  - [ ] Writer (version 1.0, 2.0)
  - [ ] Header parser
  - [ ] DType serialization
- [ ] Implement .npz format:
  - [ ] Uncompressed reader/writer
  - [ ] Compressed reader/writer (with zlib)
- [ ] Implement text I/O:
  - [ ] `loadtxt`, `savetxt`
  - [ ] `genfromtxt`
- [ ] Implement memory mapping:
  - [ ] `memmap` (Node.js only)
- [ ] Write I/O tests
- [ ] Cross-validate with Python NumPy files

#### Validation
- [ ] Files written by NumPy.js readable by Python NumPy
- [ ] Files written by Python NumPy readable by NumPy.js
- [ ] Round-trip preservation of data

### Milestone 4.2: Polynomials (Weeks 43-44)

#### Tasks
- [ ] Implement polynomial package:
  - [ ] `Polynomial` class (power series)
  - [ ] `Chebyshev` class
  - [ ] `Legendre` class
  - [ ] `Laguerre` class
  - [ ] `Hermite` class
  - [ ] `HermiteE` class
- [ ] Implement operations:
  - [ ] Evaluation (`polyval`, `chebval`, etc.)
  - [ ] Fitting (`polyfit`, `chebfit`, etc.)
  - [ ] Arithmetic (`polyadd`, `polymul`, etc.)
  - [ ] Calculus (`polyder`, `polyint`, etc.)
  - [ ] Roots (`polyroots`, `chebroots`, etc.)
- [ ] Write polynomial tests

#### Validation
- [ ] Polynomial evaluations match NumPy
- [ ] Fitting results match

### Milestone 4.3: Set Operations and Bitwise (Weeks 45)

#### Tasks
- [ ] Implement set operations:
  - [ ] `unique`, `in1d`, `isin`
  - [ ] `intersect1d`, `union1d`
  - [ ] `setdiff1d`, `setxor1d`
- [ ] Implement bitwise operations:
  - [ ] `bitwise_and`, `bitwise_or`, `bitwise_xor`
  - [ ] `bitwise_not`, `invert`
  - [ ] `left_shift`, `right_shift`
  - [ ] `packbits`, `unpackbits`
- [ ] Write set/bitwise tests

#### Validation
- [ ] All operations match NumPy

### Milestone 4.4: Complex Numbers and Advanced DTypes (Weeks 46-47)

#### Tasks
- [ ] Implement complex dtypes:
  - [ ] `complex64`, `complex128`
  - [ ] Interleaved storage
  - [ ] Nested array input format: `np.array([[1, 2]], {dtype: 'complex128'})` → `1+2j`
  - [ ] `real`, `imag`, `angle`, `conj` operations
  - [ ] Note: Consider adding string parsing ('1+2j') and object format later if needed
- [ ] Implement int64/uint64:
  - [ ] Using BigInt
  - [ ] Conversion utilities
- [ ] Implement structured dtypes:
  - [ ] Field access
  - [ ] Nested structures
- [ ] Implement datetime dtypes:
  - [ ] `datetime64`, `timedelta64`
  - [ ] Business day functions
- [ ] Write complex/advanced dtype tests

#### Validation
- [ ] Complex arithmetic matches NumPy
- [ ] Structured array access works
- [ ] Datetime operations match

### Milestone 4.5: String Operations (Week 48)

#### Tasks
- [ ] Implement string operations:
  - [ ] Character operations (add, multiply, mod, etc.)
  - [ ] Case manipulation (upper, lower, title, etc.)
  - [ ] Splitting/joining (split, join, partition, etc.)
  - [ ] Testing (isalpha, isdigit, isspace, etc.)
  - [ ] Comparison (equal, greater, less, etc.)
- [ ] Write string operation tests

#### Validation
- [ ] String operations match NumPy

### Milestone 4.6: Masked Arrays (Weeks 49-50)

#### Tasks
- [ ] Implement MaskedArray class
- [ ] Implement creation:
  - [ ] `ma.array`, `ma.masked_array`
- [ ] Implement inspection:
  - [ ] `ma.getmask`, `ma.getdata`
  - [ ] `ma.count`, `ma.count_masked`
- [ ] Implement manipulation:
  - [ ] `ma.filled`, `ma.compressed`
- [ ] Implement operations:
  - [ ] Arithmetic with mask propagation
  - [ ] Reductions ignoring masked values
- [ ] Write masked array tests

#### Validation
- [ ] Masked operations match NumPy

---

## Phase 5: Polish and Release (Month 12+)

### Goals
- Complete API coverage
- Documentation
- Performance baseline
- v1.0 release

### Milestone 5.1: API Completeness (Weeks 51-52)

#### Tasks
- [ ] Review API coverage against checklist (docs/01-API-INVENTORY.md)
- [ ] Implement any missing functions
- [ ] Fix any API inconsistencies
- [ ] Ensure 100% test coverage for public API

#### Validation
- [ ] Every function in docs/01-API-INVENTORY.md implemented
- [ ] All functions validated against Python NumPy

### Milestone 5.2: Documentation (Weeks 53-54)

#### Tasks
- [ ] Write API documentation (TypeDoc)
- [ ] Create user guide
- [ ] Write migration guide (from Python NumPy)
- [ ] Create examples and tutorials
- [ ] Write performance guide
- [ ] Document known differences from NumPy

#### Deliverables
- [ ] Comprehensive API docs
- [ ] User guide with examples
- [ ] Interactive examples (observable notebooks?)

### Milestone 5.3: Performance Baseline (Week 55)

#### Tasks
- [ ] Run comprehensive benchmark suite
- [ ] Compare against Python NumPy
- [ ] Document performance characteristics
- [ ] Identify optimization opportunities
- [ ] Create performance tracking

#### Deliverables
- [ ] Performance report
- [ ] Benchmark results
- [ ] Optimization roadmap

### Milestone 5.4: v1.0 Release (Week 56)

#### Tasks
- [ ] Final testing pass
- [ ] Fix critical bugs
- [ ] Prepare release notes
- [ ] Publish to npm
- [ ] Announce release

#### Success Criteria
- [ ] 100% API coverage (all 800+ functions)
- [ ] 100% test coverage
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Performance baseline established

---

## Post-v1.0: Optimization Phase

After v1.0, focus shifts to performance optimization while maintaining correctness.

### Optimization Roadmap

#### Phase 6: Type Specialization (Months 13-15)
- Fast paths for common dtypes
- Eliminate unnecessary type checking
- Specialized kernels for float64, int32

#### Phase 7: WASM Integration (Months 16-18)
- Compile hot paths to WASM
- Integrate with existing BLAS/LAPACK implementations
- Benchmark improvements

#### Phase 8: Advanced Optimizations (Months 19-21)
- SIMD where available
- Lazy evaluation
- Expression templates (fuse operations)
- GPU acceleration (WebGPU) for select operations

---

## Risk Mitigation

### Technical Risks

**Risk**: Linear algebra operations too slow in pure JS
- **Mitigation**: Plan to integrate WASM library (e.g., compiled LAPACK)
- **Timeline**: Investigate during Phase 3

**Risk**: FFT performance inadequate
- **Mitigation**: Use existing optimized JS FFT library (fft.js)
- **Timeline**: Phase 3

**Risk**: Test harness too slow
- **Mitigation**: Cache Python results, run tests in parallel
- **Timeline**: Ongoing

### Process Risks

**Risk**: Scope creep (adding features not in NumPy)
- **Mitigation**: Strict adherence to NumPy API only
- **Timeline**: Ongoing

**Risk**: API drift (NumPy releases new versions)
- **Mitigation**: Target NumPy 2.0 specifically, plan for updates
- **Timeline**: Ongoing

---

## Success Metrics

### v1.0 Success Criteria
- ✅ 100% API coverage (800+ functions)
- ✅ 100% validation tests passing
- ✅ .npy/.npz file compatibility
- ✅ Comprehensive documentation
- ✅ Published to npm

### Quality Metrics
- Unit test coverage: >90%
- Validation test coverage: 100% of API
- Documentation coverage: 100% of public API

### Performance Metrics (Initial)
- Simple operations: Within 100x of NumPy (acceptable)
- Complex operations: Within 1000x of NumPy (acceptable for v1.0)
- Post-optimization: Within 10x of NumPy (target for v2.0)

---

## Summary

This roadmap provides a clear path from initial setup to v1.0 release, with:
- **12-month timeline** to full API coverage
- **56 milestones** with specific deliverables
- **800+ functions** systematically implemented
- **Continuous validation** against Python NumPy
- **Extensibility** for future optimization

The key principle: **Correctness first, performance later**.

---

**Last Updated**: 2025-10-07
**Status**: Complete implementation roadmap
