# API Reference

Complete NumPy 2.0+ API checklist. Check off as implemented.

---

## Array Creation

### From Shape
- [ ] `zeros(shape, dtype?)` - Array of zeros
- [ ] `ones(shape, dtype?)` - Array of ones
- [ ] `empty(shape, dtype?)` - Uninitialized array
- [ ] `full(shape, fill_value, dtype?)` - Array filled with value
- [ ] `eye(n, m?, k?, dtype?)` - Identity matrix
- [ ] `identity(n, dtype?)` - Square identity matrix

### From Data
- [ ] `array(object, dtype?)` - Create from nested arrays
- [ ] `asarray(a, dtype?)` - Convert to array
- [ ] `copy(a)` - Deep copy

### Numerical Ranges
- [ ] `arange(start, stop, step?, dtype?)` - Evenly spaced values
- [ ] `linspace(start, stop, num?)` - Evenly spaced over interval
- [ ] `logspace(start, stop, num?, base?)` - Log-spaced values
- [ ] `geomspace(start, stop, num?)` - Geometric progression

### Like Functions
- [ ] `zeros_like(a, dtype?)` - Zeros with same shape
- [ ] `ones_like(a, dtype?)` - Ones with same shape
- [ ] `empty_like(a, dtype?)` - Empty with same shape
- [ ] `full_like(a, fill_value, dtype?)` - Full with same shape

---

## Array Manipulation

### Shape
- [ ] `reshape(a, shape)` - New shape
- [ ] `ravel(a)` - Flatten to 1D
- [ ] `flatten(a)` - Flatten (copy)
- [ ] `squeeze(a, axis?)` - Remove single-dimensional entries
- [ ] `expand_dims(a, axis)` - Add dimension

### Transpose
- [ ] `transpose(a, axes?)` - Permute dimensions
- [ ] `swapaxes(a, axis1, axis2)` - Swap two axes
- [ ] `moveaxis(a, source, destination)` - Move axes

### Joining
- [ ] `concatenate(arrays, axis?)` - Join arrays
- [ ] `stack(arrays, axis?)` - Stack along new axis
- [ ] `vstack(arrays)` - Stack vertically
- [ ] `hstack(arrays)` - Stack horizontally
- [ ] `dstack(arrays)` - Stack depth-wise

### Splitting
- [ ] `split(a, indices_or_sections, axis?)` - Split into sub-arrays
- [ ] `array_split(a, indices_or_sections, axis?)` - Split (unequal)
- [ ] `vsplit(a, indices_or_sections)` - Split vertically
- [ ] `hsplit(a, indices_or_sections)` - Split horizontally

### Tiling
- [ ] `tile(a, reps)` - Tile array
- [ ] `repeat(a, repeats, axis?)` - Repeat elements

---

## Mathematical Operations

### Arithmetic
- [ ] `add(x1, x2)` - Addition
- [ ] `subtract(x1, x2)` - Subtraction
- [ ] `multiply(x1, x2)` - Multiplication
- [ ] `divide(x1, x2)` - Division
- [ ] `power(x1, x2)` - Power
- [ ] `mod(x1, x2)` - Modulo
- [ ] `floor_divide(x1, x2)` - Floor division
- [ ] `negative(x)` - Negate
- [ ] `positive(x)` - Positive
- [ ] `absolute(x)` - Absolute value
- [ ] `sign(x)` - Sign
- [ ] `reciprocal(x)` - Reciprocal

### Trigonometric
- [ ] `sin(x)` - Sine
- [ ] `cos(x)` - Cosine
- [ ] `tan(x)` - Tangent
- [ ] `arcsin(x)` - Inverse sine
- [ ] `arccos(x)` - Inverse cosine
- [ ] `arctan(x)` - Inverse tangent
- [ ] `arctan2(x1, x2)` - Four-quadrant inverse tangent
- [ ] `hypot(x1, x2)` - Hypotenuse
- [ ] `degrees(x)` - Radians to degrees
- [ ] `radians(x)` - Degrees to radians

### Hyperbolic
- [ ] `sinh(x)` - Hyperbolic sine
- [ ] `cosh(x)` - Hyperbolic cosine
- [ ] `tanh(x)` - Hyperbolic tangent
- [ ] `arcsinh(x)` - Inverse hyperbolic sine
- [ ] `arccosh(x)` - Inverse hyperbolic cosine
- [ ] `arctanh(x)` - Inverse hyperbolic tangent

### Exponential and Logarithmic
- [ ] `exp(x)` - Exponential
- [ ] `expm1(x)` - exp(x) - 1
- [ ] `exp2(x)` - 2^x
- [ ] `log(x)` - Natural logarithm
- [ ] `log10(x)` - Base-10 logarithm
- [ ] `log2(x)` - Base-2 logarithm
- [ ] `log1p(x)` - log(1 + x)
- [ ] `logaddexp(x1, x2)` - log(exp(x1) + exp(x2))

### Rounding
- [ ] `around(a, decimals?)` - Round
- [ ] `round(a, decimals?)` - Round (alias)
- [ ] `floor(x)` - Floor
- [ ] `ceil(x)` - Ceiling
- [ ] `trunc(x)` - Truncate
- [ ] `rint(x)` - Round to nearest integer

### Other Math
- [ ] `sqrt(x)` - Square root
- [ ] `cbrt(x)` - Cube root
- [ ] `square(x)` - Square
- [ ] `clip(a, min, max)` - Clip values

---

## Reductions

### Sum and Product
- [ ] `sum(a, axis?, keepdims?)` - Sum
- [ ] `prod(a, axis?, keepdims?)` - Product
- [ ] `cumsum(a, axis?)` - Cumulative sum
- [ ] `cumprod(a, axis?)` - Cumulative product

### Statistics
- [ ] `mean(a, axis?, keepdims?)` - Mean
- [ ] `median(a, axis?, keepdims?)` - Median
- [ ] `std(a, axis?, ddof?, keepdims?)` - Standard deviation
- [ ] `var(a, axis?, ddof?, keepdims?)` - Variance
- [ ] `percentile(a, q, axis?)` - Percentile
- [ ] `quantile(a, q, axis?)` - Quantile

### Min/Max
- [ ] `min(a, axis?, keepdims?)` - Minimum
- [ ] `max(a, axis?, keepdims?)` - Maximum
- [ ] `argmin(a, axis?)` - Index of minimum
- [ ] `argmax(a, axis?)` - Index of maximum
- [ ] `ptp(a, axis?)` - Peak-to-peak (max - min)

### Logic
- [ ] `all(a, axis?, keepdims?)` - Test if all True
- [ ] `any(a, axis?, keepdims?)` - Test if any True

---

## Comparison

- [ ] `greater(x1, x2)` - Greater than
- [ ] `greater_equal(x1, x2)` - Greater or equal
- [ ] `less(x1, x2)` - Less than
- [ ] `less_equal(x1, x2)` - Less or equal
- [ ] `equal(x1, x2)` - Equal
- [ ] `not_equal(x1, x2)` - Not equal
- [ ] `allclose(a, b, rtol?, atol?)` - Close within tolerance
- [ ] `isclose(a, b, rtol?, atol?)` - Element-wise close

---

## Logic

- [ ] `logical_and(x1, x2)` - Logical AND
- [ ] `logical_or(x1, x2)` - Logical OR
- [ ] `logical_not(x)` - Logical NOT
- [ ] `logical_xor(x1, x2)` - Logical XOR

---

## Linear Algebra (numpy.linalg)

### Matrix Products
- [ ] `dot(a, b)` - Dot product
- [ ] `matmul(a, b)` - Matrix product
- [ ] `inner(a, b)` - Inner product
- [ ] `outer(a, b)` - Outer product
- [ ] `tensordot(a, b, axes)` - Tensor dot product
- [ ] `einsum(subscripts, *operands)` - Einstein summation

### Decompositions
- [ ] `linalg.cholesky(a)` - Cholesky decomposition
- [ ] `linalg.qr(a)` - QR decomposition
- [ ] `linalg.svd(a, full_matrices?)` - Singular value decomposition
- [ ] `linalg.eig(a)` - Eigenvalues and eigenvectors
- [ ] `linalg.eigh(a)` - Eigenvalues (Hermitian)
- [ ] `linalg.eigvals(a)` - Eigenvalues only

### Solving
- [ ] `linalg.solve(a, b)` - Solve linear system
- [ ] `linalg.lstsq(a, b)` - Least-squares solution
- [ ] `linalg.inv(a)` - Matrix inverse
- [ ] `linalg.pinv(a, rcond?)` - Pseudo-inverse

### Norms and Numbers
- [ ] `linalg.norm(x, ord?, axis?)` - Norm
- [ ] `linalg.det(a)` - Determinant
- [ ] `linalg.matrix_rank(a, tol?)` - Matrix rank
- [ ] `trace(a)` - Trace

---

## Random Sampling (numpy.random)

### Simple Random
- [ ] `random.rand(...shape)` - Uniform [0, 1)
- [ ] `random.randn(...shape)` - Standard normal
- [ ] `random.randint(low, high?, size?)` - Random integers
- [ ] `random.random(size?)` - Random floats [0, 1)

### Distributions
- [ ] `random.uniform(low, high, size?)` - Uniform distribution
- [ ] `random.normal(loc?, scale?, size?)` - Normal distribution
- [ ] `random.exponential(scale?, size?)` - Exponential
- [ ] `random.poisson(lam?, size?)` - Poisson
- [ ] `random.binomial(n, p, size?)` - Binomial

### Permutations
- [ ] `random.shuffle(x)` - Shuffle in-place
- [ ] `random.permutation(x)` - Permuted sequence
- [ ] `random.choice(a, size?, replace?)` - Random choice

### Generator (preferred API)
- [ ] `random.default_rng(seed?)` - Create generator
- [ ] `Generator.random(size?)` - Random floats
- [ ] `Generator.integers(low, high?, size?)` - Random integers
- [ ] `Generator.normal(loc?, scale?, size?)` - Normal distribution
- [ ] `Generator.standard_normal(size?)` - Standard normal

---

## Sorting and Searching

### Sorting
- [ ] `sort(a, axis?)` - Sort array
- [ ] `argsort(a, axis?)` - Indices that would sort
- [ ] `lexsort(keys)` - Indirect stable sort

### Searching
- [ ] `argmax(a, axis?)` - Index of maximum
- [ ] `argmin(a, axis?)` - Index of minimum
- [ ] `nonzero(a)` - Indices of non-zero elements
- [ ] `where(condition, x?, y?)` - Elements from x or y
- [ ] `searchsorted(a, v)` - Find indices to insert

### Counting
- [ ] `count_nonzero(a, axis?)` - Count non-zero elements
- [ ] `unique(a, return_index?, return_counts?)` - Unique elements

---

## Set Operations

- [ ] `unique(ar)` - Unique elements
- [ ] `in1d(ar1, ar2)` - Test membership
- [ ] `intersect1d(ar1, ar2)` - Intersection
- [ ] `union1d(ar1, ar2)` - Union
- [ ] `setdiff1d(ar1, ar2)` - Set difference
- [ ] `setxor1d(ar1, ar2)` - Symmetric difference

---

## I/O

### NumPy Files
- [ ] `load(file)` - Load array from .npy file
- [ ] `save(file, arr)` - Save array to .npy file
- [ ] `savez(file, *arrays, **kwds)` - Save multiple arrays (.npz)
- [ ] `savez_compressed(file, *arrays, **kwds)` - Compressed .npz

### Text Files
- [ ] `loadtxt(fname, dtype?)` - Load from text
- [ ] `savetxt(fname, X)` - Save to text

---

## FFT

### Standard FFT
- [ ] `fft.fft(a, n?, axis?)` - 1-D FFT
- [ ] `fft.ifft(a, n?, axis?)` - 1-D inverse FFT
- [ ] `fft.fft2(a, s?, axes?)` - 2-D FFT
- [ ] `fft.ifft2(a, s?, axes?)` - 2-D inverse FFT
- [ ] `fft.fftn(a, s?, axes?)` - N-D FFT
- [ ] `fft.ifftn(a, s?, axes?)` - N-D inverse FFT

### Real FFT
- [ ] `fft.rfft(a, n?, axis?)` - Real input FFT
- [ ] `fft.irfft(a, n?, axis?)` - Inverse real FFT

### Helpers
- [ ] `fft.fftfreq(n, d?)` - FFT frequencies
- [ ] `fft.rfftfreq(n, d?)` - Real FFT frequencies
- [ ] `fft.fftshift(x, axes?)` - Shift zero-frequency to center
- [ ] `fft.ifftshift(x, axes?)` - Inverse of fftshift

---

## Polynomials

### Polynomial Class
- [ ] `polynomial.Polynomial(coef)` - Power series
- [ ] `Polynomial.fit(x, y, deg)` - Least-squares fit
- [ ] `Polynomial.roots()` - Roots
- [ ] `Polynomial.deriv()` - Derivative
- [ ] `Polynomial.integ()` - Integral

### Legacy (numpy.poly1d)
- [ ] `poly1d(coef)` - 1-D polynomial
- [ ] `polyval(p, x)` - Evaluate polynomial
- [ ] `polyfit(x, y, deg)` - Polynomial fit
- [ ] `polyder(p)` - Derivative
- [ ] `polyint(p)` - Integral
- [ ] `polyadd(a1, a2)` - Add polynomials
- [ ] `polymul(a1, a2)` - Multiply polynomials

---

## Advanced

### Broadcasting
- [ ] `broadcast_to(array, shape)` - Broadcast to shape
- [ ] `broadcast_arrays(*args)` - Broadcast multiple arrays

### Indexing
- [ ] `take(a, indices, axis?)` - Take elements
- [ ] `put(a, ind, v)` - Put values at indices
- [ ] `choose(a, choices)` - Construct from index array

### Testing
- [ ] `allclose(a, b, rtol?, atol?)` - Arrays close
- [ ] `isclose(a, b, rtol?, atol?)` - Element-wise close
- [ ] `array_equal(a1, a2)` - Arrays equal

---

## Progress Summary

**Total Functions**: ~350 core functions

### Priority Tiers

**Tier 1 - Essential** (~50 functions):
- Array creation: zeros, ones, array, arange
- Basic math: add, multiply, matmul, dot
- Slicing and indexing
- Broadcasting
- Basic reductions: sum, mean, max, min

**Tier 2 - Important** (~100 functions):
- Linear algebra: solve, inv, svd, eig
- More math: sin, cos, exp, log
- Full reductions: std, var, median
- Comparison operators
- Reshaping operations

**Tier 3 - Extended** (~200 functions):
- Random number generation
- FFT operations
- Polynomials
- I/O operations
- Set operations
- Advanced indexing

---

**Last Updated**: 2025-10-07
