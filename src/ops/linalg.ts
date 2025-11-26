/**
 * Linear algebra operations
 *
 * Pure functions for matrix operations (matmul, etc.).
 * @module ops/linalg
 */

import { ArrayStorage } from '../core/storage';
import { promoteDTypes } from '../core/dtype';
import * as shapeOps from './shape';

/**
 * BLAS-like types for matrix operations
 */
type Layout = 'row-major' | 'column-major';
type Transpose = 'no-transpose' | 'transpose';

/**
 * Double-precision general matrix multiply (DGEMM)
 *
 * Full BLAS-compatible implementation without external dependencies.
 * Performs: C = alpha * op(A) * op(B) + beta * C
 *
 * Supports all combinations of:
 * - Row-major and column-major layouts
 * - Transpose and no-transpose operations
 * - Arbitrary alpha and beta scalars
 *
 * Uses specialized loops for each case to avoid function call overhead.
 *
 * @internal
 */
function dgemm(
  layout: Layout,
  transA: Transpose,
  transB: Transpose,
  M: number, // rows of op(A) and C
  N: number, // cols of op(B) and C
  K: number, // cols of op(A) and rows of op(B)
  alpha: number, // scalar alpha
  A: Float64Array, // matrix A
  lda: number, // leading dimension of A
  B: Float64Array, // matrix B
  ldb: number, // leading dimension of B
  beta: number, // scalar beta
  C: Float64Array, // matrix C (output)
  ldc: number // leading dimension of C
): void {
  // Apply beta scaling to C first
  if (beta === 0) {
    for (let i = 0; i < M * N; i++) {
      C[i] = 0;
    }
  } else if (beta !== 1) {
    for (let i = 0; i < M * N; i++) {
      C[i] = (C[i] ?? 0) * beta;
    }
  }

  // Select specialized loop based on layout and transpose modes
  // This avoids function call overhead in the hot loop
  const isRowMajor = layout === 'row-major';
  const transposeA = transA === 'transpose';
  const transposeB = transB === 'transpose';

  if (isRowMajor && !transposeA && !transposeB) {
    // Row-major, no transpose (most common case)
    // C[i,j] = sum_k A[i,k] * B[k,j]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[i * lda + k] ?? 0) * (B[k * ldb + j] ?? 0);
        }
        C[i * ldc + j] = (C[i * ldc + j] ?? 0) + alpha * sum;
      }
    }
  } else if (isRowMajor && transposeA && !transposeB) {
    // Row-major, A transposed
    // C[i,j] = sum_k A[k,i] * B[k,j]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[k * lda + i] ?? 0) * (B[k * ldb + j] ?? 0);
        }
        C[i * ldc + j] = (C[i * ldc + j] ?? 0) + alpha * sum;
      }
    }
  } else if (isRowMajor && !transposeA && transposeB) {
    // Row-major, B transposed
    // C[i,j] = sum_k A[i,k] * B[j,k]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[i * lda + k] ?? 0) * (B[j * ldb + k] ?? 0);
        }
        C[i * ldc + j] = (C[i * ldc + j] ?? 0) + alpha * sum;
      }
    }
  } else if (isRowMajor && transposeA && transposeB) {
    // Row-major, both transposed
    // C[i,j] = sum_k A[k,i] * B[j,k]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[k * lda + i] ?? 0) * (B[j * ldb + k] ?? 0);
        }
        C[i * ldc + j] = (C[i * ldc + j] ?? 0) + alpha * sum;
      }
    }
  } else if (!isRowMajor && !transposeA && !transposeB) {
    // Column-major, no transpose
    // C[i,j] = sum_k A[i,k] * B[k,j]
    // Column-major: A[i,k] = A[k*lda + i], C[i,j] = C[j*ldc + i]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[k * lda + i] ?? 0) * (B[j * ldb + k] ?? 0);
        }
        C[j * ldc + i] = (C[j * ldc + i] ?? 0) + alpha * sum;
      }
    }
  } else if (!isRowMajor && transposeA && !transposeB) {
    // Column-major, A transposed
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[i * lda + k] ?? 0) * (B[j * ldb + k] ?? 0);
        }
        C[j * ldc + i] = (C[j * ldc + i] ?? 0) + alpha * sum;
      }
    }
  } else if (!isRowMajor && !transposeA && transposeB) {
    // Column-major, B transposed
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[k * lda + i] ?? 0) * (B[k * ldb + j] ?? 0);
        }
        C[j * ldc + i] = (C[j * ldc + i] ?? 0) + alpha * sum;
      }
    }
  } else {
    // Column-major, both transposed
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[i * lda + k] ?? 0) * (B[k * ldb + j] ?? 0);
        }
        C[j * ldc + i] = (C[j * ldc + i] ?? 0) + alpha * sum;
      }
    }
  }
}

/**
 * Dot product of two arrays (fully NumPy-compatible)
 *
 * Behavior depends on input dimensions:
 * - 0D · 0D: Multiply scalars → scalar
 * - 0D · ND or ND · 0D: Element-wise multiply → ND
 * - 1D · 1D: Inner product → scalar
 * - 2D · 2D: Matrix multiplication → 2D
 * - 2D · 1D: Matrix-vector product → 1D
 * - 1D · 2D: Vector-matrix product → 1D
 * - ND · 1D (N≥2): Sum product over last axis of a → (N-1)D
 * - 1D · ND (N≥2): Sum product over first axis of b → (N-1)D
 * - ND · MD (N,M≥2): Sum product over last axis of a and second-to-last of b → (N+M-2)D
 *
 * For 2D·2D, prefer using matmul() instead.
 */
export function dot(a: ArrayStorage, b: ArrayStorage): ArrayStorage | number | bigint {
  const aDim = a.ndim;
  const bDim = b.ndim;

  // Case 0: Scalar (0D) cases - treat as multiplication
  if (aDim === 0 || bDim === 0) {
    // Get scalar values
    const aVal = aDim === 0 ? a.get() : null;
    const bVal = bDim === 0 ? b.get() : null;

    if (aDim === 0 && bDim === 0) {
      // Both scalars: multiply them
      if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
        return aVal * bVal;
      }
      return Number(aVal) * Number(bVal);
    } else if (aDim === 0) {
      // a is scalar, b is array: scalar * array (element-wise)
      // Equivalent to multiply(a, b)
      const resultDtype = promoteDTypes(a.dtype, b.dtype);
      const result = ArrayStorage.zeros([...b.shape], resultDtype);
      for (let i = 0; i < b.size; i++) {
        const bData = b.data[i + b.offset];
        if (typeof aVal === 'bigint' && typeof bData === 'bigint') {
          result.data[i] = aVal * bData;
        } else {
          result.data[i] = Number(aVal) * Number(bData);
        }
      }
      return result;
    } else {
      // b is scalar, a is array: array * scalar (element-wise)
      const resultDtype = promoteDTypes(a.dtype, b.dtype);
      const result = ArrayStorage.zeros([...a.shape], resultDtype);
      for (let i = 0; i < a.size; i++) {
        const aData = a.data[i + a.offset];
        if (typeof aData === 'bigint' && typeof bVal === 'bigint') {
          result.data[i] = aData * bVal;
        } else {
          result.data[i] = Number(aData) * Number(bVal);
        }
      }
      return result;
    }
  }

  // Case 1: Both 1D -> scalar (inner product)
  if (aDim === 1 && bDim === 1) {
    if (a.shape[0] !== b.shape[0]) {
      throw new Error(`dot: incompatible shapes (${a.shape[0]},) and (${b.shape[0]},)`);
    }
    const n = a.shape[0]!;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const aVal = a.get(i);
      const bVal = b.get(i);
      // Handle BigInt
      if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
        sum = Number(sum) + Number(aVal * bVal);
      } else {
        sum += Number(aVal) * Number(bVal);
      }
    }
    return sum;
  }

  // Case 2: Both 2D -> matrix multiplication (delegate to matmul)
  if (aDim === 2 && bDim === 2) {
    return matmul(a, b);
  }

  // Case 3: 2D · 1D -> matrix-vector product (returns 1D)
  if (aDim === 2 && bDim === 1) {
    const [m, k] = a.shape;
    const n = b.shape[0]!;
    if (k !== n) {
      throw new Error(`dot: incompatible shapes (${m},${k}) and (${n},)`);
    }

    const resultDtype = promoteDTypes(a.dtype, b.dtype);
    const result = ArrayStorage.zeros([m!], resultDtype);

    for (let i = 0; i < m!; i++) {
      let sum = 0;
      for (let j = 0; j < k!; j++) {
        const aVal = a.get(i, j);
        const bVal = b.get(j);
        if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
          sum = Number(sum) + Number(aVal * bVal);
        } else {
          sum += Number(aVal) * Number(bVal);
        }
      }
      result.set([i], sum);
    }

    return result;
  }

  // Case 4: 1D · 2D -> vector-matrix product (returns 1D)
  if (aDim === 1 && bDim === 2) {
    const m = a.shape[0]!;
    const [k, n] = b.shape;
    if (m !== k) {
      throw new Error(`dot: incompatible shapes (${m},) and (${k},${n})`);
    }

    const resultDtype = promoteDTypes(a.dtype, b.dtype);
    const result = ArrayStorage.zeros([n!], resultDtype);

    for (let j = 0; j < n!; j++) {
      let sum = 0;
      for (let i = 0; i < m; i++) {
        const aVal = a.get(i);
        const bVal = b.get(i, j);
        if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
          sum = Number(sum) + Number(aVal * bVal);
        } else {
          sum += Number(aVal) * Number(bVal);
        }
      }
      result.set([j], sum);
    }

    return result;
  }

  // Case 5: ND · 1D (N > 2) -> sum product over last axis, result is (N-1)D
  if (aDim > 2 && bDim === 1) {
    const lastDimA = a.shape[aDim - 1]!;
    const bSize = b.shape[0]!;
    if (lastDimA !== bSize) {
      throw new Error(`dot: incompatible shapes ${JSON.stringify(a.shape)} and (${bSize},)`);
    }

    // Result shape is a.shape[:-1]
    const resultShape = [...a.shape.slice(0, -1)];
    const resultDtype = promoteDTypes(a.dtype, b.dtype);
    const result = ArrayStorage.zeros(resultShape, resultDtype);

    // Iterate over all positions in result
    const resultSize = resultShape.reduce((acc, dim) => acc * dim, 1);
    for (let i = 0; i < resultSize; i++) {
      let sum = 0;
      // Compute multi-dimensional index for result
      let temp = i;
      const resultIdx: number[] = [];
      for (let d = resultShape.length - 1; d >= 0; d--) {
        resultIdx[d] = temp % resultShape[d]!;
        temp = Math.floor(temp / resultShape[d]!);
      }

      // Sum over the last dimension of a
      for (let k = 0; k < lastDimA; k++) {
        const aIdx = [...resultIdx, k];
        const aVal = a.get(...aIdx);
        const bVal = b.get(k);
        if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
          sum = Number(sum) + Number(aVal * bVal);
        } else {
          sum += Number(aVal) * Number(bVal);
        }
      }
      result.set(resultIdx, sum);
    }

    return result;
  }

  // Case 6: 1D · ND (N > 2) -> sum product over SECOND axis of b, result is (b.shape[0], b.shape[2:])
  // Actually for 1D·3D: sum over axis 1 of b
  // For general case: need to handle this more carefully
  if (aDim === 1 && bDim > 2) {
    const aSize = a.shape[0]!;
    // For 1D (size K) · ND, we contract over axis 1 of b (which should have size K)
    const contractAxisB = 1;
    const contractDimB = b.shape[contractAxisB]!;

    if (aSize !== contractDimB) {
      throw new Error(`dot: incompatible shapes (${aSize},) and ${JSON.stringify(b.shape)}`);
    }

    // Result shape: b.shape[0:1] + b.shape[2:]
    // For (K,) · (L, K, M, N, ...) -> (L, M, N, ...)
    const resultShape = [...b.shape.slice(0, contractAxisB), ...b.shape.slice(contractAxisB + 1)];
    const resultDtype = promoteDTypes(a.dtype, b.dtype);
    const result = ArrayStorage.zeros(resultShape, resultDtype);

    // Compute using multi-dimensional indices
    const resultSize = resultShape.reduce((acc, dim) => acc * dim, 1);
    for (let i = 0; i < resultSize; i++) {
      // Convert flat index to multi-dim result index
      let temp = i;
      const resultIdx: number[] = [];
      for (let d = resultShape.length - 1; d >= 0; d--) {
        resultIdx[d] = temp % resultShape[d]!;
        temp = Math.floor(temp / resultShape[d]!);
      }

      // Build b index by inserting the contract dimension
      // result[i,j,...] corresponds to b[i, :, j, ...]
      const bIdxBefore = resultIdx.slice(0, contractAxisB);
      const bIdxAfter = resultIdx.slice(contractAxisB);

      let sum = 0;
      for (let k = 0; k < aSize; k++) {
        const aVal = a.get(k);
        const bIdx = [...bIdxBefore, k, ...bIdxAfter];
        const bVal = b.get(...bIdx);
        if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
          sum = Number(sum) + Number(aVal * bVal);
        } else {
          sum += Number(aVal) * Number(bVal);
        }
      }
      result.set(resultIdx, sum);
    }

    return result;
  }

  // Case 7: ND · MD (N,M ≥ 2, not both 2) -> general tensor contraction
  // Result shape: a.shape[:-1] + b.shape[:-2] + b.shape[-1:]
  if (aDim >= 2 && bDim >= 2 && !(aDim === 2 && bDim === 2)) {
    const lastDimA = a.shape[aDim - 1]!;
    const secondLastDimB = b.shape[bDim - 2]!;

    if (lastDimA !== secondLastDimB) {
      throw new Error(
        `dot: incompatible shapes ${JSON.stringify(a.shape)} and ${JSON.stringify(b.shape)}`
      );
    }

    // Build result shape
    const resultShape = [...a.shape.slice(0, -1), ...b.shape.slice(0, -2), b.shape[bDim - 1]!];
    const resultDtype = promoteDTypes(a.dtype, b.dtype);
    const result = ArrayStorage.zeros(resultShape, resultDtype);

    const aOuterSize = a.shape.slice(0, -1).reduce((acc, dim) => acc * dim, 1);
    const bOuterSize = b.shape.slice(0, -2).reduce((acc, dim) => acc * dim, 1);
    const bLastDim = b.shape[bDim - 1]!;
    const contractionDim = lastDimA;

    // Iterate: result[i, j, k] = sum_m a[i, m] * b[j, m, k]
    for (let i = 0; i < aOuterSize; i++) {
      for (let j = 0; j < bOuterSize; j++) {
        for (let k = 0; k < bLastDim; k++) {
          let sum = 0;
          for (let m = 0; m < contractionDim; m++) {
            // Get a[i, m] - need to convert flat index i to multi-dim
            const aIdx = i * contractionDim + m;
            const aVal = a.data[aIdx + a.offset];

            // Get b[j, m, k] - need multi-dim indexing
            const bIdx = j * contractionDim * bLastDim + m * bLastDim + k;
            const bVal = b.data[bIdx + b.offset];

            if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
              sum = Number(sum) + Number(aVal * bVal);
            } else {
              sum += Number(aVal) * Number(bVal);
            }
          }

          // Set result at the appropriate position
          const resultIdx = i * bOuterSize * bLastDim + j * bLastDim + k;
          result.data[resultIdx] = sum;
        }
      }
    }

    return result;
  }

  // Should never reach here - all cases covered
  throw new Error(`dot: unexpected combination of dimensions ${aDim}D · ${bDim}D`);
}

/**
 * Matrix multiplication
 * Requires 2D arrays with compatible shapes
 *
 * Automatically detects transposed/non-contiguous arrays via strides
 * and uses appropriate DGEMM transpose parameters.
 *
 * Note: Currently uses float64 precision for all operations.
 * Integer inputs are promoted to float64 (matching NumPy behavior).
 */
export function matmul(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  if (a.ndim !== 2 || b.ndim !== 2) {
    throw new Error('matmul requires 2D arrays');
  }

  const [m = 0, k = 0] = a.shape;
  const [k2 = 0, n = 0] = b.shape;

  if (k !== k2) {
    throw new Error(`matmul shape mismatch: (${m},${k}) @ (${k2},${n})`);
  }

  // Determine result dtype (promote inputs, but use float64 for integer types)
  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const computeDtype =
    resultDtype.startsWith('int') || resultDtype.startsWith('uint') || resultDtype === 'bool'
      ? 'float64'
      : resultDtype;

  // For now, we only support float64 matmul (using dgemm)
  // TODO: Add float32 support using sgemm
  if (computeDtype !== 'float64') {
    throw new Error(`matmul currently only supports float64, got ${computeDtype}`);
  }

  // Convert inputs to Float64Array if needed
  let aData =
    a.dtype === 'float64'
      ? (a.data as Float64Array)
      : Float64Array.from(Array.from(a.data as ArrayLike<number>).map(Number));
  let bData =
    b.dtype === 'float64'
      ? (b.data as Float64Array)
      : Float64Array.from(Array.from(b.data as ArrayLike<number>).map(Number));

  // Handle offset for sliced arrays (views)
  // If the array has an offset, we need to pass the subarray starting from that offset
  if (a.offset > 0) {
    aData = aData.subarray(a.offset) as Float64Array;
  }
  if (b.offset > 0) {
    bData = bData.subarray(b.offset) as Float64Array;
  }

  // Detect array layout from strides
  // Row-major (C-contiguous): row stride > col stride
  // Transposed (F-contiguous or transposed view): col stride > row stride
  const [aStrideRow = 0, aStrideCol = 0] = a.strides;
  const [bStrideRow = 0, bStrideCol = 0] = b.strides;

  // Determine if arrays are effectively transposed
  // For a normal MxK array: strides are [K, 1] (row stride = K cols)
  // For a transposed KxM array (viewed as MxK): strides are [1, M] (col stride > row stride)
  const aIsTransposed = aStrideCol > aStrideRow;
  const bIsTransposed = bStrideCol > bStrideRow;

  const transA: Transpose = aIsTransposed ? 'transpose' : 'no-transpose';
  const transB: Transpose = bIsTransposed ? 'transpose' : 'no-transpose';

  // Determine leading dimensions based on memory layout
  // Leading dimension is the stride of the major dimension in memory
  let lda: number;
  let ldb: number;

  if (aIsTransposed) {
    // Array is stored with columns contiguous (F-order or transposed)
    // The leading dimension is how many elements to skip between columns
    lda = aStrideCol;
  } else {
    // Array is row-major (C-order)
    // The leading dimension is the row stride (number of elements per row)
    lda = aStrideRow;
  }

  if (bIsTransposed) {
    ldb = bStrideCol;
  } else {
    ldb = bStrideRow;
  }

  // Create result array (always row-major)
  const result = ArrayStorage.zeros([m, n], 'float64');

  // Call dgemm with detected transpose flags and leading dimensions
  dgemm(
    'row-major',
    transA,
    transB,
    m,
    n,
    k,
    1.0, // alpha
    aData,
    lda, // leading dimension of a (accounts for actual memory layout)
    bData,
    ldb, // leading dimension of b (accounts for actual memory layout)
    0.0, // beta
    result.data as Float64Array,
    n // ldc (result is always row-major with n cols)
  );

  return result;
}

/**
 * Sum along the diagonal of a 2D array
 *
 * Computes the trace (sum of diagonal elements) of a matrix.
 * For non-square matrices, sums along the diagonal up to min(rows, cols).
 *
 * @param a - Input 2D array
 * @returns Sum of diagonal elements
 */
export function trace(a: ArrayStorage): number | bigint {
  if (a.ndim !== 2) {
    throw new Error(`trace requires 2D array, got ${a.ndim}D`);
  }

  const [rows = 0, cols = 0] = a.shape;
  const diagLen = Math.min(rows, cols);

  let sum: number | bigint = 0;

  for (let i = 0; i < diagLen; i++) {
    const val = a.get(i, i);
    if (typeof val === 'bigint') {
      sum = (typeof sum === 'bigint' ? sum : BigInt(sum)) + val;
    } else {
      sum = (typeof sum === 'bigint' ? Number(sum) : sum) + val;
    }
  }

  return sum;
}

/**
 * Permute the dimensions of an array
 *
 * Standalone version of NDArray.transpose() method.
 * Returns a view with axes permuted.
 *
 * @param a - Input array
 * @param axes - Optional permutation of axes (defaults to reverse order)
 * @returns Transposed view
 */
export function transpose(a: ArrayStorage, axes?: number[]): ArrayStorage {
  return shapeOps.transpose(a, axes);
}

/**
 * Inner product of two arrays
 *
 * Computes sum product over the LAST axes of both a and b.
 * - 1D · 1D: Same as dot (ordinary inner product) → scalar
 * - ND · MD: Contracts last dimension of each → (*a.shape[:-1], *b.shape[:-1])
 *
 * Different from dot: always uses last axis of BOTH arrays.
 *
 * @param a - First array
 * @param b - Second array
 * @returns Inner product result
 */
export function inner(a: ArrayStorage, b: ArrayStorage): ArrayStorage | number | bigint {
  const aDim = a.ndim;
  const bDim = b.ndim;

  // Last dimensions must match
  const aLastDim = a.shape[aDim - 1]!;
  const bLastDim = b.shape[bDim - 1]!;

  if (aLastDim !== bLastDim) {
    throw new Error(
      `inner: incompatible shapes - last dimensions ${aLastDim} and ${bLastDim} don't match`
    );
  }

  // Special case: both 1D -> scalar
  if (aDim === 1 && bDim === 1) {
    return dot(a, b) as number;
  }

  // General case: result shape is a.shape[:-1] + b.shape[:-1]
  const resultShape = [...a.shape.slice(0, -1), ...b.shape.slice(0, -1)];
  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros(resultShape, resultDtype);

  const aOuterSize = aDim === 1 ? 1 : a.shape.slice(0, -1).reduce((acc, dim) => acc * dim, 1);
  const bOuterSize = bDim === 1 ? 1 : b.shape.slice(0, -1).reduce((acc, dim) => acc * dim, 1);
  const contractionDim = aLastDim;

  // Compute: result[i, j] = sum_k a[i, k] * b[j, k]
  for (let i = 0; i < aOuterSize; i++) {
    for (let j = 0; j < bOuterSize; j++) {
      let sum = 0;
      for (let k = 0; k < contractionDim; k++) {
        // Get a[i, k] and b[j, k]
        const aIdx = aDim === 1 ? k : i * contractionDim + k;
        const bIdx = bDim === 1 ? k : j * contractionDim + k;
        const aVal = a.data[aIdx + a.offset];
        const bVal = b.data[bIdx + b.offset];

        if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
          sum = Number(sum) + Number(aVal * bVal);
        } else {
          sum += Number(aVal) * Number(bVal);
        }
      }

      // Set result
      if (resultShape.length === 0) {
        // Scalar result
        return sum;
      }
      const resultIdx = aOuterSize === 1 ? j : i * bOuterSize + j;
      result.data[resultIdx] = sum;
    }
  }

  return result;
}

/**
 * Outer product of two vectors
 *
 * Computes out[i, j] = a[i] * b[j]
 * Input arrays are flattened if not 1D.
 *
 * @param a - First input (flattened to 1D)
 * @param b - Second input (flattened to 1D)
 * @returns 2D array of shape (a.size, b.size)
 */
export function outer(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  // Flatten inputs to 1D
  const aFlat = a.ndim === 1 ? a : shapeOps.ravel(a);
  const bFlat = b.ndim === 1 ? b : shapeOps.ravel(b);

  const m = aFlat.size;
  const n = bFlat.size;

  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros([m, n], resultDtype);

  // Compute outer product: result[i,j] = a[i] * b[j]
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const aVal = aFlat.get(i);
      const bVal = bFlat.get(j);

      let product;
      if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
        product = aVal * bVal;
      } else {
        product = Number(aVal) * Number(bVal);
      }

      result.set([i, j], product);
    }
  }

  return result;
}

/**
 * Tensor dot product along specified axes
 *
 * Computes sum product over specified axes.
 *
 * @param a - First array
 * @param b - Second array
 * @param axes - Axes to contract:
 *   - Integer N: Contract last N axes of a with first N of b
 *   - [a_axes, b_axes]: Contract specified axes
 * @returns Tensor dot product
 */
export function tensordot(
  a: ArrayStorage,
  b: ArrayStorage,
  axes: number | [number[], number[]]
): ArrayStorage | number | bigint {
  let aAxes: number[];
  let bAxes: number[];

  if (typeof axes === 'number') {
    // Contract last N axes of a with first N of b
    const N = axes;
    if (N < 0) {
      throw new Error('tensordot: axes must be non-negative');
    }
    if (N > a.ndim || N > b.ndim) {
      throw new Error('tensordot: axes exceeds array dimensions');
    }

    // Last N axes of a
    aAxes = Array.from({ length: N }, (_, i) => a.ndim - N + i);
    // First N axes of b
    bAxes = Array.from({ length: N }, (_, i) => i);
  } else {
    [aAxes, bAxes] = axes;
    if (aAxes.length !== bAxes.length) {
      throw new Error('tensordot: axes lists must have same length');
    }
  }

  // Validate axes and check dimension compatibility
  for (let i = 0; i < aAxes.length; i++) {
    const aAxis = aAxes[i]!;
    const bAxis = bAxes[i]!;
    if (aAxis < 0 || aAxis >= a.ndim || bAxis < 0 || bAxis >= b.ndim) {
      throw new Error('tensordot: axis out of bounds');
    }
    if (a.shape[aAxis] !== b.shape[bAxis]) {
      throw new Error(
        `tensordot: shape mismatch on axes ${aAxis} and ${bAxis}: ${a.shape[aAxis]} != ${b.shape[bAxis]}`
      );
    }
  }

  // Separate axes into contracted and free axes
  const aFreeAxes: number[] = [];
  const bFreeAxes: number[] = [];

  for (let i = 0; i < a.ndim; i++) {
    if (!aAxes.includes(i)) {
      aFreeAxes.push(i);
    }
  }
  for (let i = 0; i < b.ndim; i++) {
    if (!bAxes.includes(i)) {
      bFreeAxes.push(i);
    }
  }

  // Build result shape: free axes of a + free axes of b
  const resultShape = [
    ...aFreeAxes.map((ax) => a.shape[ax]!),
    ...bFreeAxes.map((ax) => b.shape[ax]!),
  ];

  // Special case: no free axes (full contraction) -> scalar result
  if (resultShape.length === 0) {
    let sum = 0;
    // Iterate over all combinations of contracted axes
    const contractSize = aAxes.map((ax) => a.shape[ax]!).reduce((acc, dim) => acc * dim, 1);

    for (let i = 0; i < contractSize; i++) {
      // Convert flat index to contracted indices
      let temp = i;
      const contractedIdx: number[] = new Array(aAxes.length);
      for (let j = aAxes.length - 1; j >= 0; j--) {
        const ax = aAxes[j]!;
        contractedIdx[j] = temp % a.shape[ax]!;
        temp = Math.floor(temp / a.shape[ax]!);
      }

      // Build full indices for a and b
      const aIdx: number[] = new Array(a.ndim);
      const bIdx: number[] = new Array(b.ndim);

      for (let j = 0; j < aAxes.length; j++) {
        aIdx[aAxes[j]!] = contractedIdx[j]!;
      }
      for (let j = 0; j < bAxes.length; j++) {
        bIdx[bAxes[j]!] = contractedIdx[j]!;
      }

      const aVal = a.get(...aIdx);
      const bVal = b.get(...bIdx);

      if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
        sum = Number(sum) + Number(aVal * bVal);
      } else {
        sum += Number(aVal) * Number(bVal);
      }
    }
    return sum;
  }

  // General case: with free axes
  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros(resultShape, resultDtype);

  const resultSize = resultShape.reduce((acc, dim) => acc * dim, 1);
  const contractSize = aAxes.map((ax) => a.shape[ax]!).reduce((acc, dim) => acc * dim, 1);

  // Iterate over all result positions
  for (let resIdx = 0; resIdx < resultSize; resIdx++) {
    // Convert flat result index to multi-dimensional
    let temp = resIdx;
    const resultIndices: number[] = [];
    for (let i = resultShape.length - 1; i >= 0; i--) {
      resultIndices[i] = temp % resultShape[i]!;
      temp = Math.floor(temp / resultShape[i]!);
    }

    // Extract indices for a's free axes and b's free axes
    const aFreeIndices = resultIndices.slice(0, aFreeAxes.length);
    const bFreeIndices = resultIndices.slice(aFreeAxes.length);

    let sum = 0;

    // Sum over all contracted axes
    for (let c = 0; c < contractSize; c++) {
      // Convert flat contracted index to multi-dimensional
      temp = c;
      const contractedIndices: number[] = [];
      for (let i = aAxes.length - 1; i >= 0; i--) {
        const ax = aAxes[i]!;
        contractedIndices[i] = temp % a.shape[ax]!;
        temp = Math.floor(temp / a.shape[ax]!);
      }

      // Build full indices for a and b
      const aFullIdx: number[] = new Array(a.ndim);
      const bFullIdx: number[] = new Array(b.ndim);

      // Fill in free axes
      for (let i = 0; i < aFreeAxes.length; i++) {
        aFullIdx[aFreeAxes[i]!] = aFreeIndices[i]!;
      }
      for (let i = 0; i < bFreeAxes.length; i++) {
        bFullIdx[bFreeAxes[i]!] = bFreeIndices[i]!;
      }

      // Fill in contracted axes
      for (let i = 0; i < aAxes.length; i++) {
        aFullIdx[aAxes[i]!] = contractedIndices[i]!;
        bFullIdx[bAxes[i]!] = contractedIndices[i]!;
      }

      const aVal = a.get(...aFullIdx);
      const bVal = b.get(...bFullIdx);

      if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
        sum = Number(sum) + Number(aVal * bVal);
      } else {
        sum += Number(aVal) * Number(bVal);
      }
    }

    result.set(resultIndices, sum);
  }

  return result;
}
