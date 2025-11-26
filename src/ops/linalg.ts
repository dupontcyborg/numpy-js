/**
 * Linear algebra operations
 *
 * Pure functions for matrix operations (matmul, etc.).
 * @module ops/linalg
 */

import { ArrayStorage } from '../core/storage';
import { promoteDTypes } from '../core/dtype';

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
 * Matrix multiplication
 * Requires 2D arrays with compatible shapes
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
  const aData =
    a.dtype === 'float64'
      ? (a.data as Float64Array)
      : Float64Array.from(Array.from(a.data as ArrayLike<number>).map(Number));
  const bData =
    b.dtype === 'float64'
      ? (b.data as Float64Array)
      : Float64Array.from(Array.from(b.data as ArrayLike<number>).map(Number));

  // Create result array
  const result = ArrayStorage.zeros([m, n], 'float64');

  // Use @stdlib dgemm (double-precision general matrix multiply)
  dgemm(
    'row-major',
    'no-transpose',
    'no-transpose',
    m,
    n,
    k,
    1.0, // alpha
    aData,
    k, // lda (leading dimension of a)
    bData,
    n, // ldb (leading dimension of b)
    0.0, // beta
    result.data as Float64Array,
    n // ldc (leading dimension of c/result)
  );

  return result;
}
