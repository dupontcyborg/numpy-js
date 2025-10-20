/**
 * Linear algebra operations
 *
 * Pure functions for matrix operations (matmul, etc.).
 * @module ops/linalg
 */

import dgemm from '@stdlib/blas/base/dgemm';
import { ArrayStorage } from '../core/storage';
import { promoteDTypes } from '../core/dtype';

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
    throw new Error(
      `matmul shape mismatch: (${m},${k}) @ (${k2},${n})`
    );
  }

  // Determine result dtype (promote inputs, but use float64 for integer types)
  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const computeDtype = resultDtype.startsWith('int') || resultDtype.startsWith('uint') || resultDtype === 'bool'
    ? 'float64'
    : resultDtype;

  // For now, we only support float64 matmul (using dgemm)
  // TODO: Add float32 support using sgemm
  if (computeDtype !== 'float64') {
    throw new Error(`matmul currently only supports float64, got ${computeDtype}`);
  }

  // Convert inputs to Float64Array if needed
  const aData = a.dtype === 'float64' ? (a.data as Float64Array) : Float64Array.from(a.data as any);
  const bData = b.dtype === 'float64' ? (b.data as Float64Array) : Float64Array.from(b.data as any);

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
