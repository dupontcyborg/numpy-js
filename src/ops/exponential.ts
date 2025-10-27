/**
 * Exponential, logarithmic, and power operations
 *
 * Pure functions for element-wise exponential operations:
 * exp, log, sqrt, power, etc.
 *
 * These functions are used by NDArray methods but are separated
 * to keep the codebase modular and testable.
 */

import { ArrayStorage } from '../core/storage';
import { elementwiseUnaryOp, elementwiseBinaryOp } from '../internal/compute';
import { isBigIntDType } from '../core/dtype';

/**
 * Square root of each element
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with sqrt applied
 */
export function sqrt(a: ArrayStorage): ArrayStorage {
  return elementwiseUnaryOp(a, Math.sqrt, false); // false = promote integers to float64
}

/**
 * Raise elements to power
 * NumPy behavior: Promotes to float64 for integer types with non-integer exponents
 *
 * @param a - Base array storage
 * @param b - Exponent (array storage or scalar)
 * @returns Result storage with power applied
 */
export function power(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return powerScalar(a, b);
  }
  return elementwiseBinaryOp(a, b, Math.pow, 'power');
}

/**
 * Power with scalar exponent (optimized path)
 * @private
 */
function powerScalar(storage: ArrayStorage, exponent: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // NumPy behavior: integer ** integer stays integer if exponent >= 0
  // integer ** negative or float exponent promotes to float64
  const isIntegerType = dtype !== 'float32' && dtype !== 'float64';
  const needsFloatPromotion = isIntegerType && (exponent < 0 || !Number.isInteger(exponent));
  const resultDtype = needsFloatPromotion ? 'float64' : dtype;

  // Create result with appropriate dtype
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic
    if (isBigIntDType(resultDtype) && Number.isInteger(exponent) && exponent >= 0) {
      // BigInt ** positive integer stays BigInt
      const thisTyped = data as BigInt64Array | BigUint64Array;
      const resultTyped = resultData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = thisTyped[i]! ** BigInt(exponent);
      }
    } else {
      // BigInt ** negative or float promotes to float64
      for (let i = 0; i < size; i++) {
        resultData[i] = Math.pow(Number(data[i]!), exponent);
      }
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.pow(Number(data[i]!), exponent);
    }
  }

  return result;
}
