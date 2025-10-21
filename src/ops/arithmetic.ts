/**
 * Arithmetic operations
 *
 * Pure functions for element-wise arithmetic operations:
 * add, subtract, multiply, divide
 *
 * These functions are used by NDArray methods but are separated
 * to keep the codebase modular and testable.
 */

import { ArrayStorage } from '../core/storage';
import { isBigIntDType } from '../core/dtype';
import { elementwiseBinaryOp } from '../internal/compute';

/**
 * Add two arrays or array and scalar
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Result storage
 */
export function add(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return addScalar(a, b);
  }
  return elementwiseBinaryOp(a, b, (x, y) => x + y, 'add');
}

/**
 * Subtract two arrays or array and scalar
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Result storage
 */
export function subtract(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return subtractScalar(a, b);
  }
  return elementwiseBinaryOp(a, b, (x, y) => x - y, 'subtract');
}

/**
 * Multiply two arrays or array and scalar
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Result storage
 */
export function multiply(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return multiplyScalar(a, b);
  }
  return elementwiseBinaryOp(a, b, (x, y) => x * y, 'multiply');
}

/**
 * Divide two arrays or array and scalar
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Result storage
 */
export function divide(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return divideScalar(a, b);
  }
  return elementwiseBinaryOp(a, b, (x, y) => x / y, 'divide');
}

/**
 * Add scalar to array (optimized path)
 * @private
 */
function addScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic - no precision loss
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const scalarBig = BigInt(Math.round(scalar));
    for (let i = 0; i < size; i++) {
      resultTyped[i] = thisTyped[i]! + scalarBig;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(data[i]!) + scalar;
    }
  }

  return result;
}

/**
 * Subtract scalar from array (optimized path)
 * @private
 */
function subtractScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic - no precision loss
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const scalarBig = BigInt(Math.round(scalar));
    for (let i = 0; i < size; i++) {
      resultTyped[i] = thisTyped[i]! - scalarBig;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(data[i]!) - scalar;
    }
  }

  return result;
}

/**
 * Multiply array by scalar (optimized path)
 * @private
 */
function multiplyScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic - no precision loss
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const scalarBig = BigInt(Math.round(scalar));
    for (let i = 0; i < size; i++) {
      resultTyped[i] = thisTyped[i]! * scalarBig;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(data[i]!) * scalar;
    }
  }

  return result;
}

/**
 * Divide array by scalar (optimized path)
 * @private
 */
function divideScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt division - no precision loss
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const scalarBig = BigInt(Math.round(scalar));
    for (let i = 0; i < size; i++) {
      resultTyped[i] = thisTyped[i]! / scalarBig;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(data[i]!) / scalar;
    }
  }

  return result;
}
