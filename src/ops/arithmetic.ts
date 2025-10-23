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
 * NumPy behavior: Integer division always promotes to float
 * Type promotion rules:
 * - float64 + anything → float64
 * - float32 + integer → float32
 * - integer + integer → float64
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Result storage with promoted float dtype
 */
export function divide(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return divideScalar(a, b);
  }

  // Determine result dtype using NumPy promotion rules
  const aIsFloat64 = a.dtype === 'float64';
  const bIsFloat64 = b.dtype === 'float64';
  const aIsFloat32 = a.dtype === 'float32';
  const bIsFloat32 = b.dtype === 'float32';

  // If either is float64, result is float64
  if (aIsFloat64 || bIsFloat64) {
    const aFloat = aIsFloat64 ? a : convertToFloatDType(a, 'float64');
    const bFloat = bIsFloat64 ? b : convertToFloatDType(b, 'float64');
    return elementwiseBinaryOp(aFloat, bFloat, (x, y) => x / y, 'divide');
  }

  // If either is float32, result is float32
  if (aIsFloat32 || bIsFloat32) {
    const aFloat = aIsFloat32 ? a : convertToFloatDType(a, 'float32');
    const bFloat = bIsFloat32 ? b : convertToFloatDType(b, 'float32');
    return elementwiseBinaryOp(aFloat, bFloat, (x, y) => x / y, 'divide');
  }

  // Both are integers, promote to float64
  const aFloat = convertToFloatDType(a, 'float64');
  const bFloat = convertToFloatDType(b, 'float64');
  return elementwiseBinaryOp(aFloat, bFloat, (x, y) => x / y, 'divide');
}

/**
 * Convert ArrayStorage to float dtype
 * @private
 */
function convertToFloatDType(
  storage: ArrayStorage,
  targetDtype: 'float32' | 'float64'
): ArrayStorage {
  const result = ArrayStorage.zeros(Array.from(storage.shape), targetDtype);
  const size = storage.size;
  const srcData = storage.data;
  const dstData = result.data;

  for (let i = 0; i < size; i++) {
    dstData[i] = Number(srcData[i]!);
  }

  return result;
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
 * NumPy behavior: Integer division promotes to float64
 * @private
 */
function divideScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // NumPy behavior: Integer division always promotes to float64
  // This allows representing inf/nan for division by zero
  // Bool is also promoted to float64 (NumPy behavior)
  const isIntegerType = dtype !== 'float32' && dtype !== 'float64';
  const resultDtype = isIntegerType ? 'float64' : dtype;

  // Create result with promoted dtype
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // Convert BigInt to Number for division (promotes to float64)
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(data[i]!) / scalar;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(data[i]!) / scalar;
    }
  }

  return result;
}
