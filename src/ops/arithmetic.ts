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

/**
 * Absolute value of each element
 * Preserves dtype
 *
 * @param a - Input array storage
 * @returns Result storage with absolute values
 */
export function absolute(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      const val = thisTyped[i]!;
      resultTyped[i] = val < 0n ? -val : val;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.abs(Number(data[i]!));
    }
  }

  return result;
}

/**
 * Numerical negative (element-wise negation)
 * Preserves dtype
 *
 * @param a - Input array storage
 * @returns Result storage with negated values
 */
export function negative(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      resultTyped[i] = -thisTyped[i]!;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = -Number(data[i]!);
    }
  }

  return result;
}

/**
 * Sign of each element (-1, 0, or 1)
 * Preserves dtype
 *
 * @param a - Input array storage
 * @returns Result storage with signs
 */
export function sign(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      const val = thisTyped[i]!;
      resultTyped[i] = val > 0n ? 1n : val < 0n ? -1n : 0n;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      const val = Number(data[i]!);
      resultData[i] = val > 0 ? 1 : val < 0 ? -1 : 0;
    }
  }

  return result;
}

/**
 * Modulo operation (remainder after division)
 * NumPy behavior: Uses floor modulo (sign follows divisor), not JavaScript's truncate modulo
 * Preserves dtype for integer types
 *
 * @param a - Dividend array storage
 * @param b - Divisor (array storage or scalar)
 * @returns Result storage with modulo values
 */
export function mod(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return modScalar(a, b);
  }
  // NumPy uses floor modulo: ((x % y) + y) % y for proper sign handling
  return elementwiseBinaryOp(a, b, (x, y) => ((x % y) + y) % y, 'mod');
}

/**
 * Modulo with scalar divisor (optimized path)
 * NumPy uses floor modulo: result has same sign as divisor
 * @private
 */
function modScalar(storage: ArrayStorage, divisor: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic - use floor modulo
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const divisorBig = BigInt(Math.round(divisor));
    for (let i = 0; i < size; i++) {
      const val = thisTyped[i]!;
      // Floor modulo for BigInt
      resultTyped[i] = ((val % divisorBig) + divisorBig) % divisorBig;
    }
  } else {
    // Regular numeric types - use floor modulo
    for (let i = 0; i < size; i++) {
      const val = Number(data[i]!);
      // Floor modulo: ((x % y) + y) % y
      resultData[i] = ((val % divisor) + divisor) % divisor;
    }
  }

  return result;
}

/**
 * Floor division (division with result rounded down)
 * NumPy behavior: Preserves integer types
 *
 * @param a - Dividend array storage
 * @param b - Divisor (array storage or scalar)
 * @returns Result storage with floor division values
 */
export function floorDivide(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return floorDivideScalar(a, b);
  }
  return elementwiseBinaryOp(a, b, (x, y) => Math.floor(x / y), 'floor_divide');
}

/**
 * Floor division with scalar divisor (optimized path)
 * @private
 */
function floorDivideScalar(storage: ArrayStorage, divisor: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // NumPy behavior: floor_divide preserves integer types
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt floor division
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const divisorBig = BigInt(Math.round(divisor));
    for (let i = 0; i < size; i++) {
      resultTyped[i] = thisTyped[i]! / divisorBig;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.floor(Number(data[i]!) / divisor);
    }
  }

  return result;
}

/**
 * Unary positive (returns a copy of the array)
 * Preserves dtype
 *
 * @param a - Input array storage
 * @returns Result storage (copy of input)
 */
export function positive(a: ArrayStorage): ArrayStorage {
  // Positive is essentially a no-op that returns a copy
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  // Copy data
  for (let i = 0; i < size; i++) {
    resultData[i] = data[i]!;
  }

  return result;
}

/**
 * Reciprocal (1/x) of each element
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with reciprocal values
 */
export function reciprocal(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  // NumPy behavior: reciprocal always promotes integers to float64
  const isIntegerType = dtype !== 'float32' && dtype !== 'float64';
  const resultDtype = isIntegerType ? 'float64' : dtype;

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt input promotes to float64
    for (let i = 0; i < size; i++) {
      resultData[i] = 1.0 / Number(data[i]!);
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = 1.0 / Number(data[i]!);
    }
  }

  return result;
}
