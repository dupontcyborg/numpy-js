/**
 * Comparison operations
 *
 * Element-wise comparison operations that return boolean arrays:
 * greater, greater_equal, less, less_equal, equal, not_equal,
 * isclose, allclose
 */

import { ArrayStorage } from '../core/storage';
import { isBigIntDType } from '../core/dtype';
import { elementwiseComparisonOp } from '../internal/compute';

/**
 * Element-wise greater than comparison (a > b)
 */
export function greater(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return greaterScalar(a, b);
  }
  return elementwiseComparisonOp(a, b, (x, y) => x > y);
}

/**
 * Element-wise greater than or equal comparison (a >= b)
 */
export function greaterEqual(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return greaterEqualScalar(a, b);
  }
  return elementwiseComparisonOp(a, b, (x, y) => x >= y);
}

/**
 * Element-wise less than comparison (a < b)
 */
export function less(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return lessScalar(a, b);
  }
  return elementwiseComparisonOp(a, b, (x, y) => x < y);
}

/**
 * Element-wise less than or equal comparison (a <= b)
 */
export function lessEqual(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return lessEqualScalar(a, b);
  }
  return elementwiseComparisonOp(a, b, (x, y) => x <= y);
}

/**
 * Element-wise equality comparison (a == b)
 */
export function equal(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return equalScalar(a, b);
  }
  return elementwiseComparisonOp(a, b, (x, y) => x === y);
}

/**
 * Element-wise inequality comparison (a != b)
 */
export function notEqual(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return notEqualScalar(a, b);
  }
  return elementwiseComparisonOp(a, b, (x, y) => x !== y);
}

/**
 * Element-wise "close" comparison with tolerance
 * Returns true where |a - b| <= atol + rtol * |b|
 */
export function isclose(
  a: ArrayStorage,
  b: ArrayStorage | number,
  rtol: number = 1e-5,
  atol: number = 1e-8
): ArrayStorage {
  if (typeof b === 'number') {
    return iscloseScalar(a, b, rtol, atol);
  }
  return elementwiseComparisonOp(a, b, (x, y) => {
    const diff = Math.abs(x - y);
    const threshold = atol + rtol * Math.abs(y);
    return diff <= threshold;
  });
}

/**
 * Check if all elements are close (scalar result)
 * Returns true if all elements satisfy isclose condition
 */
export function allclose(
  a: ArrayStorage,
  b: ArrayStorage | number,
  rtol: number = 1e-5,
  atol: number = 1e-8
): boolean {
  const closeResult = isclose(a, b, rtol, atol);
  const data = closeResult.data as Uint8Array;

  // Check if all values are 1 (true)
  for (let i = 0; i < closeResult.size; i++) {
    if (data[i] === 0) {
      return false;
    }
  }
  return true;
}

// Scalar comparison optimized paths

function greaterScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;

  for (let i = 0; i < storage.size; i++) {
    data[i] = thisData[i]! > scalar ? 1 : 0;
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

function greaterEqualScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;

  for (let i = 0; i < storage.size; i++) {
    data[i] = thisData[i]! >= scalar ? 1 : 0;
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

function lessScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;

  for (let i = 0; i < storage.size; i++) {
    data[i] = thisData[i]! < scalar ? 1 : 0;
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

function lessEqualScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;

  for (let i = 0; i < storage.size; i++) {
    data[i] = thisData[i]! <= scalar ? 1 : 0;
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

function equalScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;
  const dtype = storage.dtype;

  if (isBigIntDType(dtype)) {
    // BigInt comparison: convert scalar to BigInt
    const scalarBig = BigInt(Math.round(scalar));
    const typedData = thisData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < storage.size; i++) {
      data[i] = typedData[i]! === scalarBig ? 1 : 0;
    }
  } else {
    // Regular comparison
    for (let i = 0; i < storage.size; i++) {
      data[i] = thisData[i]! === scalar ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

function notEqualScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;

  for (let i = 0; i < storage.size; i++) {
    data[i] = thisData[i]! !== scalar ? 1 : 0;
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

function iscloseScalar(
  storage: ArrayStorage,
  scalar: number,
  rtol: number,
  atol: number
): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;
  const dtype = storage.dtype;

  if (isBigIntDType(dtype)) {
    // For BigInt, convert to Number for comparison
    const thisTyped = thisData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < storage.size; i++) {
      const a = Number(thisTyped[i]!);
      const diff = Math.abs(a - scalar);
      const threshold = atol + rtol * Math.abs(scalar);
      data[i] = diff <= threshold ? 1 : 0;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < storage.size; i++) {
      const a = Number(thisData[i]!);
      const diff = Math.abs(a - scalar);
      const threshold = atol + rtol * Math.abs(scalar);
      data[i] = diff <= threshold ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}
