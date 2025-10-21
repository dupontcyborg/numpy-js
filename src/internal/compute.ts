/**
 * Computation backend abstraction
 *
 * Internal module for element-wise and broadcast operations.
 * Provides a swappable backend for different computation strategies.
 *
 * @internal
 */

/// <reference types="@stdlib/types"/>

import broadcastArrays from '@stdlib/ndarray/broadcast-arrays';
import type { ndarray as StdlibNDArray } from '@stdlib/types/ndarray';
import { ArrayStorage } from '../core/storage';
import { promoteDTypes, isBigIntDType } from '../core/dtype';

// Internal interface for stdlib ndarray with iget method (not in @stdlib types but exists at runtime)
interface StdlibNDArrayInternal extends StdlibNDArray {
  iget(idx: number): number | bigint;
}

/**
 * Perform element-wise operation with broadcasting
 *
 * @param a - First array storage
 * @param b - Second array storage
 * @param op - Operation to perform (a, b) => result
 * @param opName - Name of operation (for special handling)
 * @returns Result storage
 */
export function elementwiseBinaryOp(
  a: ArrayStorage,
  b: ArrayStorage,
  op: (a: number, b: number) => number,
  opName: string
): ArrayStorage {
  // Get stdlib arrays for broadcasting
  let aBroadcast: StdlibNDArrayInternal;
  let bBroadcast: StdlibNDArrayInternal;

  try {
    [aBroadcast, bBroadcast] = broadcastArrays([a.stdlib, b.stdlib]) as [
      StdlibNDArrayInternal,
      StdlibNDArrayInternal,
    ];
  } catch (error) {
    // Re-throw with NumPy-compatible error message
    throw new Error(
      `operands could not be broadcast together with shapes ${JSON.stringify(a.shape)} ${JSON.stringify(b.shape)}`
    );
  }

  // Determine output dtype using NumPy promotion rules
  const resultDtype = promoteDTypes(a.dtype, b.dtype);

  // Get output shape from broadcast result
  const outputShape = Array.from(aBroadcast.shape as ArrayLike<number>);

  // Create result storage
  const result = ArrayStorage.zeros(outputShape, resultDtype);
  const resultData = result.data;
  const size = result.size;

  if (isBigIntDType(resultDtype)) {
    // BigInt arithmetic - no precision loss
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      const aRaw = aBroadcast.iget(i);
      const bRaw = bBroadcast.iget(i);

      // Convert to BigInt - handle case where value is already BigInt
      const aVal = typeof aRaw === 'bigint' ? aRaw : BigInt(Math.round(aRaw));
      const bVal = typeof bRaw === 'bigint' ? bRaw : BigInt(Math.round(bRaw));

      // Use BigInt operations
      if (opName === 'add') {
        resultTyped[i] = aVal + bVal;
      } else if (opName === 'subtract') {
        resultTyped[i] = aVal - bVal;
      } else if (opName === 'multiply') {
        resultTyped[i] = aVal * bVal;
      } else if (opName === 'divide') {
        resultTyped[i] = aVal / bVal;
      } else {
        resultTyped[i] = BigInt(Math.round(op(Number(aVal), Number(bVal))));
      }
    }
  } else {
    // Regular numeric types (including float dtypes)
    // Need to convert BigInt values to Number if mixing dtypes
    const needsConversion = isBigIntDType(a.dtype) || isBigIntDType(b.dtype);

    for (let i = 0; i < size; i++) {
      const aRaw = aBroadcast.iget(i);
      const bRaw = bBroadcast.iget(i);

      // Convert to Number if needed (handles BigInt → float promotion)
      const aVal = needsConversion && typeof aRaw === 'bigint' ? Number(aRaw) : Number(aRaw);
      const bVal = needsConversion && typeof bRaw === 'bigint' ? Number(bRaw) : Number(bRaw);

      resultData[i] = op(aVal, bVal);
    }
  }

  return result;
}

/**
 * Perform element-wise comparison with broadcasting
 * Returns boolean array (dtype: 'bool', stored as Uint8Array)
 */
export function elementwiseComparisonOp(
  a: ArrayStorage,
  b: ArrayStorage,
  op: (a: number, b: number) => boolean
): ArrayStorage {
  // Get stdlib arrays for broadcasting
  let aBroadcast: StdlibNDArrayInternal;
  let bBroadcast: StdlibNDArrayInternal;

  try {
    [aBroadcast, bBroadcast] = broadcastArrays([a.stdlib, b.stdlib]) as [
      StdlibNDArrayInternal,
      StdlibNDArrayInternal,
    ];
  } catch (error) {
    // Re-throw with NumPy-compatible error message
    throw new Error(
      `operands could not be broadcast together with shapes ${JSON.stringify(a.shape)} ${JSON.stringify(b.shape)}`
    );
  }

  // Get output shape
  const outputShape = Array.from(aBroadcast.shape as ArrayLike<number>);
  const size = outputShape.reduce((a, b) => a * b, 1);

  // Create result array with bool dtype
  const resultData = new Uint8Array(size);

  // Check if we need to convert BigInt to Number for comparison
  const needsConversion = isBigIntDType(a.dtype) || isBigIntDType(b.dtype);

  // Perform element-wise comparison
  for (let i = 0; i < size; i++) {
    const aRaw = aBroadcast.iget(i);
    const bRaw = bBroadcast.iget(i);

    // Convert BigInt to Number if needed
    const aVal = needsConversion && typeof aRaw === 'bigint' ? Number(aRaw) : Number(aRaw);
    const bVal = needsConversion && typeof bRaw === 'bigint' ? Number(bRaw) : Number(bRaw);

    resultData[i] = op(aVal, bVal) ? 1 : 0;
  }

  return ArrayStorage.fromData(resultData, outputShape, 'bool');
}
