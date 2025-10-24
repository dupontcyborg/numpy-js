/**
 * Reduction operations (sum, mean, max, min)
 *
 * Pure functions for reducing arrays along axes.
 * @module ops/reduction
 */

import { ArrayStorage } from '../core/storage';
import { isBigIntDType } from '../core/dtype';
import { outerIndexToMultiIndex, multiIndexToLinear } from '../internal/indexing';

/**
 * Sum array elements over a given axis
 */
export function sum(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Sum all elements - return scalar
    if (isBigIntDType(dtype)) {
      const typedData = data as BigInt64Array | BigUint64Array;
      let total = BigInt(0);
      for (let i = 0; i < size; i++) {
        total += typedData[i]!;
      }
      return Number(total);
    } else {
      let total = 0;
      for (let i = 0; i < size; i++) {
        total += Number(data[i]!);
      }
      return total;
    }
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar - reuse scalar sum logic
    return sum(storage);
  }

  // Create result storage
  const result = ArrayStorage.zeros(outputShape, dtype);
  const resultData = result.data;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let sumVal = BigInt(0);
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        sumVal += typedData[linearIdx]!;
      }
      resultTyped[outerIdx] = sumVal;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let sumVal = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        sumVal += Number(data[linearIdx]!);
      }
      resultData[outerIdx] = sumVal;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
  }

  return result;
}

/**
 * Compute the arithmetic mean along the specified axis
 * Note: mean() returns float64 for integer dtypes, matching NumPy behavior
 */
export function mean(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number {
  const dtype = storage.dtype;
  const shape = storage.shape;

  if (axis === undefined) {
    return (sum(storage) as number) / storage.size;
  }

  // Normalize negative axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = shape.length + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= shape.length) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${shape.length}`);
  }

  const sumResult = sum(storage, axis, keepdims);
  if (typeof sumResult === 'number') {
    return sumResult / shape[normalizedAxis]!;
  }

  // Divide by the size of the reduced axis
  const divisor = shape[normalizedAxis]!;

  // For integer dtypes, mean returns float64 (matching NumPy behavior)
  let resultDtype = dtype;
  if (isBigIntDType(dtype) || dtype.startsWith('int') || dtype.startsWith('uint')) {
    resultDtype = 'float64';
  }

  const result = ArrayStorage.zeros(Array.from(sumResult.shape), resultDtype);
  const resultData = result.data;
  const sumData = sumResult.data;

  if (isBigIntDType(dtype)) {
    // Convert BigInt sum results to float for mean
    const sumTyped = sumData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < resultData.length; i++) {
      resultData[i] = Number(sumTyped[i]!) / divisor;
    }
  } else {
    for (let i = 0; i < resultData.length; i++) {
      resultData[i] = Number(sumData[i]!) / divisor;
    }
  }

  return result;
}

/**
 * Return the maximum along a given axis
 */
export function max(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Max of all elements - return scalar
    if (size === 0) {
      throw new Error('max of empty array');
    }

    let maxVal = data[0]!;
    for (let i = 1; i < size; i++) {
      if (data[i]! > maxVal) {
        maxVal = data[i]!;
      }
    }
    return Number(maxVal);
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return max(storage);
  }

  // Create result storage
  const result = ArrayStorage.zeros(outputShape, dtype);
  const resultData = result.data;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Initialize with first value along axis
      const firstIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, 0, shape);
      const firstIdx = multiIndexToLinear(firstIndices, shape);
      let maxVal = typedData[firstIdx]!;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = typedData[linearIdx]!;
        if (val > maxVal) {
          maxVal = val;
        }
      }
      resultTyped[outerIdx] = maxVal;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let maxVal = -Infinity;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = Number(data[linearIdx]!);
        if (val > maxVal) {
          maxVal = val;
        }
      }
      resultData[outerIdx] = maxVal;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
  }

  return result;
}

/**
 * Product array elements over a given axis
 */
export function prod(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Product of all elements - return scalar
    if (isBigIntDType(dtype)) {
      const typedData = data as BigInt64Array | BigUint64Array;
      let product = BigInt(1);
      for (let i = 0; i < size; i++) {
        product *= typedData[i]!;
      }
      return Number(product);
    } else {
      let product = 1;
      for (let i = 0; i < size; i++) {
        product *= Number(data[i]!);
      }
      return product;
    }
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar - reuse scalar prod logic
    return prod(storage);
  }

  // Create result storage
  const result = ArrayStorage.zeros(outputShape, dtype);
  const resultData = result.data;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let prodVal = BigInt(1);
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        prodVal *= typedData[linearIdx]!;
      }
      resultTyped[outerIdx] = prodVal;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let prodVal = 1;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        prodVal *= Number(data[linearIdx]!);
      }
      resultData[outerIdx] = prodVal;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
  }

  return result;
}

/**
 * Return the minimum along a given axis
 */
export function min(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Min of all elements - return scalar
    if (size === 0) {
      throw new Error('min of empty array');
    }

    let minVal = data[0]!;
    for (let i = 1; i < size; i++) {
      if (data[i]! < minVal) {
        minVal = data[i]!;
      }
    }
    return Number(minVal);
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return min(storage);
  }

  // Create result storage
  const result = ArrayStorage.zeros(outputShape, dtype);
  const resultData = result.data;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Initialize with first value along axis
      const firstIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, 0, shape);
      const firstIdx = multiIndexToLinear(firstIndices, shape);
      let minVal = typedData[firstIdx]!;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = typedData[linearIdx]!;
        if (val < minVal) {
          minVal = val;
        }
      }
      resultTyped[outerIdx] = minVal;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let minVal = Infinity;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = Number(data[linearIdx]!);
        if (val < minVal) {
          minVal = val;
        }
      }
      resultData[outerIdx] = minVal;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
  }

  return result;
}

/**
 * Return the indices of the minimum values along a given axis
 */
export function argmin(storage: ArrayStorage, axis?: number): ArrayStorage | number {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Argmin of all elements - return scalar index
    if (size === 0) {
      throw new Error('argmin of empty array');
    }

    let minVal = data[0]!;
    let minIdx = 0;
    for (let i = 1; i < size; i++) {
      if (data[i]! < minVal) {
        minVal = data[i]!;
        minIdx = i;
      }
    }
    return minIdx;
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return argmin(storage);
  }

  // Create result storage with int32 dtype (indices are always integers)
  const result = ArrayStorage.zeros(outputShape, 'int32');
  const resultData = result.data;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Initialize with first value along axis
      const firstIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, 0, shape);
      const firstIdx = multiIndexToLinear(firstIndices, shape);
      let minVal = typedData[firstIdx]!;
      let minAxisIdx = 0;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = typedData[linearIdx]!;
        if (val < minVal) {
          minVal = val;
          minAxisIdx = axisIdx;
        }
      }
      resultData[outerIdx] = minAxisIdx;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let minVal = Infinity;
      let minAxisIdx = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = Number(data[linearIdx]!);
        if (val < minVal) {
          minVal = val;
          minAxisIdx = axisIdx;
        }
      }
      resultData[outerIdx] = minAxisIdx;
    }
  }

  return result;
}

/**
 * Return the indices of the maximum values along a given axis
 */
export function argmax(storage: ArrayStorage, axis?: number): ArrayStorage | number {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Argmax of all elements - return scalar index
    if (size === 0) {
      throw new Error('argmax of empty array');
    }

    let maxVal = data[0]!;
    let maxIdx = 0;
    for (let i = 1; i < size; i++) {
      if (data[i]! > maxVal) {
        maxVal = data[i]!;
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return argmax(storage);
  }

  // Create result storage with int32 dtype (indices are always integers)
  const result = ArrayStorage.zeros(outputShape, 'int32');
  const resultData = result.data;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Initialize with first value along axis
      const firstIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, 0, shape);
      const firstIdx = multiIndexToLinear(firstIndices, shape);
      let maxVal = typedData[firstIdx]!;
      let maxAxisIdx = 0;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = typedData[linearIdx]!;
        if (val > maxVal) {
          maxVal = val;
          maxAxisIdx = axisIdx;
        }
      }
      resultData[outerIdx] = maxAxisIdx;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let maxVal = -Infinity;
      let maxAxisIdx = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = Number(data[linearIdx]!);
        if (val > maxVal) {
          maxVal = val;
          maxAxisIdx = axisIdx;
        }
      }
      resultData[outerIdx] = maxAxisIdx;
    }
  }

  return result;
}

/**
 * Compute the variance along the specified axis
 * @param storage - Input array storage
 * @param axis - Axis along which to compute variance
 * @param ddof - Delta degrees of freedom (default: 0)
 * @param keepdims - Keep dimensions (default: false)
 */
export function variance(
  storage: ArrayStorage,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): ArrayStorage | number {
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  // Compute mean
  const meanResult = mean(storage, axis, keepdims);

  if (axis === undefined) {
    // Variance of all elements - return scalar
    const meanVal = meanResult as number;
    let sumSqDiff = 0;

    for (let i = 0; i < size; i++) {
      const diff = Number(data[i]!) - meanVal;
      sumSqDiff += diff * diff;
    }

    return sumSqDiff / (size - ddof);
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;
  const meanArray = meanResult as ArrayStorage;
  const meanData = meanArray.data;

  // Compute output shape (same as mean's output shape)
  const outputShape = keepdims
    ? meanArray.shape
    : Array.from(shape).filter((_, i) => i !== normalizedAxis);

  // Result is always float64 for variance
  const result = ArrayStorage.zeros(Array.from(outputShape), 'float64');
  const resultData = result.data;

  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  // Compute variance for each position
  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let sumSqDiff = 0;
    const meanVal = Number(meanData[outerIdx]!);

    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      const diff = Number(data[linearIdx]!) - meanVal;
      sumSqDiff += diff * diff;
    }

    resultData[outerIdx] = sumSqDiff / (axisSize - ddof);
  }

  return result;
}

/**
 * Compute the standard deviation along the specified axis
 * @param storage - Input array storage
 * @param axis - Axis along which to compute std
 * @param ddof - Delta degrees of freedom (default: 0)
 * @param keepdims - Keep dimensions (default: false)
 */
export function std(
  storage: ArrayStorage,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): ArrayStorage | number {
  const varResult = variance(storage, axis, ddof, keepdims);

  if (typeof varResult === 'number') {
    return Math.sqrt(varResult);
  }

  // Apply sqrt element-wise
  const result = ArrayStorage.zeros(Array.from(varResult.shape), 'float64');
  const varData = varResult.data;
  const resultData = result.data;

  for (let i = 0; i < varData.length; i++) {
    resultData[i] = Math.sqrt(Number(varData[i]!));
  }

  return result;
}

/**
 * Test whether all array elements along a given axis evaluate to True
 */
export function all(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | boolean {
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Test all elements
    for (let i = 0; i < size; i++) {
      if (!data[i]) {
        return false;
      }
    }
    return true;
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return all(storage);
  }

  // Create result storage with bool dtype
  const result = ArrayStorage.zeros(outputShape, 'bool');
  const resultData = result.data as Uint8Array;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let allTrue = true;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      if (!data[linearIdx]) {
        allTrue = false;
        break;
      }
    }
    resultData[outerIdx] = allTrue ? 1 : 0;
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'bool');
  }

  return result;
}

/**
 * Test whether any array elements along a given axis evaluate to True
 */
export function any(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | boolean {
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Test all elements
    for (let i = 0; i < size; i++) {
      if (data[i]) {
        return true;
      }
    }
    return false;
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return any(storage);
  }

  // Create result storage with bool dtype
  const result = ArrayStorage.zeros(outputShape, 'bool');
  const resultData = result.data as Uint8Array;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let anyTrue = false;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      if (data[linearIdx]) {
        anyTrue = true;
        break;
      }
    }
    resultData[outerIdx] = anyTrue ? 1 : 0;
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'bool');
  }

  return result;
}
