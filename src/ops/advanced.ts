/**
 * Advanced array operations
 *
 * Broadcasting, indexing, and comparison functions.
 * @module ops/advanced
 */

import { ArrayStorage, computeStrides } from '../core/storage';
import { getTypedArrayConstructor, isBigIntDType, type TypedArray } from '../core/dtype';
import { computeBroadcastShape, broadcastTo } from '../core/broadcasting';

/**
 * Broadcast an array to a given shape
 * Returns a read-only view on the original array
 */
export function broadcast_to(storage: ArrayStorage, targetShape: number[]): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const targetNdim = targetShape.length;

  if (targetNdim < ndim) {
    throw new Error(`input operand has more dimensions than allowed by the axis remapping`);
  }

  // Validate that broadcasting is possible
  const broadcastedShape = computeBroadcastShape([Array.from(shape), targetShape]);
  if (broadcastedShape === null) {
    throw new Error(
      `operands could not be broadcast together with shape (${shape.join(',')}) (${targetShape.join(',')})`
    );
  }

  // Check result matches target
  for (let i = 0; i < targetNdim; i++) {
    if (broadcastedShape[i] !== targetShape[i]) {
      throw new Error(
        `operands could not be broadcast together with shape (${shape.join(',')}) (${targetShape.join(',')})`
      );
    }
  }

  return broadcastTo(storage, targetShape);
}

/**
 * Broadcast multiple arrays to a common shape
 * Returns views on the original arrays
 */
export function broadcast_arrays(storages: ArrayStorage[]): ArrayStorage[] {
  if (storages.length === 0) {
    return [];
  }

  if (storages.length === 1) {
    return [storages[0]!];
  }

  // Compute broadcast shape
  const shapes = storages.map((s) => Array.from(s.shape));
  const targetShape = computeBroadcastShape(shapes);

  if (targetShape === null) {
    throw new Error(
      `operands could not be broadcast together with shapes ${shapes.map((s) => `(${s.join(',')})`).join(' ')}`
    );
  }

  // Broadcast each array to the target shape
  return storages.map((s) => broadcastTo(s, targetShape));
}

/**
 * Take elements from an array along an axis
 */
export function take(storage: ArrayStorage, indices: number[], axis?: number): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;

  if (axis === undefined) {
    // Flatten and take
    const flatSize = storage.size;

    // Validate indices
    for (const idx of indices) {
      const normalizedIdx = idx < 0 ? flatSize + idx : idx;
      if (normalizedIdx < 0 || normalizedIdx >= flatSize) {
        throw new Error(`index ${idx} is out of bounds for axis 0 with size ${flatSize}`);
      }
    }

    // Create output array
    const outputSize = indices.length;
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot take from array with dtype ${dtype}`);
    }
    const outputData = new Constructor(outputSize);

    for (let i = 0; i < outputSize; i++) {
      let idx = indices[i]!;
      if (idx < 0) idx = flatSize + idx;
      const value = storage.iget(idx);

      if (isBigIntDType(dtype)) {
        (outputData as BigInt64Array | BigUint64Array)[i] = value as bigint;
      } else {
        (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value as number;
      }
    }

    return ArrayStorage.fromData(outputData, [outputSize], dtype);
  }

  // Take along specified axis
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;

  // Validate indices
  for (const idx of indices) {
    const normalizedIdx = idx < 0 ? axisSize + idx : idx;
    if (normalizedIdx < 0 || normalizedIdx >= axisSize) {
      throw new Error(
        `index ${idx} is out of bounds for axis ${normalizedAxis} with size ${axisSize}`
      );
    }
  }

  // Calculate output shape
  const outputShape = Array.from(shape);
  outputShape[normalizedAxis] = indices.length;

  const outputSize = outputShape.reduce((a, b) => a * b, 1);
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot take from array with dtype ${dtype}`);
  }
  const outputData = new Constructor(outputSize);
  const outputStrides = computeStrides(outputShape);

  // Iterate through output positions
  const outputIndices = new Array(ndim).fill(0);
  for (let i = 0; i < outputSize; i++) {
    // Compute source index
    const sourceIndices = [...outputIndices];
    let targetIdx = outputIndices[normalizedAxis]!;
    let sourceAxisIdx = indices[targetIdx]!;
    if (sourceAxisIdx < 0) sourceAxisIdx = axisSize + sourceAxisIdx;
    sourceIndices[normalizedAxis] = sourceAxisIdx;

    const value = storage.get(...sourceIndices);

    // Write to output
    let outIdx = 0;
    for (let d = 0; d < ndim; d++) {
      outIdx += outputIndices[d]! * outputStrides[d]!;
    }

    if (isBigIntDType(dtype)) {
      (outputData as BigInt64Array | BigUint64Array)[outIdx] = value as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[outIdx] = value as number;
    }

    // Increment indices
    for (let d = ndim - 1; d >= 0; d--) {
      outputIndices[d]++;
      if (outputIndices[d]! < outputShape[d]!) {
        break;
      }
      outputIndices[d] = 0;
    }
  }

  return ArrayStorage.fromData(outputData, outputShape, dtype);
}

/**
 * Put values at specified indices (modifies array in-place)
 */
export function put(
  storage: ArrayStorage,
  indices: number[],
  values: ArrayStorage | number | bigint
): void {
  const flatSize = storage.size;
  const dtype = storage.dtype;

  // Get values to put
  let valueArray: (number | bigint)[];
  if (typeof values === 'number' || typeof values === 'bigint') {
    valueArray = new Array(indices.length).fill(values);
  } else {
    // Extract values from storage
    valueArray = [];
    for (let i = 0; i < values.size; i++) {
      valueArray.push(values.iget(i));
    }
    // Broadcast values if needed
    if (valueArray.length === 1) {
      valueArray = new Array(indices.length).fill(valueArray[0]);
    } else if (valueArray.length !== indices.length) {
      // Tile values to match indices length
      const original = [...valueArray];
      valueArray = [];
      for (let i = 0; i < indices.length; i++) {
        valueArray.push(original[i % original.length]!);
      }
    }
  }

  // Put values at indices
  for (let i = 0; i < indices.length; i++) {
    let idx = indices[i]!;
    if (idx < 0) idx = flatSize + idx;

    if (idx < 0 || idx >= flatSize) {
      throw new Error(`index ${indices[i]} is out of bounds for axis 0 with size ${flatSize}`);
    }

    let value = valueArray[i]!;

    // Convert value to appropriate type
    if (isBigIntDType(dtype)) {
      if (typeof value !== 'bigint') {
        value = BigInt(Math.round(Number(value)));
      }
    } else {
      if (typeof value === 'bigint') {
        value = Number(value);
      }
    }

    storage.iset(idx, value);
  }
}

/**
 * Construct array from index array and choices
 */
export function choose(indexStorage: ArrayStorage, choices: ArrayStorage[]): ArrayStorage {
  if (choices.length === 0) {
    throw new Error('choices cannot be empty');
  }

  const indexShape = indexStorage.shape;
  const numChoices = choices.length;
  const dtype = choices[0]!.dtype;

  // Validate that all choices have compatible shapes
  const shapes = choices.map((c) => Array.from(c.shape));
  shapes.unshift(Array.from(indexShape));
  const broadcastedShape = computeBroadcastShape(shapes);

  if (broadcastedShape === null) {
    throw new Error('operands could not be broadcast together');
  }

  // Broadcast index array and choices to common shape
  const broadcastedIndex = broadcastTo(indexStorage, broadcastedShape);
  const broadcastedChoices = choices.map((c) => broadcastTo(c, broadcastedShape));

  // Create output array
  const outputSize = broadcastedShape.reduce((a, b) => a * b, 1);
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot choose with dtype ${dtype}`);
  }
  const outputData = new Constructor(outputSize);

  // Fill output
  for (let i = 0; i < outputSize; i++) {
    const choiceIdx = Number(broadcastedIndex.iget(i));

    if (choiceIdx < 0 || choiceIdx >= numChoices) {
      throw new Error(`index ${choiceIdx} is out of bounds for axis 0 with size ${numChoices}`);
    }

    const value = broadcastedChoices[choiceIdx]!.iget(i);

    if (isBigIntDType(dtype)) {
      (outputData as BigInt64Array | BigUint64Array)[i] = value as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value as number;
    }
  }

  return ArrayStorage.fromData(outputData, broadcastedShape, dtype);
}

/**
 * Check if two arrays are element-wise equal
 */
export function array_equal(a: ArrayStorage, b: ArrayStorage, equal_nan: boolean = false): boolean {
  // Check shapes match
  if (a.ndim !== b.ndim) {
    return false;
  }

  for (let i = 0; i < a.ndim; i++) {
    if (a.shape[i] !== b.shape[i]) {
      return false;
    }
  }

  // Check all elements
  const size = a.size;
  for (let i = 0; i < size; i++) {
    const aVal = a.iget(i);
    const bVal = b.iget(i);

    // Handle NaN comparison
    if (equal_nan) {
      const aIsNaN = typeof aVal === 'number' && Number.isNaN(aVal);
      const bIsNaN = typeof bVal === 'number' && Number.isNaN(bVal);
      if (aIsNaN && bIsNaN) {
        continue;
      }
    }

    if (aVal !== bVal) {
      return false;
    }
  }

  return true;
}
