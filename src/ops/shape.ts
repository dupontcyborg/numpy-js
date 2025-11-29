/**
 * Shape manipulation operations
 *
 * Pure functions for reshaping, transposing, and manipulating array dimensions.
 * @module ops/shape
 */

import { ArrayStorage, computeStrides } from '../core/storage';
import { getTypedArrayConstructor, type TypedArray } from '../core/dtype';

/**
 * Reshape array to a new shape
 * Returns a view if array is C-contiguous, otherwise copies data
 */
export function reshape(storage: ArrayStorage, newShape: number[]): ArrayStorage {
  const size = storage.size;
  const dtype = storage.dtype;

  // Check if -1 is in the shape (infer dimension)
  const negIndex = newShape.indexOf(-1);
  let finalShape: number[];

  if (negIndex !== -1) {
    // Infer the dimension at negIndex
    const knownSize = newShape.reduce((acc, dim, i) => (i === negIndex ? acc : acc * dim), 1);
    const inferredDim = size / knownSize;

    if (!Number.isInteger(inferredDim)) {
      throw new Error(
        `cannot reshape array of size ${size} into shape ${JSON.stringify(newShape)}`
      );
    }

    finalShape = newShape.map((dim, i) => (i === negIndex ? inferredDim : dim));
  } else {
    finalShape = newShape;
  }

  // Validate that the new shape has the same total size
  const newSize = finalShape.reduce((a, b) => a * b, 1);
  if (newSize !== size) {
    throw new Error(
      `cannot reshape array of size ${size} into shape ${JSON.stringify(finalShape)}`
    );
  }

  // Fast path: if array is C-contiguous, create a view (no copy)
  if (storage.isCContiguous) {
    const data = storage.data;
    return ArrayStorage.fromData(data, finalShape, dtype, computeStrides(finalShape), 0);
  }

  // Slow path: array is not contiguous, must copy data first
  // Create contiguous copy, then reshape
  const contiguousCopy = storage.copy(); // copy() creates C-contiguous array
  const data = contiguousCopy.data;
  return ArrayStorage.fromData(data, finalShape, dtype, computeStrides(finalShape), 0);
}

/**
 * Return a flattened copy of the array
 * Creates 1D array containing all elements in row-major order
 * Always returns a copy (matching NumPy behavior)
 */
export function flatten(storage: ArrayStorage): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const dtype = storage.dtype;
  const Constructor = getTypedArrayConstructor(dtype);

  if (!Constructor) {
    throw new Error(`Cannot flatten array with dtype ${dtype}`);
  }

  // Always create a copy (NumPy flatten() behavior)
  // Create new data buffer and copy elements in row-major order
  // This respects the current array's strides (handles transposed arrays correctly)
  const newData = new Constructor(size);
  let idx = 0;

  // Helper function to recursively iterate through all indices in row-major order
  const flattenRecursive = (indices: number[], dim: number) => {
    if (dim === ndim) {
      // At leaf, copy the value using get method which respects strides
      const value = storage.get(...indices);
      if (typeof value === 'bigint') {
        (newData as BigInt64Array | BigUint64Array)[idx++] = value;
      } else {
        (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[idx++] = value;
      }
      return;
    }

    // Iterate through current dimension
    for (let i = 0; i < shape[dim]!; i++) {
      indices[dim] = i;
      flattenRecursive(indices, dim + 1);
    }
  };

  flattenRecursive(new Array(ndim), 0);

  return ArrayStorage.fromData(newData, [size], dtype, [1], 0);
}

/**
 * Return a flattened array (view when possible, otherwise copy)
 * Returns a view if array is C-contiguous, otherwise copies data
 */
export function ravel(storage: ArrayStorage): ArrayStorage {
  const size = storage.size;
  const dtype = storage.dtype;

  // Fast path: if array is C-contiguous, create a view (no copy needed)
  if (storage.isCContiguous) {
    const data = storage.data;
    return ArrayStorage.fromData(data, [size], dtype, [1], 0);
  }

  // Slow path: array is not contiguous, must copy like flatten()
  return flatten(storage);
}

/**
 * Transpose array (permute dimensions)
 * Returns a view with transposed dimensions
 */
export function transpose(storage: ArrayStorage, axes?: number[]): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const strides = storage.strides;
  const data = storage.data;
  const dtype = storage.dtype;

  let permutation: number[];

  if (axes === undefined) {
    // Default: reverse all dimensions
    permutation = Array.from({ length: ndim }, (_, i) => ndim - 1 - i);
  } else {
    // Validate axes
    if (axes.length !== ndim) {
      throw new Error(`axes must have length ${ndim}, got ${axes.length}`);
    }

    // Check that axes is a valid permutation
    const seen = new Set<number>();
    for (const axis of axes) {
      const normalizedAxis = axis < 0 ? ndim + axis : axis;
      if (normalizedAxis < 0 || normalizedAxis >= ndim) {
        throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
      }
      if (seen.has(normalizedAxis)) {
        throw new Error(`repeated axis in transpose`);
      }
      seen.add(normalizedAxis);
    }

    permutation = axes.map((ax) => (ax < 0 ? ndim + ax : ax));
  }

  // Compute new shape and strides
  const newShape = permutation.map((i) => shape[i]!);
  const oldStrides = Array.from(strides);
  const newStrides = permutation.map((i) => oldStrides[i]!);

  // Create transposed view
  return ArrayStorage.fromData(data, newShape, dtype, newStrides, storage.offset);
}

/**
 * Remove axes of length 1
 * Returns a view with specified dimensions removed
 */
export function squeeze(storage: ArrayStorage, axis?: number): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const strides = storage.strides;
  const data = storage.data;
  const dtype = storage.dtype;

  if (axis === undefined) {
    // Remove all axes with size 1
    const newShape: number[] = [];
    const newStrides: number[] = [];

    for (let i = 0; i < ndim; i++) {
      if (shape[i] !== 1) {
        newShape.push(shape[i]!);
        newStrides.push(strides[i]!);
      }
    }

    // If all dimensions were 1, result would be a scalar (0-d array)
    // For now, keep at least one dimension
    if (newShape.length === 0) {
      newShape.push(1);
      newStrides.push(1);
    }

    return ArrayStorage.fromData(data, newShape, dtype, newStrides, storage.offset);
  } else {
    // Normalize axis
    const normalizedAxis = axis < 0 ? ndim + axis : axis;

    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    // Check that the axis has size 1
    if (shape[normalizedAxis] !== 1) {
      throw new Error(
        `cannot select an axis which has size not equal to one (axis ${axis} has size ${shape[normalizedAxis]})`
      );
    }

    // Remove the specified axis
    const newShape: number[] = [];
    const newStrides: number[] = [];

    for (let i = 0; i < ndim; i++) {
      if (i !== normalizedAxis) {
        newShape.push(shape[i]!);
        newStrides.push(strides[i]!);
      }
    }

    return ArrayStorage.fromData(data, newShape, dtype, newStrides, storage.offset);
  }
}

/**
 * Expand the shape by inserting a new axis of length 1
 * Returns a view with additional dimension
 */
export function expandDims(storage: ArrayStorage, axis: number): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const strides = storage.strides;
  const data = storage.data;
  const dtype = storage.dtype;

  // Normalize axis (can be from -ndim-1 to ndim)
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + axis + 1;
  }

  if (normalizedAxis < 0 || normalizedAxis > ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim + 1}`);
  }

  // Insert 1 at the specified position
  const newShape = [...Array.from(shape)];
  newShape.splice(normalizedAxis, 0, 1);

  // Insert a stride at the new axis position
  // The stride for a dimension of size 1 doesn't matter, but conventionally
  // it should be the product of all dimensions to its right
  const newStrides = [...Array.from(strides)];
  const insertedStride =
    normalizedAxis < ndim ? strides[normalizedAxis]! * (shape[normalizedAxis] || 1) : 1;
  newStrides.splice(normalizedAxis, 0, insertedStride);

  return ArrayStorage.fromData(data, newShape, dtype, newStrides, storage.offset);
}

/**
 * Swap two axes of an array
 * Returns a view with axes swapped
 */
export function swapaxes(storage: ArrayStorage, axis1: number, axis2: number): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const strides = storage.strides;
  const data = storage.data;
  const dtype = storage.dtype;

  // Normalize axes
  let normalizedAxis1 = axis1 < 0 ? ndim + axis1 : axis1;
  let normalizedAxis2 = axis2 < 0 ? ndim + axis2 : axis2;

  if (normalizedAxis1 < 0 || normalizedAxis1 >= ndim) {
    throw new Error(`axis1 ${axis1} is out of bounds for array of dimension ${ndim}`);
  }
  if (normalizedAxis2 < 0 || normalizedAxis2 >= ndim) {
    throw new Error(`axis2 ${axis2} is out of bounds for array of dimension ${ndim}`);
  }

  // If same axis, return a view without change
  if (normalizedAxis1 === normalizedAxis2) {
    return ArrayStorage.fromData(
      data,
      Array.from(shape),
      dtype,
      Array.from(strides),
      storage.offset
    );
  }

  // Swap shape and strides
  const newShape = Array.from(shape);
  const newStrides = Array.from(strides);

  [newShape[normalizedAxis1], newShape[normalizedAxis2]] = [
    newShape[normalizedAxis2]!,
    newShape[normalizedAxis1]!,
  ];
  [newStrides[normalizedAxis1], newStrides[normalizedAxis2]] = [
    newStrides[normalizedAxis2]!,
    newStrides[normalizedAxis1]!,
  ];

  return ArrayStorage.fromData(data, newShape, dtype, newStrides, storage.offset);
}

/**
 * Move axes to new positions
 * Returns a view with axes moved
 */
export function moveaxis(
  storage: ArrayStorage,
  source: number | number[],
  destination: number | number[]
): ArrayStorage {
  const ndim = storage.ndim;

  // Convert to arrays
  const sourceArr = Array.isArray(source) ? source : [source];
  const destArr = Array.isArray(destination) ? destination : [destination];

  if (sourceArr.length !== destArr.length) {
    throw new Error('source and destination must have the same number of elements');
  }

  // Normalize axes
  const normalizedSource = sourceArr.map((ax) => {
    const normalized = ax < 0 ? ndim + ax : ax;
    if (normalized < 0 || normalized >= ndim) {
      throw new Error(`source axis ${ax} is out of bounds for array of dimension ${ndim}`);
    }
    return normalized;
  });

  const normalizedDest = destArr.map((ax) => {
    const normalized = ax < 0 ? ndim + ax : ax;
    if (normalized < 0 || normalized >= ndim) {
      throw new Error(`destination axis ${ax} is out of bounds for array of dimension ${ndim}`);
    }
    return normalized;
  });

  // Check for duplicate source/dest axes
  if (new Set(normalizedSource).size !== normalizedSource.length) {
    throw new Error('repeated axis in source');
  }
  if (new Set(normalizedDest).size !== normalizedDest.length) {
    throw new Error('repeated axis in destination');
  }

  // Build permutation
  // Start with axes not in source
  const order: number[] = [];
  for (let i = 0; i < ndim; i++) {
    if (!normalizedSource.includes(i)) {
      order.push(i);
    }
  }

  // Insert source axes at destination positions
  for (let i = 0; i < normalizedSource.length; i++) {
    const dst = normalizedDest[i]!;
    order.splice(dst, 0, normalizedSource[i]!);
  }

  return transpose(storage, order);
}

/**
 * Concatenate arrays along an axis
 */
export function concatenate(storages: ArrayStorage[], axis: number = 0): ArrayStorage {
  if (storages.length === 0) {
    throw new Error('need at least one array to concatenate');
  }

  if (storages.length === 1) {
    return storages[0]!.copy();
  }

  const first = storages[0]!;
  const ndim = first.ndim;
  const dtype = first.dtype;

  // Normalize axis
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Validate shapes: all arrays must have same shape except along axis
  for (let i = 1; i < storages.length; i++) {
    const s = storages[i]!;
    if (s.ndim !== ndim) {
      throw new Error('all the input arrays must have same number of dimensions');
    }
    for (let d = 0; d < ndim; d++) {
      if (d !== normalizedAxis && s.shape[d] !== first.shape[d]) {
        throw new Error(
          `all the input array dimensions except for the concatenation axis must match exactly`
        );
      }
    }
  }

  // Calculate output shape
  const outputShape = Array.from(first.shape);
  let totalAlongAxis = first.shape[normalizedAxis]!;
  for (let i = 1; i < storages.length; i++) {
    totalAlongAxis += storages[i]!.shape[normalizedAxis]!;
  }
  outputShape[normalizedAxis] = totalAlongAxis;

  // Create output array
  const outputSize = outputShape.reduce((a, b) => a * b, 1);
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot concatenate arrays with dtype ${dtype}`);
  }
  const outputData = new Constructor(outputSize);
  const outputStrides = computeStrides(outputShape);

  // Copy data from each input array
  let offset = 0;
  for (const storage of storages) {
    const axisSize = storage.shape[normalizedAxis]!;
    copyToOutput(storage, outputData, outputShape, outputStrides, normalizedAxis, offset, dtype);
    offset += axisSize;
  }

  return ArrayStorage.fromData(outputData, outputShape, dtype);
}

/**
 * Helper to copy data into concatenated output
 */
function copyToOutput(
  source: ArrayStorage,
  outputData: TypedArray,
  _outputShape: number[],
  outputStrides: number[],
  axis: number,
  axisOffset: number,
  dtype: string
): void {
  const sourceShape = source.shape;
  const ndim = sourceShape.length;
  const sourceSize = source.size;

  // Iterate through all elements in source
  const indices = new Array(ndim).fill(0);

  for (let i = 0; i < sourceSize; i++) {
    // Get value from source
    const value = source.get(...indices);

    // Compute output index
    const outputIndices = [...indices];
    outputIndices[axis] += axisOffset;

    let outputIdx = 0;
    for (let d = 0; d < ndim; d++) {
      outputIdx += outputIndices[d]! * outputStrides[d]!;
    }

    // Write to output
    if (dtype === 'int64' || dtype === 'uint64') {
      (outputData as BigInt64Array | BigUint64Array)[outputIdx] = value as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[outputIdx] =
        value as number;
    }

    // Increment indices
    for (let d = ndim - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d]! < sourceShape[d]!) {
        break;
      }
      indices[d] = 0;
    }
  }
}

/**
 * Stack arrays along a new axis
 */
export function stack(storages: ArrayStorage[], axis: number = 0): ArrayStorage {
  if (storages.length === 0) {
    throw new Error('need at least one array to stack');
  }

  const first = storages[0]!;
  const shape = first.shape;
  const ndim = first.ndim;

  // Normalize axis
  const normalizedAxis = axis < 0 ? ndim + 1 + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis > ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim + 1}`);
  }

  // Validate shapes: all arrays must have exact same shape
  for (let i = 1; i < storages.length; i++) {
    const s = storages[i]!;
    if (s.ndim !== ndim) {
      throw new Error('all input arrays must have the same shape');
    }
    for (let d = 0; d < ndim; d++) {
      if (s.shape[d] !== shape[d]) {
        throw new Error('all input arrays must have the same shape');
      }
    }
  }

  // Expand dims on each array, then concatenate
  const expanded = storages.map((s) => expandDims(s, normalizedAxis));
  return concatenate(expanded, normalizedAxis);
}

/**
 * Stack arrays vertically (row-wise)
 * vstack is equivalent to concatenation along the first axis after
 * 1-D arrays of shape (N,) have been reshaped to (1,N)
 */
export function vstack(storages: ArrayStorage[]): ArrayStorage {
  if (storages.length === 0) {
    throw new Error('need at least one array to stack');
  }

  // For 1D arrays, reshape to (1, N) first
  const prepared = storages.map((s) => {
    if (s.ndim === 1) {
      return reshape(s, [1, s.shape[0]!]);
    }
    return s;
  });

  return concatenate(prepared, 0);
}

/**
 * Stack arrays horizontally (column-wise)
 * hstack is equivalent to concatenation along the second axis,
 * except for 1-D arrays where it concatenates along the first axis
 */
export function hstack(storages: ArrayStorage[]): ArrayStorage {
  if (storages.length === 0) {
    throw new Error('need at least one array to stack');
  }

  // Check if all arrays are 1D
  const allOneDim = storages.every((s) => s.ndim === 1);

  if (allOneDim) {
    // For 1D arrays, concatenate along axis 0
    return concatenate(storages, 0);
  }

  // For higher-dimensional arrays, concatenate along axis 1
  return concatenate(storages, 1);
}

/**
 * Stack arrays depth-wise (along third axis)
 * dstack is equivalent to concatenation along the third axis after
 * 2-D arrays of shape (M,N) have been reshaped to (M,N,1) and
 * 1-D arrays of shape (N,) have been reshaped to (1,N,1)
 */
export function dstack(storages: ArrayStorage[]): ArrayStorage {
  if (storages.length === 0) {
    throw new Error('need at least one array to stack');
  }

  // Prepare arrays
  const prepared = storages.map((s) => {
    if (s.ndim === 1) {
      // Reshape (N,) to (1, N, 1)
      return reshape(expandDims(reshape(s, [1, s.shape[0]!]), 2), [1, s.shape[0]!, 1]);
    } else if (s.ndim === 2) {
      // Reshape (M, N) to (M, N, 1)
      return expandDims(s, 2);
    }
    return s;
  });

  return concatenate(prepared, 2);
}

/**
 * Split array into sub-arrays
 */
export function split(
  storage: ArrayStorage,
  indicesOrSections: number | number[],
  axis: number = 0
): ArrayStorage[] {
  const shape = storage.shape;
  const ndim = shape.length;

  // Normalize axis
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;

  let splitIndices: number[];

  if (typeof indicesOrSections === 'number') {
    // Split into N equal sections
    if (axisSize % indicesOrSections !== 0) {
      throw new Error(`array split does not result in an equal division`);
    }
    const sectionSize = axisSize / indicesOrSections;
    splitIndices = [];
    for (let i = 1; i < indicesOrSections; i++) {
      splitIndices.push(i * sectionSize);
    }
  } else {
    // Split at specified indices
    splitIndices = indicesOrSections;
  }

  return splitAtIndices(storage, splitIndices, normalizedAxis);
}

/**
 * Split array into sub-arrays (allows unequal splits)
 */
export function arraySplit(
  storage: ArrayStorage,
  indicesOrSections: number | number[],
  axis: number = 0
): ArrayStorage[] {
  const shape = storage.shape;
  const ndim = shape.length;

  // Normalize axis
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;

  let splitIndices: number[];

  if (typeof indicesOrSections === 'number') {
    // Split into N sections (may be unequal)
    const numSections = indicesOrSections;
    const sectionSize = Math.floor(axisSize / numSections);
    const remainder = axisSize % numSections;

    splitIndices = [];
    let offset = 0;
    for (let i = 0; i < numSections - 1; i++) {
      offset += sectionSize + (i < remainder ? 1 : 0);
      splitIndices.push(offset);
    }
  } else {
    // Split at specified indices
    splitIndices = indicesOrSections;
  }

  return splitAtIndices(storage, splitIndices, normalizedAxis);
}

/**
 * Helper to split array at specified indices
 */
function splitAtIndices(
  storage: ArrayStorage,
  indices: number[],
  axis: number
): ArrayStorage[] {
  const shape = storage.shape;
  const axisSize = shape[axis]!;

  // Add boundaries
  const boundaries = [0, ...indices, axisSize];
  const result: ArrayStorage[] = [];

  for (let i = 0; i < boundaries.length - 1; i++) {
    const start = boundaries[i]!;
    const end = boundaries[i + 1]!;

    if (start > end) {
      throw new Error('split indices must be in ascending order');
    }

    // Create slice
    const sliceShape = Array.from(shape);
    sliceShape[axis] = end - start;

    // Calculate new offset and strides
    const newOffset = storage.offset + start * storage.strides[axis]!;

    result.push(
      ArrayStorage.fromData(
        storage.data,
        sliceShape,
        storage.dtype,
        Array.from(storage.strides),
        newOffset
      )
    );
  }

  return result;
}

/**
 * Split array vertically (row-wise)
 */
export function vsplit(
  storage: ArrayStorage,
  indicesOrSections: number | number[]
): ArrayStorage[] {
  if (storage.ndim < 2) {
    throw new Error('vsplit only works on arrays of 2 or more dimensions');
  }
  return arraySplit(storage, indicesOrSections, 0);
}

/**
 * Split array horizontally (column-wise)
 */
export function hsplit(
  storage: ArrayStorage,
  indicesOrSections: number | number[]
): ArrayStorage[] {
  if (storage.ndim < 1) {
    throw new Error('hsplit only works on arrays of 1 or more dimensions');
  }
  // For 1D arrays, split along axis 0; for higher dims, split along axis 1
  const axis = storage.ndim === 1 ? 0 : 1;
  return arraySplit(storage, indicesOrSections, axis);
}

/**
 * Tile array by repeating along each axis
 */
export function tile(storage: ArrayStorage, reps: number | number[]): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;

  // Normalize reps to array
  const repsArr = Array.isArray(reps) ? reps : [reps];

  // Pad reps or shape to match dimensions
  const maxDim = Math.max(ndim, repsArr.length);
  const paddedShape = new Array(maxDim).fill(1);
  const paddedReps = new Array(maxDim).fill(1);

  // Fill from the right
  for (let i = 0; i < ndim; i++) {
    paddedShape[maxDim - ndim + i] = shape[i]!;
  }
  for (let i = 0; i < repsArr.length; i++) {
    paddedReps[maxDim - repsArr.length + i] = repsArr[i]!;
  }

  // Calculate output shape
  const outputShape = paddedShape.map((s, i) => s * paddedReps[i]!);
  const outputSize = outputShape.reduce((a, b) => a * b, 1);

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot tile array with dtype ${dtype}`);
  }
  const outputData = new Constructor(outputSize);
  const outputStrides = computeStrides(outputShape);

  // If we need to expand dimensions of input, reshape it
  let expandedStorage = storage;
  if (ndim < maxDim) {
    expandedStorage = reshape(storage, paddedShape);
  }

  // Fill output by iterating through all output positions
  const outputIndices = new Array(maxDim).fill(0);
  for (let i = 0; i < outputSize; i++) {
    // Compute source index (wrap around)
    const sourceIndices = outputIndices.map((idx, d) => idx % paddedShape[d]!);
    const value = expandedStorage.get(...sourceIndices);

    // Write to output
    let outputIdx = 0;
    for (let d = 0; d < maxDim; d++) {
      outputIdx += outputIndices[d]! * outputStrides[d]!;
    }

    if (dtype === 'int64' || dtype === 'uint64') {
      (outputData as BigInt64Array | BigUint64Array)[outputIdx] = value as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[outputIdx] =
        value as number;
    }

    // Increment indices
    for (let d = maxDim - 1; d >= 0; d--) {
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
 * Repeat elements of an array
 */
export function repeat(
  storage: ArrayStorage,
  repeats: number | number[],
  axis?: number
): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;
  const size = storage.size;

  if (axis === undefined) {
    // Flatten and repeat each element
    const flatSize = size;
    const repeatsArr = Array.isArray(repeats) ? repeats : new Array(flatSize).fill(repeats);

    if (repeatsArr.length !== flatSize) {
      throw new Error(
        `operands could not be broadcast together with shape (${flatSize},) (${repeatsArr.length},)`
      );
    }

    const outputSize = repeatsArr.reduce((a, b) => a + b, 0);
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot repeat array with dtype ${dtype}`);
    }
    const outputData = new Constructor(outputSize);

    let outIdx = 0;
    for (let i = 0; i < flatSize; i++) {
      const value = storage.iget(i);
      const rep = repeatsArr[i]!;
      for (let r = 0; r < rep; r++) {
        if (dtype === 'int64' || dtype === 'uint64') {
          (outputData as BigInt64Array | BigUint64Array)[outIdx++] = value as bigint;
        } else {
          (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[outIdx++] =
            value as number;
        }
      }
    }

    return ArrayStorage.fromData(outputData, [outputSize], dtype);
  }

  // Repeat along specified axis
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;
  const repeatsArr = Array.isArray(repeats) ? repeats : new Array(axisSize).fill(repeats);

  if (repeatsArr.length !== axisSize) {
    throw new Error(
      `operands could not be broadcast together with shape (${axisSize},) (${repeatsArr.length},)`
    );
  }

  // Calculate output shape
  const outputShape = Array.from(shape);
  outputShape[normalizedAxis] = repeatsArr.reduce((a, b) => a + b, 0);

  const outputSize = outputShape.reduce((a, b) => a * b, 1);
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot repeat array with dtype ${dtype}`);
  }
  const outputData = new Constructor(outputSize);
  const outputStrides = computeStrides(outputShape);

  // Iterate through source and write repeated values
  const sourceIndices = new Array(ndim).fill(0);
  const outputIndices = new Array(ndim).fill(0);

  // Track cumulative positions along axis
  const axisPositions: number[] = [0];
  for (let i = 0; i < axisSize; i++) {
    axisPositions.push(axisPositions[i]! + repeatsArr[i]!);
  }

  for (let i = 0; i < size; i++) {
    const value = storage.get(...sourceIndices);
    const axisIdx = sourceIndices[normalizedAxis]!;
    const rep = repeatsArr[axisIdx]!;

    // Write repeated values
    for (let r = 0; r < rep; r++) {
      // Set output indices
      for (let d = 0; d < ndim; d++) {
        if (d === normalizedAxis) {
          outputIndices[d] = axisPositions[axisIdx]! + r;
        } else {
          outputIndices[d] = sourceIndices[d];
        }
      }

      // Calculate output flat index
      let outIdx = 0;
      for (let d = 0; d < ndim; d++) {
        outIdx += outputIndices[d]! * outputStrides[d]!;
      }

      if (dtype === 'int64' || dtype === 'uint64') {
        (outputData as BigInt64Array | BigUint64Array)[outIdx] = value as bigint;
      } else {
        (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[outIdx] =
          value as number;
      }
    }

    // Increment source indices
    for (let d = ndim - 1; d >= 0; d--) {
      sourceIndices[d]++;
      if (sourceIndices[d]! < shape[d]!) {
        break;
      }
      sourceIndices[d] = 0;
    }
  }

  return ArrayStorage.fromData(outputData, outputShape, dtype);
}
