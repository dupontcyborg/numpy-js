/**
 * Shape manipulation operations
 *
 * Pure functions for reshaping, transposing, and manipulating array dimensions.
 * @module ops/shape
 */

import stdlib_ndarray from '@stdlib/ndarray';
import { ArrayStorage } from '../core/storage';
import { computeStrides } from '../internal/indexing';
import { toStdlibDType, getTypedArrayConstructor, type TypedArray } from '../core/dtype';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type StdlibNDArray = any;

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
    const stdlibArray = stdlib_ndarray.ndarray(
      toStdlibDType(dtype) as StdlibNDArray,
      data,
      finalShape,
      computeStrides(finalShape),
      0,
      'row-major'
    );
    return ArrayStorage.fromStdlib(stdlibArray, dtype);
  }

  // Slow path: array is not contiguous, must copy data first
  // Create contiguous copy, then reshape
  const contiguousCopy = storage.copy(); // copy() creates C-contiguous array
  const data = contiguousCopy.data;
  const stdlibArray = stdlib_ndarray.ndarray(
    toStdlibDType(dtype) as StdlibNDArray,
    data,
    finalShape,
    computeStrides(finalShape),
    0,
    'row-major'
  );
  return ArrayStorage.fromStdlib(stdlibArray, dtype);
}

/**
 * Return a flattened copy of the array
 * Creates 1D array containing all elements in row-major order
 * Always returns a copy (matching NumPy behavior)
 */
export function flatten(storage: ArrayStorage): ArrayStorage {
  const stdlib = storage.stdlib;
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
      // At leaf, copy the value using stdlib's get method which respects strides
      const value = stdlib.get(...indices);
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

  const stdlibArray = stdlib_ndarray.ndarray(
    toStdlibDType(dtype) as StdlibNDArray,
    newData,
    [size],
    [1],
    0,
    'row-major'
  );
  return ArrayStorage.fromStdlib(stdlibArray, dtype);
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
    const stdlibArray = stdlib_ndarray.ndarray(
      toStdlibDType(dtype) as StdlibNDArray,
      data,
      [size],
      [1],
      0,
      'row-major'
    );
    return ArrayStorage.fromStdlib(stdlibArray, dtype);
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
  const stdlib = storage.stdlib;

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
  const stdlibArray = stdlib_ndarray.ndarray(
    toStdlibDType(dtype) as StdlibNDArray,
    data,
    newShape,
    newStrides,
    stdlib_ndarray.offset(stdlib),
    'row-major'
  );

  return ArrayStorage.fromStdlib(stdlibArray, dtype);
}

/**
 * Remove axes of length 1
 * Returns a view with specified dimensions removed
 */
export function squeeze(storage: ArrayStorage, axis?: number): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const dtype = storage.dtype;

  if (axis === undefined) {
    // Remove all axes with size 1
    const newShape = Array.from(shape).filter((dim) => dim !== 1);

    // If all dimensions were 1, result would be a scalar (0-d array)
    // For now, keep at least one dimension since stdlib may not fully support 0-d arrays
    if (newShape.length === 0) {
      newShape.push(1);
    }

    const newStrides = computeStrides(newShape);
    const stdlibArray = stdlib_ndarray.ndarray(
      toStdlibDType(dtype) as StdlibNDArray,
      data,
      newShape,
      newStrides,
      0,
      'row-major'
    );

    return ArrayStorage.fromStdlib(stdlibArray, dtype);
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
    const newShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);

    const newStrides = computeStrides(newShape);
    const stdlibArray = stdlib_ndarray.ndarray(
      toStdlibDType(dtype) as StdlibNDArray,
      data,
      newShape,
      newStrides,
      0,
      'row-major'
    );

    return ArrayStorage.fromStdlib(stdlibArray, dtype);
  }
}

/**
 * Expand the shape by inserting a new axis of length 1
 * Returns a view with additional dimension
 */
export function expandDims(storage: ArrayStorage, axis: number): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
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

  const newStrides = computeStrides(newShape);
  const stdlibArray = stdlib_ndarray.ndarray(
    toStdlibDType(dtype) as StdlibNDArray,
    data,
    newShape,
    newStrides,
    0,
    'row-major'
  );

  return ArrayStorage.fromStdlib(stdlibArray, dtype);
}
