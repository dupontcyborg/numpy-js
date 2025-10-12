/**
 * NDArray - NumPy-compatible multidimensional array
 *
 * Thin wrapper around @stdlib/ndarray providing NumPy-like API
 */

import stdlib_ndarray from '@stdlib/ndarray';
import dgemm from '@stdlib/blas/base/dgemm';
import stdlib_slice from '@stdlib/ndarray/slice';
import Slice from '@stdlib/slice/ctor';
import {
  computeBroadcastShape,
  broadcastStdlibArrays,
  broadcastErrorMessage,
} from './broadcasting';
import { parseSlice, normalizeSlice } from './slicing';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type StdlibNDArray = any; // @stdlib types not available yet

export class NDArray {
  // Internal @stdlib ndarray
  private _data: StdlibNDArray;

  constructor(stdlibArray: StdlibNDArray) {
    this._data = stdlibArray;
  }

  // NumPy properties
  get shape(): readonly number[] {
    return Array.from(stdlib_ndarray.shape(this._data));
  }

  get ndim(): number {
    return stdlib_ndarray.ndims(this._data);
  }

  get size(): number {
    return stdlib_ndarray.numel(this._data);
  }

  get dtype(): string {
    return stdlib_ndarray.dtype(this._data);
  }

  get data(): TypedArray {
    return stdlib_ndarray.dataBuffer(this._data);
  }

  get strides(): readonly number[] {
    return Array.from(stdlib_ndarray.strides(this._data));
  }

  /**
   * Helper method for element-wise operations with broadcasting
   * @private
   */
  private _elementwiseOp(
    other: NDArray,
    op: (a: number, b: number) => number,
    opName: string
  ): NDArray {
    // Check if shapes are broadcast-compatible
    const outputShape = computeBroadcastShape([this.shape, other.shape]);

    if (outputShape === null) {
      throw new Error(broadcastErrorMessage([this.shape, other.shape], opName));
    }

    // Broadcast both arrays to the output shape
    const [broadcastThis, broadcastOther] = broadcastStdlibArrays([this._data, other._data]);

    // Create result array with the output shape
    const result = zeros(outputShape);
    const resultData = result.data;

    // Perform element-wise operation
    // Both broadcast arrays now have the same shape and size
    const size = stdlib_ndarray.numel(broadcastThis);

    for (let i = 0; i < size; i++) {
      // Use iget to handle strided access correctly
      const a = broadcastThis.iget(i);
      const b = broadcastOther.iget(i);
      resultData[i] = op(a, b);
    }

    return result;
  }

  // Arithmetic operations
  add(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      // Scalar addition
      const result = zeros(Array.from(this.shape));
      const thisData = this.data;
      const resultData = result.data;
      for (let i = 0; i < this.size; i++) {
        resultData[i] = thisData[i]! + other;
      }
      return result;
    } else {
      // Array addition with broadcasting
      return this._elementwiseOp(other, (a, b) => a + b, 'add');
    }
  }

  subtract(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      const result = zeros(Array.from(this.shape));
      const thisData = this.data;
      const resultData = result.data;
      for (let i = 0; i < this.size; i++) {
        resultData[i] = thisData[i]! - other;
      }
      return result;
    } else {
      // Array subtraction with broadcasting
      return this._elementwiseOp(other, (a, b) => a - b, 'subtract');
    }
  }

  multiply(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      const result = zeros(Array.from(this.shape));
      const thisData = this.data;
      const resultData = result.data;
      for (let i = 0; i < this.size; i++) {
        resultData[i] = thisData[i]! * other;
      }
      return result;
    } else {
      // Array multiplication with broadcasting
      return this._elementwiseOp(other, (a, b) => a * b, 'multiply');
    }
  }

  divide(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      const result = zeros(Array.from(this.shape));
      const thisData = this.data;
      const resultData = result.data;
      for (let i = 0; i < this.size; i++) {
        resultData[i] = thisData[i]! / other;
      }
      return result;
    } else {
      // Array division with broadcasting
      return this._elementwiseOp(other, (a, b) => a / b, 'divide');
    }
  }

  // Reductions
  sum(axis?: number): NDArray | number {
    if (axis === undefined) {
      // Sum all elements
      const data = this.data;
      let total = 0;
      for (let i = 0; i < this.size; i++) {
        total += data[i]!;
      }
      return total;
    } else {
      throw new Error('sum with axis not yet implemented');
    }
  }

  mean(axis?: number): NDArray | number {
    if (axis === undefined) {
      return (this.sum() as number) / this.size;
    } else {
      throw new Error('mean with axis not yet implemented');
    }
  }

  max(axis?: number): NDArray | number {
    if (axis === undefined) {
      const data = this.data;
      if (this.size === 0) {
        throw new Error('max of empty array');
      }
      let maxVal = data[0]!;
      for (let i = 1; i < this.size; i++) {
        if (data[i]! > maxVal) {
          maxVal = data[i]!;
        }
      }
      return maxVal;
    } else {
      throw new Error('max with axis not yet implemented');
    }
  }

  min(axis?: number): NDArray | number {
    if (axis === undefined) {
      const data = this.data;
      if (this.size === 0) {
        throw new Error('min of empty array');
      }
      let minVal = data[0]!;
      for (let i = 1; i < this.size; i++) {
        if (data[i]! < minVal) {
          minVal = data[i]!;
        }
      }
      return minVal;
    } else {
      throw new Error('min with axis not yet implemented');
    }
  }

  // Matrix multiplication
  matmul(other: NDArray): NDArray {
    if (this.ndim !== 2 || other.ndim !== 2) {
      throw new Error('matmul requires 2D arrays');
    }

    const [m = 0, k = 0] = this.shape;
    const [k2 = 0, n = 0] = other.shape;

    if (k !== k2) {
      throw new Error(`matmul shape mismatch: (${m},${k}) @ (${k2},${n})`);
    }

    // Create result array
    const result = zeros([m, n]);

    // Use @stdlib dgemm
    dgemm(
      'row-major',
      'no-transpose',
      'no-transpose',
      m,
      n,
      k,
      1.0, // alpha
      this.data as Float64Array,
      k, // lda
      other.data as Float64Array,
      n, // ldb
      0.0, // beta
      result.data as Float64Array,
      n // ldc
    );

    return result;
  }

  // Slicing
  /**
   * Slice the array using NumPy-style string syntax
   *
   * @param sliceStrs - Slice specifications, one per dimension
   * @returns Sliced view of the array
   *
   * @example
   * ```typescript
   * const arr = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
   * const row = arr.slice('0', ':');     // First row: [1, 2, 3]
   * const col = arr.slice(':', '1');     // Second column: [2, 5, 8]
   * const sub = arr.slice('0:2', '1:3'); // Top-right 2x2: [[2, 3], [5, 6]]
   * ```
   */
  slice(...sliceStrs: string[]): NDArray {
    if (sliceStrs.length === 0) {
      // No slices provided, return full array
      return this;
    }

    if (sliceStrs.length > this.ndim) {
      throw new Error(
        `Too many indices for array: array is ${this.ndim}-dimensional, but ${sliceStrs.length} were indexed`
      );
    }

    // Parse slice strings and normalize them
    const sliceSpecs = sliceStrs.map((str, i) => {
      const spec = parseSlice(str);
      const normalized = normalizeSlice(spec, this.shape[i]!);
      return normalized;
    });

    // Pad with full slices for remaining dimensions
    while (sliceSpecs.length < this.ndim) {
      sliceSpecs.push({
        start: 0,
        stop: this.shape[sliceSpecs.length]!,
        step: 1,
        isIndex: false,
      });
    }

    // Convert to @stdlib Slice objects
    const stdlibSlices = sliceSpecs.map((spec, i) => {
      if (spec.isIndex) {
        // Single index - use integer directly
        return spec.start;
      } else {
        // Slice - create Slice object
        // @stdlib Slice constructor: new Slice(start, stop, step)
        // null means use default (beginning/end)

        // For positive step: null start = 0, null stop = size
        // For negative step: null start = size-1, null stop = -1
        let start: number | null = spec.start;
        let stop: number | null = spec.stop;

        if (spec.step > 0) {
          // Forward slice
          if (start === 0) start = null;
          if (stop === this.shape[i]!) stop = null;
        } else {
          // Backward slice
          if (start === this.shape[i]! - 1) start = null;
          if (stop === -1) stop = null;
        }

        return new Slice(start, stop, spec.step);
      }
    });

    // Use @stdlib's slice function
    const result = stdlib_slice(this._data, ...stdlibSlices);
    return new NDArray(result);
  }

  /**
   * Get a single row (convenience method)
   * @param i - Row index
   * @returns Row as 1D or (n-1)D array
   */
  row(i: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('row() requires at least 2 dimensions');
    }
    return this.slice(String(i), ':');
  }

  /**
   * Get a single column (convenience method)
   * @param j - Column index
   * @returns Column as 1D or (n-1)D array
   */
  col(j: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('col() requires at least 2 dimensions');
    }
    return this.slice(':', String(j));
  }

  /**
   * Get a range of rows (convenience method)
   * @param start - Start row index
   * @param stop - Stop row index (exclusive)
   * @returns Rows as array
   */
  rows(start: number, stop: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('rows() requires at least 2 dimensions');
    }
    return this.slice(`${start}:${stop}`, ':');
  }

  /**
   * Get a range of columns (convenience method)
   * @param start - Start column index
   * @param stop - Stop column index (exclusive)
   * @returns Columns as array
   */
  cols(start: number, stop: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('cols() requires at least 2 dimensions');
    }
    return this.slice(':', `${start}:${stop}`);
  }

  // String representation
  toString(): string {
    return `NDArray(shape=${JSON.stringify(this.shape)}, dtype=${this.dtype})`;
  }

  // Convert to nested JavaScript array
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  toArray(): any {
    // Handle 0-dimensional arrays (scalars)
    if (this.ndim === 0) {
      // For 0-d arrays, return the scalar value directly
      return this._data.get();
    }
    // Use @stdlib's built-in conversion
    const arr = stdlib_ndarray.ndarray2array(this._data);
    return arr;
  }
}

// Type alias for TypedArray
type TypedArray =
  | Float64Array
  | Float32Array
  | Int32Array
  | Int16Array
  | Int8Array
  | Uint32Array
  | Uint16Array
  | Uint8Array
  | Uint8ClampedArray;

/**
 * Create array of zeros
 */
export function zeros(shape: number[]): NDArray {
  const stdlibArray = stdlib_ndarray.zeros(shape);
  return new NDArray(stdlibArray);
}

/**
 * Create array of ones
 */
export function ones(shape: number[]): NDArray {
  // @stdlib doesn't have ones() - create zeros and fill with 1
  const stdlibArray = stdlib_ndarray.zeros(shape);
  const data = stdlib_ndarray.dataBuffer(stdlibArray) as TypedArray;
  data.fill(1);
  return new NDArray(stdlibArray);
}

/**
 * Create array from nested JavaScript arrays
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function array(data: any): NDArray {
  const stdlibArray = stdlib_ndarray.array(data);
  return new NDArray(stdlibArray);
}

/**
 * Create array with evenly spaced values within a given interval
 * Similar to Python's range() but returns floats
 */
export function arange(start: number, stop?: number, step: number = 1): NDArray {
  // Handle single argument: arange(stop) -> arange(0, stop, 1)
  let actualStart = start;
  let actualStop = stop;

  if (stop === undefined) {
    actualStart = 0;
    actualStop = start;
  }

  if (actualStop === undefined) {
    throw new Error('stop is required');
  }

  // Calculate length
  const length = Math.max(0, Math.ceil((actualStop - actualStart) / step));

  // Create array
  const data = new Float64Array(length);
  for (let i = 0; i < length; i++) {
    data[i] = actualStart + i * step;
  }

  const stdlibArray = stdlib_ndarray.ndarray('float64', data, [length], [1], 0, 'row-major');
  return new NDArray(stdlibArray);
}

/**
 * Create array with evenly spaced values over a specified interval
 */
export function linspace(start: number, stop: number, num: number = 50): NDArray {
  if (num < 0) {
    throw new Error('num must be non-negative');
  }

  if (num === 0) {
    return array([]);
  }

  if (num === 1) {
    return array([start]);
  }

  const data = new Float64Array(num);
  const step = (stop - start) / (num - 1);

  for (let i = 0; i < num; i++) {
    data[i] = start + i * step;
  }

  const stdlibArray = stdlib_ndarray.ndarray('float64', data, [num], [1], 0, 'row-major');
  return new NDArray(stdlibArray);
}

/**
 * Create identity matrix
 */
export function eye(n: number, m?: number, k: number = 0): NDArray {
  const cols = m ?? n;
  const result = zeros([n, cols]);
  const data = result.data as Float64Array;

  // Fill diagonal
  for (let i = 0; i < n; i++) {
    const j = i + k;
    if (j >= 0 && j < cols) {
      data[i * cols + j] = 1;
    }
  }

  return result;
}
