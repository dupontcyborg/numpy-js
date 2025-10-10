/**
 * NDArray - NumPy-compatible multidimensional array
 *
 * Thin wrapper around @stdlib/ndarray providing NumPy-like API
 */

import stdlib_ndarray from '@stdlib/ndarray';
import dgemm from '@stdlib/blas/base/dgemm';

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
      // Array addition (TODO: broadcasting)
      if (this.size !== other.size) {
        throw new Error(
          'add: arrays must have same size for now (broadcasting not yet implemented)'
        );
      }
      const result = zeros(Array.from(this.shape));
      const thisData = this.data;
      const otherData = other.data;
      const resultData = result.data;
      for (let i = 0; i < this.size; i++) {
        resultData[i] = thisData[i]! + otherData[i]!;
      }
      return result;
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
      if (this.size !== other.size) {
        throw new Error(
          'subtract: arrays must have same size for now (broadcasting not yet implemented)'
        );
      }
      const result = zeros(Array.from(this.shape));
      const thisData = this.data;
      const otherData = other.data;
      const resultData = result.data;
      for (let i = 0; i < this.size; i++) {
        resultData[i] = thisData[i]! - otherData[i]!;
      }
      return result;
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
      if (this.size !== other.size) {
        throw new Error(
          'multiply: arrays must have same size for now (broadcasting not yet implemented)'
        );
      }
      const result = zeros(Array.from(this.shape));
      const thisData = this.data;
      const otherData = other.data;
      const resultData = result.data;
      for (let i = 0; i < this.size; i++) {
        resultData[i] = thisData[i]! * otherData[i]!;
      }
      return result;
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
      if (this.size !== other.size) {
        throw new Error(
          'divide: arrays must have same size for now (broadcasting not yet implemented)'
        );
      }
      const result = zeros(Array.from(this.shape));
      const thisData = this.data;
      const otherData = other.data;
      const resultData = result.data;
      for (let i = 0; i < this.size; i++) {
        resultData[i] = thisData[i]! / otherData[i]!;
      }
      return result;
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

  // String representation
  toString(): string {
    return `NDArray(shape=${JSON.stringify(this.shape)}, dtype=${this.dtype})`;
  }

  // Convert to nested JavaScript array
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  toArray(): any {
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
