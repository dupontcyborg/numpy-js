/**
 * ArrayStorage - Internal storage abstraction
 *
 * Wraps @stdlib/ndarray to provide a clean internal interface
 * while keeping stdlib as an implementation detail.
 *
 * @internal - This is not part of the public API
 */

/// <reference types="@stdlib/types"/>

import ndarray from '@stdlib/ndarray/ctor';
import dtype from '@stdlib/ndarray/dtype';
import shape from '@stdlib/ndarray/shape';
import strides from '@stdlib/ndarray/strides';
import ndims from '@stdlib/ndarray/ndims';
import numel from '@stdlib/ndarray/numel';
import data from '@stdlib/ndarray/data-buffer';
import type { ndarray as StdlibNDArray } from '@stdlib/types/ndarray';
import {
  type DType,
  type TypedArray,
  DEFAULT_DTYPE,
  getTypedArrayConstructor,
  isBigIntDType,
  toStdlibDType,
} from './dtype';

/**
 * Internal storage for NDArray data
 * Manages the underlying TypedArray and metadata
 */
export class ArrayStorage {
  // Internal @stdlib ndarray
  private _stdlib: StdlibNDArray;
  // Store our actual dtype (since @stdlib may store a mapped dtype)
  private _dtype: DType;

  constructor(stdlibArray: StdlibNDArray, dt?: DType) {
    this._stdlib = stdlibArray;
    this._dtype = dt || (dtype(stdlibArray) as DType);
  }

  /**
   * Shape of the array
   */
  get shape(): readonly number[] {
    return Array.from(shape(this._stdlib));
  }

  /**
   * Number of dimensions
   */
  get ndim(): number {
    return ndims(this._stdlib);
  }

  /**
   * Total number of elements
   */
  get size(): number {
    return numel(this._stdlib);
  }

  /**
   * Data type
   */
  get dtype(): DType {
    return this._dtype;
  }

  /**
   * Underlying data buffer
   */
  get data(): TypedArray {
    return data(this._stdlib) as TypedArray;
  }

  /**
   * Strides (steps in each dimension)
   */
  get strides(): readonly number[] {
    return Array.from(strides(this._stdlib));
  }

  /**
   * Direct access to stdlib ndarray (for internal use)
   * @internal
   */
  get stdlib(): StdlibNDArray {
    return this._stdlib;
  }

  /**
   * Check if array is C-contiguous (row-major, no gaps)
   */
  get isCContiguous(): boolean {
    const shape = this.shape;
    const strides = this.strides;
    const ndim = shape.length;

    if (ndim === 0) return true;
    if (ndim === 1) return strides[0] === 1;

    // Check if strides match row-major order
    let expectedStride = 1;
    for (let i = ndim - 1; i >= 0; i--) {
      if (strides[i] !== expectedStride) return false;
      expectedStride *= shape[i]!;
    }
    return true;
  }

  /**
   * Check if array is F-contiguous (column-major, no gaps)
   */
  get isFContiguous(): boolean {
    const shape = this.shape;
    const strides = this.strides;
    const ndim = shape.length;

    if (ndim === 0) return true;
    if (ndim === 1) return strides[0] === 1;

    // Check if strides match column-major order
    let expectedStride = 1;
    for (let i = 0; i < ndim; i++) {
      if (strides[i] !== expectedStride) return false;
      expectedStride *= shape[i]!;
    }
    return true;
  }

  /**
   * Create a deep copy of this storage
   */
  copy(): ArrayStorage {
    const shape = Array.from(this.shape);
    const data = this.data;
    const dtype = this._dtype;

    // Get TypedArray constructor
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot copy array with dtype ${dtype}`);
    }

    // Create new data buffer and copy
    if (isBigIntDType(dtype)) {
      const newData = new Constructor(this.size) as BigInt64Array | BigUint64Array;
      const typedData = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < this.size; i++) {
        newData[i] = typedData[i]!;
      }

      const stdlibArray = ndarray(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        toStdlibDType(dtype) as any,
        newData,
        shape,
        this._computeStrides(shape),
        0,
        'row-major'
      );
      return new ArrayStorage(stdlibArray, dtype);
    } else {
      // For all other types, use TypedArray.slice() or set()
      const newData = new Constructor(this.size);
      const typedData = data as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>).set(typedData);

      const stdlibArray = ndarray(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        toStdlibDType(dtype) as any,
        newData,
        shape,
        this._computeStrides(shape),
        0,
        'row-major'
      );
      return new ArrayStorage(stdlibArray, dtype);
    }
  }

  /**
   * Create storage from stdlib ndarray
   */
  static fromStdlib(stdlibArray: StdlibNDArray, dtype?: DType): ArrayStorage {
    return new ArrayStorage(stdlibArray, dtype);
  }

  /**
   * Create storage from TypedArray data
   */
  static fromData(data: TypedArray, shape: number[], dtype: DType): ArrayStorage {
    const strides = ArrayStorage._computeStrides(shape);
    const stdlibArray = ndarray(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      toStdlibDType(dtype) as any,
      data,
      shape,
      strides,
      0,
      'row-major'
    );
    return new ArrayStorage(stdlibArray, dtype);
  }

  /**
   * Create storage with zeros
   */
  static zeros(shape: number[], dtype: DType = DEFAULT_DTYPE): ArrayStorage {
    const size = shape.reduce((a, b) => a * b, 1);

    // Get TypedArray constructor
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot create array with dtype ${dtype}`);
    }

    const data = new Constructor(size);
    const stdlibArray = ndarray(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      toStdlibDType(dtype) as any,
      data,
      shape,
      ArrayStorage._computeStrides(shape),
      0,
      'row-major'
    );

    return new ArrayStorage(stdlibArray, dtype);
  }

  /**
   * Create storage with ones
   */
  static ones(shape: number[], dtype: DType = DEFAULT_DTYPE): ArrayStorage {
    const storage = ArrayStorage.zeros(shape, dtype);
    const data = storage.data;

    // Fill with ones
    if (isBigIntDType(dtype)) {
      const typedData = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < storage.size; i++) {
        typedData[i] = BigInt(1);
      }
    } else {
      const typedData = data as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < storage.size; i++) {
        typedData[i] = 1;
      }
    }

    return storage;
  }

  /**
   * Compute strides for row-major (C-order) layout
   * @private
   */
  private _computeStrides(shape: readonly number[]): number[] {
    return ArrayStorage._computeStrides(shape);
  }

  /**
   * Compute strides for row-major (C-order) layout
   * @private
   */
  private static _computeStrides(shape: readonly number[]): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i]!;
    }
    return strides;
  }
}

/**
 * Compute strides for a given shape (row-major order)
 * @internal
 */
export function computeStrides(shape: readonly number[]): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i]!;
  }
  return strides;
}
