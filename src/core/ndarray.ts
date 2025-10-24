/**
 * NDArray - NumPy-compatible multidimensional array
 *
 * Thin wrapper around @stdlib/ndarray providing NumPy-like API
 */

import stdlib_ndarray from '@stdlib/ndarray';
import stdlib_slice from '@stdlib/ndarray/slice';
import Slice from '@stdlib/slice/ctor';
import { parseSlice, normalizeSlice } from './slicing';
import {
  type DType,
  type TypedArray,
  DEFAULT_DTYPE,
  getTypedArrayConstructor,
  isBigIntDType,
  toStdlibDType,
} from './dtype';
import { ArrayStorage } from './storage';
import * as arithmeticOps from '../ops/arithmetic';
import * as comparisonOps from '../ops/comparison';
import * as reductionOps from '../ops/reduction';
import * as shapeOps from '../ops/shape';
import * as linalgOps from '../ops/linalg';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type StdlibNDArray = any; // @stdlib types not available yet

export class NDArray {
  // Internal @stdlib ndarray
  private _data: StdlibNDArray;
  // Store our actual dtype (since @stdlib may store a mapped dtype)
  private _dtype?: DType;
  // Track if this array is a view of another array
  private _base?: NDArray;

  constructor(stdlibArray: StdlibNDArray, dtype?: DType, base?: NDArray) {
    this._data = stdlibArray;
    this._dtype = dtype;
    this._base = base;
  }

  /**
   * Get internal storage (for ops modules)
   * @internal
   */
  get _storage(): ArrayStorage {
    return new ArrayStorage(this._data, this._dtype);
  }

  /**
   * Create NDArray from storage (for ops modules)
   * @internal
   */
  static _fromStorage(storage: ArrayStorage, base?: NDArray): NDArray {
    return new NDArray(storage.stdlib, storage.dtype, base);
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
    // Return our stored dtype if available, otherwise get from stdlib
    return this._dtype || stdlib_ndarray.dtype(this._data);
  }

  get data(): TypedArray {
    return stdlib_ndarray.dataBuffer(this._data);
  }

  get strides(): readonly number[] {
    return Array.from(stdlib_ndarray.strides(this._data));
  }

  /**
   * Array flags (similar to NumPy's flags)
   * Provides information about memory layout
   */
  get flags(): {
    C_CONTIGUOUS: boolean;
    F_CONTIGUOUS: boolean;
    OWNDATA: boolean;
  } {
    const storage = this._storage;
    return {
      C_CONTIGUOUS: storage.isCContiguous,
      F_CONTIGUOUS: storage.isFContiguous,
      OWNDATA: this._base === undefined, // True if we own data, false if we're a view
    };
  }

  /**
   * Base array if this is a view, null if this array owns its data
   * Similar to NumPy's base attribute
   */
  get base(): NDArray | null {
    return this._base ?? null;
  }

  /**
   * Get a single element from the array
   * @param indices - Array of indices, one per dimension (e.g., [0, 1] for 2D array)
   * @returns The element value (BigInt for int64/uint64, number otherwise)
   *
   * @example
   * ```typescript
   * const arr = array([[1, 2], [3, 4]]);
   * arr.get([0, 1]);  // Returns 2
   * arr.get([-1, -1]); // Returns 4 (negative indexing supported)
   * ```
   */
  get(indices: number[]): number | bigint {
    // Validate number of indices
    if (indices.length !== this.ndim) {
      throw new Error(
        `Index has ${indices.length} dimensions, but array has ${this.ndim} dimensions`
      );
    }

    // Normalize negative indices
    const normalizedIndices = indices.map((idx, dim) => {
      let normalized = idx;
      if (normalized < 0) {
        normalized = this.shape[dim]! + normalized;
      }
      // Validate bounds
      if (normalized < 0 || normalized >= this.shape[dim]!) {
        throw new Error(
          `Index ${idx} is out of bounds for axis ${dim} with size ${this.shape[dim]}`
        );
      }
      return normalized;
    });

    // Use @stdlib's get method
    const value = this._data.get(...normalizedIndices);

    // Return proper JS type based on dtype
    const currentDtype = this.dtype as DType;
    if (isBigIntDType(currentDtype)) {
      // For BigInt dtypes, ensure we return a BigInt
      return typeof value === 'bigint' ? value : BigInt(Math.round(value));
    }

    // For all other dtypes, return as number
    return Number(value);
  }

  /**
   * Set a single element in the array
   * @param indices - Array of indices, one per dimension (e.g., [0, 1] for 2D array)
   * @param value - Value to set (will be converted to array's dtype)
   *
   * @example
   * ```typescript
   * const arr = zeros([2, 3]);
   * arr.set([0, 1], 42);  // Set element at position [0, 1] to 42
   * arr.set([-1, -1], 99); // Set last element to 99 (negative indexing supported)
   * ```
   */
  set(indices: number[], value: number | bigint): void {
    // Validate number of indices
    if (indices.length !== this.ndim) {
      throw new Error(
        `Index has ${indices.length} dimensions, but array has ${this.ndim} dimensions`
      );
    }

    // Normalize negative indices
    const normalizedIndices = indices.map((idx, dim) => {
      let normalized = idx;
      if (normalized < 0) {
        normalized = this.shape[dim]! + normalized;
      }
      // Validate bounds
      if (normalized < 0 || normalized >= this.shape[dim]!) {
        throw new Error(
          `Index ${idx} is out of bounds for axis ${dim} with size ${this.shape[dim]}`
        );
      }
      return normalized;
    });

    // Convert value to appropriate type based on dtype
    const currentDtype = this.dtype as DType;
    let convertedValue: number | bigint;

    if (isBigIntDType(currentDtype)) {
      // Convert to BigInt for BigInt dtypes
      convertedValue = typeof value === 'bigint' ? value : BigInt(Math.round(value));
    } else if (currentDtype === 'bool') {
      // Convert to 0 or 1 for bool dtype
      convertedValue = value ? 1 : 0;
    } else {
      // Convert to number for all other dtypes
      convertedValue = Number(value);
    }

    // Use @stdlib's set method
    this._data.set(...normalizedIndices, convertedValue);
  }

  /**
   * Return a deep copy of the array
   * Creates a new array with copied data (not a view)
   *
   * @returns Deep copy of the array
   *
   * @example
   * ```typescript
   * const arr = array([[1, 2], [3, 4]]);
   * const copied = arr.copy();
   * copied.set([0, 0], 99);  // Doesn't affect original
   * console.log(arr.get([0, 0]));  // Still 1
   * ```
   */
  copy(): NDArray {
    const currentDtype = this.dtype as DType;
    const shape = Array.from(this.shape);
    const data = this.data;

    // Get TypedArray constructor
    const Constructor = getTypedArrayConstructor(currentDtype);
    if (!Constructor) {
      throw new Error(`Cannot copy array with dtype ${currentDtype}`);
    }

    // Create new data buffer and copy
    if (isBigIntDType(currentDtype)) {
      const newData = new Constructor(this.size) as BigInt64Array | BigUint64Array;
      const typedData = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < this.size; i++) {
        newData[i] = typedData[i]!;
      }

      const stdlibArray = stdlib_ndarray.ndarray(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        toStdlibDType(currentDtype) as any,
        newData,
        shape,
        computeStrides(shape),
        0,
        'row-major'
      );
      return new NDArray(stdlibArray, currentDtype);
    } else {
      // For all other types, use TypedArray.slice() or set()
      const newData = new Constructor(this.size);
      const typedData = data as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>).set(typedData);

      const stdlibArray = stdlib_ndarray.ndarray(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        toStdlibDType(currentDtype) as any,
        newData,
        shape,
        computeStrides(shape),
        0,
        'row-major'
      );
      return new NDArray(stdlibArray, currentDtype);
    }
  }

  /**
   * Cast array to a different dtype
   * @param dtype - Target dtype
   * @param copy - If false and dtype matches, return self; otherwise create copy (default: true)
   * @returns Array with specified dtype
   */
  astype(dtype: DType, copy: boolean = true): NDArray {
    const currentDtype = this.dtype as DType;

    // If dtype matches and copy=false, return self
    if (currentDtype === dtype && !copy) {
      return this;
    }

    // If dtype matches and copy=true, create a copy
    if (currentDtype === dtype && copy) {
      // Use array() to create a copy
      return array(this.toArray(), dtype);
    }

    // Need to convert dtype
    const shape = Array.from(this.shape);
    const size = this.size;

    // Get TypedArray constructor for conversion
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot convert to dtype ${dtype}`);
    }
    const newData = new Constructor(size);
    const oldData = this.data;

    // Handle BigInt to other types
    if (isBigIntDType(currentDtype) && !isBigIntDType(dtype)) {
      const typedOldData = oldData as BigInt64Array | BigUint64Array;
      if (dtype === 'bool') {
        for (let i = 0; i < size; i++) {
          (newData as Uint8Array)[i] = typedOldData[i] !== BigInt(0) ? 1 : 0;
        }
      } else {
        for (let i = 0; i < size; i++) {
          (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = Number(
            typedOldData[i]
          );
        }
      }
    }
    // Handle other types to BigInt
    else if (!isBigIntDType(currentDtype) && isBigIntDType(dtype)) {
      const typedOldData = oldData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < size; i++) {
        (newData as BigInt64Array | BigUint64Array)[i] = BigInt(
          Math.round(Number(typedOldData[i]))
        );
      }
    }
    // Handle other types to bool
    else if (dtype === 'bool') {
      const typedOldData = oldData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < size; i++) {
        (newData as Uint8Array)[i] = typedOldData[i] !== 0 ? 1 : 0;
      }
    }
    // Handle bool to other types
    else if (currentDtype === 'bool' && !isBigIntDType(dtype)) {
      const typedOldData = oldData as Uint8Array;
      for (let i = 0; i < size; i++) {
        (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = typedOldData[i]!;
      }
    }
    // Handle regular numeric conversions
    else if (!isBigIntDType(currentDtype) && !isBigIntDType(dtype)) {
      const typedOldData = oldData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < size; i++) {
        (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = typedOldData[i]!;
      }
    }
    // Handle BigInt to BigInt conversions (int64 <-> uint64)
    else {
      const typedOldData = oldData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        (newData as BigInt64Array | BigUint64Array)[i] = typedOldData[i]!;
      }
    }

    const stdlibArray = stdlib_ndarray.ndarray(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      toStdlibDType(dtype) as any,
      newData,
      shape,
      computeStrides(shape),
      0,
      'row-major'
    );
    return new NDArray(stdlibArray, dtype);
  }

  /**
   * Helper method for element-wise operations with broadcasting
   * @private
   */
  // Arithmetic operations
  add(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.add(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  subtract(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.subtract(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  multiply(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.multiply(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  divide(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.divide(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  // Comparison operations
  /**
   * Element-wise greater than comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  greater(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.greater(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise greater than or equal comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  greater_equal(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.greaterEqual(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise less than comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  less(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.less(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise less than or equal comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  less_equal(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.lessEqual(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise equality comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  equal(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.equal(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise not equal comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  not_equal(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.notEqual(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise comparison with tolerance
   * Returns True where |a - b| <= (atol + rtol * |b|)
   * @param other - Value or array to compare with
   * @param rtol - Relative tolerance (default: 1e-5)
   * @param atol - Absolute tolerance (default: 1e-8)
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  isclose(other: NDArray | number, rtol: number = 1e-5, atol: number = 1e-8): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.isclose(this._storage, otherStorage, rtol, atol);
    return NDArray._fromStorage(resultStorage);
  }

  allclose(other: NDArray | number, rtol: number = 1e-5, atol: number = 1e-8): boolean {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    return comparisonOps.allclose(this._storage, otherStorage, rtol, atol);
  }

  /**
   * Helper method for isclose operation with broadcasting
   * @private
   */
  /**
   * Helper method for comparison operations with broadcasting
   * @private
   */
  // Reductions
  /**
   * Sum array elements over a given axis
   * @param axis - Axis along which to sum. If undefined, sum all elements
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Sum of array elements, or array of sums along axis
   */
  sum(axis?: number, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.sum(this._storage, axis, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Compute the arithmetic mean along the specified axis
   * @param axis - Axis along which to compute mean. If undefined, compute mean of all elements
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Mean of array elements, or array of means along axis
   *
   * Note: mean() returns float64 for integer dtypes, matching NumPy behavior
   */
  mean(axis?: number, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.mean(this._storage, axis, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Return the maximum along a given axis
   * @param axis - Axis along which to compute maximum. If undefined, compute maximum of all elements
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Maximum of array elements, or array of maximums along axis
   */
  max(axis?: number, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.max(this._storage, axis, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Return the minimum along a given axis
   * @param axis - Axis along which to compute minimum. If undefined, compute minimum of all elements
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Minimum of array elements, or array of minimums along axis
   */
  min(axis?: number, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.min(this._storage, axis, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Product of array elements over a given axis
   * @param axis - Axis along which to compute the product. If undefined, product of all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Product of array elements, or array of products along axis
   */
  prod(axis?: number, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.prod(this._storage, axis, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Indices of the minimum values along an axis
   * @param axis - Axis along which to find minimum indices. If undefined, index of global minimum.
   * @returns Indices of minimum values
   */
  argmin(axis?: number): NDArray | number {
    const result = reductionOps.argmin(this._storage, axis);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Indices of the maximum values along an axis
   * @param axis - Axis along which to find maximum indices. If undefined, index of global maximum.
   * @returns Indices of maximum values
   */
  argmax(axis?: number): NDArray | number {
    const result = reductionOps.argmax(this._storage, axis);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Compute variance along the specified axis
   * @param axis - Axis along which to compute variance. If undefined, variance of all elements.
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Variance of array elements
   */
  var(axis?: number, ddof: number = 0, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.variance(this._storage, axis, ddof, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Compute standard deviation along the specified axis
   * @param axis - Axis along which to compute std. If undefined, std of all elements.
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Standard deviation of array elements
   */
  std(axis?: number, ddof: number = 0, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.std(this._storage, axis, ddof, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Test whether all array elements along a given axis evaluate to True
   * @param axis - Axis along which to perform logical AND. If undefined, test all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Boolean or array of booleans
   */
  all(axis?: number, keepdims: boolean = false): NDArray | boolean {
    const result = reductionOps.all(this._storage, axis, keepdims);
    return typeof result === 'boolean' ? result : NDArray._fromStorage(result);
  }

  /**
   * Test whether any array elements along a given axis evaluate to True
   * @param axis - Axis along which to perform logical OR. If undefined, test all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Boolean or array of booleans
   */
  any(axis?: number, keepdims: boolean = false): NDArray | boolean {
    const result = reductionOps.any(this._storage, axis, keepdims);
    return typeof result === 'boolean' ? result : NDArray._fromStorage(result);
  }

  // Shape manipulation
  /**
   * Reshape array to a new shape
   * Returns a new array with the specified shape
   * @param shape - New shape (must be compatible with current size)
   * @returns Reshaped array
   */
  reshape(...shape: number[]): NDArray {
    // Flatten the input if it's a nested array (e.g., reshape([2, 3]) or reshape(2, 3))
    const newShape = shape.length === 1 && Array.isArray(shape[0]) ? shape[0] : shape;
    const resultStorage = shapeOps.reshape(this._storage, newShape);
    // Check if result shares same data buffer (view) or is a copy
    const isView = resultStorage.data === this.data;
    const base = isView ? (this._base ?? this) : undefined;
    return NDArray._fromStorage(resultStorage, base);
  }

  /**
   * Return a flattened copy of the array
   * @returns 1D array containing all elements
   */
  flatten(): NDArray {
    const resultStorage = shapeOps.flatten(this._storage);
    // flatten() always creates a copy, so no base tracking
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Return a flattened array (view when possible, otherwise copy)
   * @returns 1D array containing all elements
   */
  ravel(): NDArray {
    const resultStorage = shapeOps.ravel(this._storage);
    // Check if result shares same data buffer (view) or is a copy
    const isView = resultStorage.data === this.data;
    const base = isView ? (this._base ?? this) : undefined;
    return NDArray._fromStorage(resultStorage, base);
  }

  /**
   * Transpose array (permute dimensions)
   * @param axes - Permutation of axes. If undefined, reverse the dimensions
   * @returns Transposed array (always a view)
   */
  transpose(axes?: number[]): NDArray {
    const resultStorage = shapeOps.transpose(this._storage, axes);
    // transpose() always creates a view
    const base = this._base ?? this;
    return NDArray._fromStorage(resultStorage, base);
  }

  /**
   * Remove axes of length 1
   * @param axis - Axis to squeeze. If undefined, squeeze all axes of length 1
   * @returns Array with specified dimensions removed (always a view)
   */
  squeeze(axis?: number): NDArray {
    const resultStorage = shapeOps.squeeze(this._storage, axis);
    // squeeze() always creates a view
    const base = this._base ?? this;
    return NDArray._fromStorage(resultStorage, base);
  }

  /**
   * Expand the shape by inserting a new axis of length 1
   * @param axis - Position where new axis is placed
   * @returns Array with additional dimension (always a view)
   */
  expand_dims(axis: number): NDArray {
    const resultStorage = shapeOps.expandDims(this._storage, axis);
    // expand_dims() always creates a view
    const base = this._base ?? this;
    return NDArray._fromStorage(resultStorage, base);
  }

  // Matrix multiplication
  matmul(other: NDArray): NDArray {
    const resultStorage = linalgOps.matmul(this._storage, other._storage);
    return NDArray._fromStorage(resultStorage);
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
      // No slices provided, return the same array (not a new view)
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

        // @ts-expect-error - stdlib Slice type definitions don't match actual behavior
        return new Slice(start, stop, spec.step);
      }
    });

    // Use @stdlib's slice function
    const result = stdlib_slice(this._data, ...stdlibSlices);
    // slice() always creates a view
    const base = this._base ?? this;
    return new NDArray(result, this._dtype, base);
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

// Note: TypedArray is imported from './dtype'

/**
 * Create array of zeros
 */
export function zeros(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArray {
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create zeros array with dtype ${dtype}`);
  }
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Constructor(size);
  const stdlibArray = stdlib_ndarray.ndarray(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toStdlibDType(dtype) as any, // Cast to any since stdlib doesn't support all our dtypes
    data,
    shape,
    computeStrides(shape),
    0,
    'row-major'
  );
  return new NDArray(stdlibArray, dtype);
}

/**
 * Helper function to compute row-major strides
 */
function computeStrides(shape: number[]): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i]!;
  }
  return strides;
}

/**
 * Create array of ones
 */
export function ones(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArray {
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create ones array with dtype ${dtype}`);
  }
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Constructor(size);

  // Handle BigInt types
  if (isBigIntDType(dtype)) {
    (data as BigInt64Array | BigUint64Array).fill(BigInt(1));
  } else {
    (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>).fill(1);
  }

  const stdlibArray = stdlib_ndarray.ndarray(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toStdlibDType(dtype) as any, // Cast to any since stdlib doesn't support all our dtypes
    data,
    shape,
    computeStrides(shape),
    0,
    'row-major'
  );
  return new NDArray(stdlibArray, dtype);
}

/**
 * Create array from nested JavaScript arrays
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function array(data: any, dtype?: DType): NDArray {
  // If no dtype specified, use stdlib's default inference
  if (dtype === undefined) {
    const stdlibArray = stdlib_ndarray.array(data);
    return new NDArray(stdlibArray);
  }

  // With explicit dtype, create via stdlib then convert if needed
  const stdlibArray = stdlib_ndarray.array(data);
  const currentDtype = stdlib_ndarray.dtype(stdlibArray) as DType;

  // If dtype matches, return as-is
  if (currentDtype === dtype) {
    return new NDArray(stdlibArray);
  }

  // Need to convert dtype
  const shape = Array.from(stdlib_ndarray.shape(stdlibArray));
  const size = shape.reduce((a, b) => a * b, 1);

  // Get TypedArray constructor for conversion
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create array with dtype ${dtype}`);
  }
  const newData = new Constructor(size);
  const oldData = stdlib_ndarray.dataBuffer(stdlibArray);

  // Convert values
  if (isBigIntDType(dtype)) {
    for (let i = 0; i < size; i++) {
      (newData as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(Number(oldData[i])));
    }
  } else if (dtype === 'bool') {
    for (let i = 0; i < size; i++) {
      (newData as Uint8Array)[i] = oldData[i] ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = Number(oldData[i]);
    }
  }

  const newStdlibArray = stdlib_ndarray.ndarray(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toStdlibDType(dtype) as any, // Cast to any since stdlib doesn't support all our dtypes
    newData,
    shape,
    computeStrides(shape),
    0,
    'row-major'
  );
  return new NDArray(newStdlibArray, dtype);
}

/**
 * Create array with evenly spaced values within a given interval
 * Similar to Python's range() but returns floats
 */
export function arange(
  start: number,
  stop?: number,
  step: number = 1,
  dtype: DType = DEFAULT_DTYPE
): NDArray {
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

  // Create array with specified dtype
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create arange array with dtype ${dtype}`);
  }

  const data = new Constructor(length);

  // Fill with values
  if (isBigIntDType(dtype)) {
    for (let i = 0; i < length; i++) {
      (data as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(actualStart + i * step));
    }
  } else if (dtype === 'bool') {
    for (let i = 0; i < length; i++) {
      (data as Uint8Array)[i] = actualStart + i * step !== 0 ? 1 : 0;
    }
  } else {
    for (let i = 0; i < length; i++) {
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = actualStart + i * step;
    }
  }

  const stdlibArray = stdlib_ndarray.ndarray(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toStdlibDType(dtype) as any,
    data,
    [length],
    [1],
    0,
    'row-major'
  );
  return new NDArray(stdlibArray, dtype);
}

/**
 * Create array with evenly spaced values over a specified interval
 */
export function linspace(
  start: number,
  stop: number,
  num: number = 50,
  dtype: DType = DEFAULT_DTYPE
): NDArray {
  if (num < 0) {
    throw new Error('num must be non-negative');
  }

  if (num === 0) {
    return array([], dtype);
  }

  if (num === 1) {
    return array([start], dtype);
  }

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create linspace array with dtype ${dtype}`);
  }

  const data = new Constructor(num);
  const step = (stop - start) / (num - 1);

  // Fill with values
  if (isBigIntDType(dtype)) {
    for (let i = 0; i < num; i++) {
      (data as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(start + i * step));
    }
  } else if (dtype === 'bool') {
    for (let i = 0; i < num; i++) {
      (data as Uint8Array)[i] = start + i * step !== 0 ? 1 : 0;
    }
  } else {
    for (let i = 0; i < num; i++) {
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = start + i * step;
    }
  }

  const stdlibArray = stdlib_ndarray.ndarray(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toStdlibDType(dtype) as any,
    data,
    [num],
    [1],
    0,
    'row-major'
  );
  return new NDArray(stdlibArray, dtype);
}

/**
 * Create array with logarithmically spaced values
 * Returns num samples, equally spaced on a log scale from base^start to base^stop
 */
export function logspace(
  start: number,
  stop: number,
  num: number = 50,
  base: number = 10.0,
  dtype: DType = DEFAULT_DTYPE
): NDArray {
  if (num < 0) {
    throw new Error('num must be non-negative');
  }

  if (num === 0) {
    return array([], dtype);
  }

  if (num === 1) {
    return array([Math.pow(base, start)], dtype);
  }

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create logspace array with dtype ${dtype}`);
  }

  const data = new Constructor(num);
  const step = (stop - start) / (num - 1);

  // Fill with logarithmically spaced values: base^(start + i*step)
  if (isBigIntDType(dtype)) {
    for (let i = 0; i < num; i++) {
      const exponent = start + i * step;
      (data as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(Math.pow(base, exponent)));
    }
  } else if (dtype === 'bool') {
    for (let i = 0; i < num; i++) {
      const exponent = start + i * step;
      (data as Uint8Array)[i] = Math.pow(base, exponent) !== 0 ? 1 : 0;
    }
  } else {
    for (let i = 0; i < num; i++) {
      const exponent = start + i * step;
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = Math.pow(base, exponent);
    }
  }

  const stdlibArray = stdlib_ndarray.ndarray(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toStdlibDType(dtype) as any,
    data,
    [num],
    [1],
    0,
    'row-major'
  );
  return new NDArray(stdlibArray, dtype);
}

/**
 * Create array with geometrically spaced values
 * Returns num samples, equally spaced on a log scale (geometric progression)
 */
export function geomspace(
  start: number,
  stop: number,
  num: number = 50,
  dtype: DType = DEFAULT_DTYPE
): NDArray {
  if (num < 0) {
    throw new Error('num must be non-negative');
  }

  if (start === 0 || stop === 0) {
    throw new Error('Geometric sequence cannot include zero');
  }

  if (num === 0) {
    return array([], dtype);
  }

  if (num === 1) {
    return array([start], dtype);
  }

  // For geometric progression, we need to handle negative values carefully
  // NumPy uses: sign * exp(linspace(log(abs(start)), log(abs(stop)), num))
  const signStart = Math.sign(start);
  const signStop = Math.sign(stop);

  if (signStart !== signStop) {
    throw new Error('Geometric sequence cannot contain both positive and negative values');
  }

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create geomspace array with dtype ${dtype}`);
  }

  const data = new Constructor(num);
  const logStart = Math.log(Math.abs(start));
  const logStop = Math.log(Math.abs(stop));
  const step = (logStop - logStart) / (num - 1);

  // Fill with geometrically spaced values
  if (isBigIntDType(dtype)) {
    for (let i = 0; i < num; i++) {
      const value = signStart * Math.exp(logStart + i * step);
      (data as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(value));
    }
  } else if (dtype === 'bool') {
    for (let i = 0; i < num; i++) {
      const value = signStart * Math.exp(logStart + i * step);
      (data as Uint8Array)[i] = value !== 0 ? 1 : 0;
    }
  } else {
    for (let i = 0; i < num; i++) {
      const value = signStart * Math.exp(logStart + i * step);
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value;
    }
  }

  const stdlibArray = stdlib_ndarray.ndarray(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toStdlibDType(dtype) as any,
    data,
    [num],
    [1],
    0,
    'row-major'
  );
  return new NDArray(stdlibArray, dtype);
}

/**
 * Create identity matrix
 */
export function eye(n: number, m?: number, k: number = 0, dtype: DType = DEFAULT_DTYPE): NDArray {
  const cols = m ?? n;
  const result = zeros([n, cols], dtype);
  const data = result.data;

  // Fill diagonal (the zeros() call already created the correct dtype, so data is pre-filled with 0)
  if (isBigIntDType(dtype)) {
    const typedData = data as unknown as BigInt64Array | BigUint64Array;
    for (let i = 0; i < n; i++) {
      const j = i + k;
      if (j >= 0 && j < cols) {
        typedData[i * cols + j] = BigInt(1);
      }
    }
  } else {
    const typedData = data as unknown as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
    for (let i = 0; i < n; i++) {
      const j = i + k;
      if (j >= 0 && j < cols) {
        typedData[i * cols + j] = 1;
      }
    }
  }

  return result;
}

/**
 * Create an uninitialized array
 * Note: Unlike NumPy, TypedArrays are zero-initialized by default in JavaScript
 */
export function empty(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArray {
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create empty array with dtype ${dtype}`);
  }
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Constructor(size);
  const stdlibArray = stdlib_ndarray.ndarray(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toStdlibDType(dtype) as any,
    data,
    shape,
    computeStrides(shape),
    0,
    'row-major'
  );
  return new NDArray(stdlibArray, dtype);
}

/**
 * Create array filled with a constant value
 */
export function full(
  shape: number[],
  fill_value: number | bigint | boolean,
  dtype?: DType
): NDArray {
  // Infer dtype from fill_value if not specified
  let actualDtype = dtype;
  if (!actualDtype) {
    if (typeof fill_value === 'bigint') {
      actualDtype = 'int64';
    } else if (typeof fill_value === 'boolean') {
      actualDtype = 'bool';
    } else if (Number.isInteger(fill_value)) {
      actualDtype = 'int32';
    } else {
      actualDtype = DEFAULT_DTYPE;
    }
  }

  const Constructor = getTypedArrayConstructor(actualDtype);
  if (!Constructor) {
    throw new Error(`Cannot create full array with dtype ${actualDtype}`);
  }
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Constructor(size);

  // Fill with value
  if (isBigIntDType(actualDtype)) {
    const bigintValue =
      typeof fill_value === 'bigint' ? fill_value : BigInt(Math.round(Number(fill_value)));
    (data as BigInt64Array | BigUint64Array).fill(bigintValue);
  } else if (actualDtype === 'bool') {
    (data as Uint8Array).fill(fill_value ? 1 : 0);
  } else {
    (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>).fill(Number(fill_value));
  }

  const stdlibArray = stdlib_ndarray.ndarray(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toStdlibDType(actualDtype) as any,
    data,
    shape,
    computeStrides(shape),
    0,
    'row-major'
  );
  return new NDArray(stdlibArray, actualDtype);
}

/**
 * Create a square identity matrix
 */
export function identity(n: number, dtype: DType = DEFAULT_DTYPE): NDArray {
  return eye(n, n, 0, dtype);
}

/**
 * Convert input to an ndarray
 * If input is already an NDArray, optionally convert dtype
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function asarray(a: NDArray | any, dtype?: DType): NDArray {
  // If already an NDArray
  if (a instanceof NDArray) {
    // If dtype matches or not specified, return as-is
    if (!dtype || a.dtype === dtype) {
      return a;
    }
    // Need to convert dtype - create new array from data
    const shape = Array.from(a.shape);
    const size = a.size;
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot create array with dtype ${dtype}`);
    }
    const newData = new Constructor(size);
    const oldData = a.data;

    // Convert values
    if (isBigIntDType(dtype)) {
      for (let i = 0; i < size; i++) {
        (newData as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(Number(oldData[i])));
      }
    } else if (dtype === 'bool') {
      for (let i = 0; i < size; i++) {
        (newData as Uint8Array)[i] = oldData[i] ? 1 : 0;
      }
    } else {
      for (let i = 0; i < size; i++) {
        (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = Number(oldData[i]);
      }
    }

    const newStdlibArray = stdlib_ndarray.ndarray(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      toStdlibDType(dtype) as any,
      newData,
      shape,
      computeStrides(shape),
      0,
      'row-major'
    );
    return new NDArray(newStdlibArray, dtype);
  }

  // Otherwise, use array() to create from data
  return array(a, dtype);
}

/**
 * Create a deep copy of an array
 */
export function copy(a: NDArray): NDArray {
  const shape = Array.from(a.shape);
  const dtype = a.dtype as DType;
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot copy array with dtype ${dtype}`);
  }

  const size = a.size;
  const oldData = a.data;
  const newData = new Constructor(size);

  // For C-contiguous arrays, we can just copy the buffer
  if (a.flags.C_CONTIGUOUS) {
    if (isBigIntDType(dtype)) {
      (newData as BigInt64Array | BigUint64Array).set(oldData as BigInt64Array | BigUint64Array);
    } else {
      (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>).set(
        oldData as Exclude<TypedArray, BigInt64Array | BigUint64Array>
      );
    }
  } else {
    // For non-contiguous arrays, iterate through elements
    // Convert flat index to multi-index and read using get()
    const shapeArr = Array.from(shape);
    const strides = computeStrides(shapeArr);
    for (let flatIdx = 0; flatIdx < size; flatIdx++) {
      // Convert flat index to multi-index
      const indices: number[] = [];
      let remaining = flatIdx;
      for (let dim = 0; dim < shapeArr.length; dim++) {
        const stride = strides[dim]!;
        const idx = Math.floor(remaining / stride);
        indices.push(idx);
        remaining -= idx * stride;
      }
      const value = a.get(indices);
      if (isBigIntDType(dtype)) {
        (newData as BigInt64Array | BigUint64Array)[flatIdx] = value as bigint;
      } else {
        (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[flatIdx] = value as number;
      }
    }
  }

  const stdlibArray = stdlib_ndarray.ndarray(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    toStdlibDType(dtype) as any,
    newData,
    shape,
    computeStrides(shape),
    0,
    'row-major'
  );
  return new NDArray(stdlibArray, dtype);
}

/**
 * Create array of zeros with the same shape as another array
 */
export function zeros_like(a: NDArray, dtype?: DType): NDArray {
  return zeros(Array.from(a.shape), dtype ?? (a.dtype as DType));
}

/**
 * Create array of ones with the same shape as another array
 */
export function ones_like(a: NDArray, dtype?: DType): NDArray {
  return ones(Array.from(a.shape), dtype ?? (a.dtype as DType));
}

/**
 * Create uninitialized array with the same shape as another array
 */
export function empty_like(a: NDArray, dtype?: DType): NDArray {
  return empty(Array.from(a.shape), dtype ?? (a.dtype as DType));
}

/**
 * Create array filled with a constant value, same shape as another array
 */
export function full_like(
  a: NDArray,
  fill_value: number | bigint | boolean,
  dtype?: DType
): NDArray {
  return full(Array.from(a.shape), fill_value, dtype ?? (a.dtype as DType));
}
