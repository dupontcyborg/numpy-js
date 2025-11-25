/**
 * NDArray - NumPy-compatible multidimensional array
 *
 * Core array class providing NumPy-like API
 */

import { parseSlice, normalizeSlice } from './slicing';
import {
  type DType,
  type TypedArray,
  DEFAULT_DTYPE,
  getTypedArrayConstructor,
  isBigIntDType,
} from './dtype';
import { ArrayStorage } from './storage';
import * as arithmeticOps from '../ops/arithmetic';
import * as comparisonOps from '../ops/comparison';
import * as reductionOps from '../ops/reduction';
import * as shapeOps from '../ops/shape';
import * as linalgOps from '../ops/linalg';
import * as exponentialOps from '../ops/exponential';

export class NDArray {
  // Internal storage
  private _storage: ArrayStorage;
  // Track if this array is a view of another array
  private _base?: NDArray;

  constructor(storage: ArrayStorage, base?: NDArray) {
    this._storage = storage;
    this._base = base;
  }

  /**
   * Get internal storage (for ops modules)
   * @internal
   */
  get storage(): ArrayStorage {
    return this._storage;
  }

  /**
   * Create NDArray from storage (for ops modules)
   * @internal
   */
  static _fromStorage(storage: ArrayStorage, base?: NDArray): NDArray {
    return new NDArray(storage, base);
  }

  // NumPy properties
  get shape(): readonly number[] {
    return this._storage.shape;
  }

  get ndim(): number {
    return this._storage.ndim;
  }

  get size(): number {
    return this._storage.size;
  }

  get dtype(): string {
    return this._storage.dtype;
  }

  get data(): TypedArray {
    return this._storage.data;
  }

  get strides(): readonly number[] {
    return this._storage.strides;
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
    return {
      C_CONTIGUOUS: this._storage.isCContiguous,
      F_CONTIGUOUS: this._storage.isFContiguous,
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

    return this._storage.get(...normalizedIndices);
  }

  /**
   * Set a single element in the array
   * @param indices - Array of indices, one per dimension (e.g., [0, 1] for 2D array)
   * @param value - Value to set (will be converted to array's dtype)
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

    this._storage.set(normalizedIndices, convertedValue);
  }

  /**
   * Return a deep copy of the array
   */
  copy(): NDArray {
    return new NDArray(this._storage.copy());
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
      return this.copy();
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

    const storage = ArrayStorage.fromData(newData, shape, dtype);
    return new NDArray(storage);
  }

  // Arithmetic operations
  /**
   * Element-wise addition
   * @param other - Array or scalar to add
   * @returns Result of addition with broadcasting
   */
  add(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.add(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise subtraction
   * @param other - Array or scalar to subtract
   * @returns Result of subtraction with broadcasting
   */
  subtract(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.subtract(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise multiplication
   * @param other - Array or scalar to multiply
   * @returns Result of multiplication with broadcasting
   */
  multiply(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.multiply(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise division
   * @param other - Array or scalar to divide by
   * @returns Result of division with broadcasting
   */
  divide(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.divide(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise modulo operation
   * @param other - Array or scalar divisor
   * @returns Remainder after division
   */
  mod(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.mod(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise floor division
   * @param other - Array or scalar to divide by
   * @returns Floor of the quotient
   */
  floor_divide(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.floorDivide(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Numerical positive (element-wise +x)
   * @returns Copy of the array
   */
  positive(): NDArray {
    const resultStorage = arithmeticOps.positive(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise reciprocal (1/x)
   * @returns New array with reciprocals
   */
  reciprocal(): NDArray {
    const resultStorage = arithmeticOps.reciprocal(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  // Mathematical operations
  /**
   * Square root of each element
   * Promotes integer types to float64
   * @returns New array with square roots
   */
  sqrt(): NDArray {
    const resultStorage = exponentialOps.sqrt(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Raise elements to power
   * @param exponent - Power to raise to (array or scalar)
   * @returns New array with powered values
   */
  power(exponent: NDArray | number): NDArray {
    const exponentStorage = typeof exponent === 'number' ? exponent : exponent._storage;
    const resultStorage = exponentialOps.power(this._storage, exponentStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Absolute value of each element
   * @returns New array with absolute values
   */
  absolute(): NDArray {
    const resultStorage = arithmeticOps.absolute(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Numerical negative (element-wise negation)
   * @returns New array with negated values
   */
  negative(): NDArray {
    const resultStorage = arithmeticOps.negative(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Sign of each element (-1, 0, or 1)
   * @returns New array with signs
   */
  sign(): NDArray {
    const resultStorage = arithmeticOps.sign(this._storage);
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

  /**
   * Element-wise comparison with tolerance
   * Returns True where |a - b| <= (atol + rtol * |b|)
   * @param other - Value or array to compare with
   * @param rtol - Relative tolerance (default: 1e-5)
   * @param atol - Absolute tolerance (default: 1e-8)
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  allclose(other: NDArray | number, rtol: number = 1e-5, atol: number = 1e-8): boolean {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    return comparisonOps.allclose(this._storage, otherStorage, rtol, atol);
  }

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
    const newShape = shape.length === 1 && Array.isArray(shape[0]) ? shape[0] : shape;
    const resultStorage = shapeOps.reshape(this._storage, newShape);
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
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Return a flattened array (view when possible, otherwise copy)
   * @returns 1D array containing all elements
   */
  ravel(): NDArray {
    const resultStorage = shapeOps.ravel(this._storage);
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
    const base = this._base ?? this;
    return NDArray._fromStorage(resultStorage, base);
  }

  // Matrix multiplication
  /**
   * Matrix multiplication
   * @param other - Array to multiply with
   * @returns Result of matrix multiplication
   */
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
   */
  slice(...sliceStrs: string[]): NDArray {
    if (sliceStrs.length === 0) {
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

    // Calculate new shape and strides
    const newShape: number[] = [];
    const newStrides: number[] = [];
    let newOffset = this._storage.offset;

    for (let i = 0; i < sliceSpecs.length; i++) {
      const spec = sliceSpecs[i]!;
      const stride = this._storage.strides[i]!;

      // Update offset based on start position
      newOffset += spec.start * stride;

      if (!spec.isIndex) {
        // Calculate size of this dimension
        // For positive step: (stop - start) / step
        // For negative step: (start - stop) / |step| (since we go from high to low)
        let dimSize: number;
        if (spec.step > 0) {
          dimSize = Math.max(0, Math.ceil((spec.stop - spec.start) / spec.step));
        } else {
          // Negative step: iterate from start down to (but not including) stop
          dimSize = Math.max(0, Math.ceil((spec.start - spec.stop) / Math.abs(spec.step)));
        }
        newShape.push(dimSize);
        newStrides.push(stride * spec.step);
      }
      // If isIndex is true, this dimension is removed (scalar indexing)
    }

    // Create sliced view
    const slicedStorage = ArrayStorage.fromData(
      this._storage.data,
      newShape,
      this._storage.dtype,
      newStrides,
      newOffset
    );

    const base = this._base ?? this;
    return new NDArray(slicedStorage, base);
  }

  // Convenience methods
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
  /**
   * String representation of the array
   * @returns String describing the array shape and dtype
   */
  toString(): string {
    return `NDArray(shape=${JSON.stringify(this.shape)}, dtype=${this.dtype})`;
  }

  /**
   * Convert to nested JavaScript array
   * @returns Nested JavaScript array representation
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  toArray(): any {
    // Handle 0-dimensional arrays (scalars)
    if (this.ndim === 0) {
      return this._storage.iget(0);
    }

    const shape = this.shape;
    const ndim = shape.length;

    // Recursive function to build nested array
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const buildNestedArray = (indices: number[], dim: number): any => {
      if (dim === ndim) {
        return this._storage.get(...indices);
      }

      const arr = [];
      for (let i = 0; i < shape[dim]!; i++) {
        indices[dim] = i;
        arr.push(buildNestedArray(indices, dim + 1));
      }
      return arr;
    };

    return buildNestedArray(new Array(ndim), 0);
  }
}

// Creation functions

/**
 * Create array of zeros
 * @param shape - Shape of the array
 * @param dtype - Data type (default: float64)
 * @returns Array filled with zeros
 */
export function zeros(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArray {
  const storage = ArrayStorage.zeros(shape, dtype);
  return new NDArray(storage);
}

/**
 * Create array of ones
 * @param shape - Shape of the array
 * @param dtype - Data type (default: float64)
 * @returns Array filled with ones
 */
export function ones(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArray {
  const storage = ArrayStorage.ones(shape, dtype);
  return new NDArray(storage);
}

/**
 * Helper to infer shape from nested arrays
 */
function inferShape(data: unknown): number[] {
  const shape: number[] = [];
  let current = data;
  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0];
  }
  return shape;
}

/**
 * Helper to check if data contains BigInt values
 */
function containsBigInt(data: unknown): boolean {
  if (typeof data === 'bigint') return true;
  if (Array.isArray(data)) {
    return data.some((item) => containsBigInt(item));
  }
  return false;
}

/**
 * Helper to flatten nested arrays keeping BigInt values
 */
function flattenKeepBigInt(data: unknown): unknown[] {
  const result: unknown[] = [];
  function flatten(arr: unknown): void {
    if (Array.isArray(arr)) {
      arr.forEach((item) => flatten(item));
    } else {
      result.push(arr);
    }
  }
  flatten(data);
  return result;
}

/**
 * Create array from nested JavaScript arrays
 * @param data - Nested arrays or existing NDArray
 * @param dtype - Data type (optional, will be inferred if not provided)
 * @returns New NDArray
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function array(data: any, dtype?: DType): NDArray {
  // If data is already an NDArray, optionally convert dtype
  if (data instanceof NDArray) {
    if (!dtype || data.dtype === dtype) {
      return data.copy();
    }
    return data.astype(dtype);
  }

  const hasBigInt = containsBigInt(data);

  // Infer shape from nested arrays
  const shape = inferShape(data);
  const size = shape.reduce((a: number, b: number) => a * b, 1);

  // Determine dtype
  let actualDtype = dtype;
  if (!actualDtype) {
    if (hasBigInt) {
      actualDtype = 'int64';
    } else {
      actualDtype = DEFAULT_DTYPE;
    }
  }

  // Get TypedArray constructor
  const Constructor = getTypedArrayConstructor(actualDtype);
  if (!Constructor) {
    throw new Error(`Cannot create array with dtype ${actualDtype}`);
  }

  const typedData = new Constructor(size);
  const flatData = flattenKeepBigInt(data);

  // Fill the typed array
  if (isBigIntDType(actualDtype)) {
    const bigintData = typedData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      const val = flatData[i];
      bigintData[i] = typeof val === 'bigint' ? val : BigInt(Math.round(Number(val)));
    }
  } else if (actualDtype === 'bool') {
    const boolData = typedData as Uint8Array;
    for (let i = 0; i < size; i++) {
      boolData[i] = flatData[i] ? 1 : 0;
    }
  } else {
    const numData = typedData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
    for (let i = 0; i < size; i++) {
      const val = flatData[i];
      numData[i] = typeof val === 'bigint' ? Number(val) : Number(val);
    }
  }

  const storage = ArrayStorage.fromData(typedData, shape, actualDtype);
  return new NDArray(storage);
}

/**
 * Create array with evenly spaced values within a given interval
 * Similar to Python's range() but returns array
 * @param start - Start value (or stop if only one argument)
 * @param stop - Stop value (exclusive)
 * @param step - Step between values (default: 1)
 * @param dtype - Data type (default: float64)
 * @returns Array of evenly spaced values
 */
export function arange(
  start: number,
  stop?: number,
  step: number = 1,
  dtype: DType = DEFAULT_DTYPE
): NDArray {
  let actualStart = start;
  let actualStop = stop;

  if (stop === undefined) {
    actualStart = 0;
    actualStop = start;
  }

  if (actualStop === undefined) {
    throw new Error('stop is required');
  }

  const length = Math.max(0, Math.ceil((actualStop - actualStart) / step));

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create arange array with dtype ${dtype}`);
  }

  const data = new Constructor(length);

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

  const storage = ArrayStorage.fromData(data, [length], dtype);
  return new NDArray(storage);
}

/**
 * Create array with evenly spaced values over a specified interval
 * @param start - Starting value
 * @param stop - Ending value (inclusive)
 * @param num - Number of samples (default: 50)
 * @param dtype - Data type (default: float64)
 * @returns Array of evenly spaced values
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

  const storage = ArrayStorage.fromData(data, [num], dtype);
  return new NDArray(storage);
}

/**
 * Create array with logarithmically spaced values
 * Returns num samples, equally spaced on a log scale from base^start to base^stop
 * @param start - base^start is the starting value
 * @param stop - base^stop is the ending value
 * @param num - Number of samples (default: 50)
 * @param base - Base of the log space (default: 10.0)
 * @param dtype - Data type (default: float64)
 * @returns Array of logarithmically spaced values
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

  const storage = ArrayStorage.fromData(data, [num], dtype);
  return new NDArray(storage);
}

/**
 * Create array with geometrically spaced values
 * Returns num samples, equally spaced on a log scale (geometric progression)
 * @param start - Starting value
 * @param stop - Ending value
 * @param num - Number of samples (default: 50)
 * @param dtype - Data type (default: float64)
 * @returns Array of geometrically spaced values
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

  const storage = ArrayStorage.fromData(data, [num], dtype);
  return new NDArray(storage);
}

/**
 * Create identity matrix
 * @param n - Number of rows
 * @param m - Number of columns (default: n)
 * @param k - Index of diagonal (0 for main diagonal, positive for upper, negative for lower)
 * @param dtype - Data type (default: float64)
 * @returns Identity matrix
 */
export function eye(n: number, m?: number, k: number = 0, dtype: DType = DEFAULT_DTYPE): NDArray {
  const cols = m ?? n;
  const result = zeros([n, cols], dtype);
  const data = result.data;

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
 * Note: TypedArrays are zero-initialized by default in JavaScript
 * @param shape - Shape of the array
 * @param dtype - Data type (default: float64)
 * @returns Uninitialized array
 */
export function empty(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArray {
  return zeros(shape, dtype);
}

/**
 * Create array filled with a constant value
 * @param shape - Shape of the array
 * @param fill_value - Value to fill the array with
 * @param dtype - Data type (optional, inferred from fill_value if not provided)
 * @returns Array filled with the constant value
 */
export function full(
  shape: number[],
  fill_value: number | bigint | boolean,
  dtype?: DType
): NDArray {
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

  if (isBigIntDType(actualDtype)) {
    const bigintValue =
      typeof fill_value === 'bigint' ? fill_value : BigInt(Math.round(Number(fill_value)));
    (data as BigInt64Array | BigUint64Array).fill(bigintValue);
  } else if (actualDtype === 'bool') {
    (data as Uint8Array).fill(fill_value ? 1 : 0);
  } else {
    (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>).fill(Number(fill_value));
  }

  const storage = ArrayStorage.fromData(data, shape, actualDtype);
  return new NDArray(storage);
}

/**
 * Create a square identity matrix
 * @param n - Size of the square matrix
 * @param dtype - Data type (default: float64)
 * @returns n×n identity matrix
 */
export function identity(n: number, dtype: DType = DEFAULT_DTYPE): NDArray {
  return eye(n, n, 0, dtype);
}

/**
 * Convert input to an ndarray
 * @param a - Input data (array-like or NDArray)
 * @param dtype - Data type (optional)
 * @returns NDArray representation of the input
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function asarray(a: NDArray | any, dtype?: DType): NDArray {
  if (a instanceof NDArray) {
    if (!dtype || a.dtype === dtype) {
      return a;
    }
    return a.astype(dtype);
  }
  return array(a, dtype);
}

/**
 * Create a deep copy of an array
 * @param a - Array to copy
 * @returns Deep copy of the array
 */
export function copy(a: NDArray): NDArray {
  return a.copy();
}

/**
 * Create array of zeros with the same shape as another array
 * @param a - Array to match shape from
 * @param dtype - Data type (optional, uses a's dtype if not provided)
 * @returns Array of zeros
 */
export function zeros_like(a: NDArray, dtype?: DType): NDArray {
  return zeros(Array.from(a.shape), dtype ?? (a.dtype as DType));
}

/**
 * Create array of ones with the same shape as another array
 * @param a - Array to match shape from
 * @param dtype - Data type (optional, uses a's dtype if not provided)
 * @returns Array of ones
 */
export function ones_like(a: NDArray, dtype?: DType): NDArray {
  return ones(Array.from(a.shape), dtype ?? (a.dtype as DType));
}

/**
 * Create uninitialized array with the same shape as another array
 * @param a - Array to match shape from
 * @param dtype - Data type (optional, uses a's dtype if not provided)
 * @returns Uninitialized array
 */
export function empty_like(a: NDArray, dtype?: DType): NDArray {
  return empty(Array.from(a.shape), dtype ?? (a.dtype as DType));
}

/**
 * Create array filled with a constant value, same shape as another array
 * @param a - Array to match shape from
 * @param fill_value - Value to fill with
 * @param dtype - Data type (optional, uses a's dtype if not provided)
 * @returns Filled array
 */
export function full_like(
  a: NDArray,
  fill_value: number | bigint | boolean,
  dtype?: DType
): NDArray {
  return full(Array.from(a.shape), fill_value, dtype ?? (a.dtype as DType));
}

// Mathematical functions (standalone)

/**
 * Element-wise square root
 * @param x - Input array
 * @returns Array of square roots
 */
export function sqrt(x: NDArray): NDArray {
  return x.sqrt();
}

/**
 * Element-wise power
 * @param x - Base array
 * @param exponent - Exponent (array or scalar)
 * @returns Array of x raised to exponent
 */
export function power(x: NDArray, exponent: NDArray | number): NDArray {
  return x.power(exponent);
}

/**
 * Element-wise absolute value
 * @param x - Input array
 * @returns Array of absolute values
 */
export function absolute(x: NDArray): NDArray {
  return x.absolute();
}

/**
 * Element-wise negation
 * @param x - Input array
 * @returns Array of negated values
 */
export function negative(x: NDArray): NDArray {
  return x.negative();
}

/**
 * Element-wise sign (-1, 0, or 1)
 * @param x - Input array
 * @returns Array of signs
 */
export function sign(x: NDArray): NDArray {
  return x.sign();
}

/**
 * Element-wise modulo
 * @param x - Dividend array
 * @param divisor - Divisor (array or scalar)
 * @returns Remainder after division
 */
export function mod(x: NDArray, divisor: NDArray | number): NDArray {
  return x.mod(divisor);
}

/**
 * Element-wise floor division
 * @param x - Dividend array
 * @param divisor - Divisor (array or scalar)
 * @returns Floor of the quotient
 */
export function floor_divide(x: NDArray, divisor: NDArray | number): NDArray {
  return x.floor_divide(divisor);
}

/**
 * Element-wise positive (unary +)
 * @param x - Input array
 * @returns Copy of the array
 */
export function positive(x: NDArray): NDArray {
  return x.positive();
}

/**
 * Element-wise reciprocal (1/x)
 * @param x - Input array
 * @returns Array of reciprocals
 */
export function reciprocal(x: NDArray): NDArray {
  return x.reciprocal();
}
