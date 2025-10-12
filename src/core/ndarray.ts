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
    const outputShape = computeBroadcastShape([Array.from(this.shape), Array.from(other.shape)]);

    if (outputShape === null) {
      throw new Error(
        broadcastErrorMessage([Array.from(this.shape), Array.from(other.shape)], opName)
      );
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

  // Comparison operations
  /**
   * Element-wise greater than comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  greater(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      // Create a uint8 array for boolean result
      const data = new Uint8Array(this.size);
      const thisData = this.data;
      for (let i = 0; i < this.size; i++) {
        data[i] = thisData[i]! > other ? 1 : 0;
      }
      const stdlibArray = stdlib_ndarray.ndarray(
        'uint8',
        data,
        Array.from(this.shape),
        this._computeStrides(this.shape),
        0,
        'row-major'
      );
      return new NDArray(stdlibArray);
    } else {
      // Array comparison with broadcasting
      return this._comparisonOp(other, (a, b) => a > b, 'greater');
    }
  }

  /**
   * Element-wise greater than or equal comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  greater_equal(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      const data = new Uint8Array(this.size);
      const thisData = this.data;
      for (let i = 0; i < this.size; i++) {
        data[i] = thisData[i]! >= other ? 1 : 0;
      }
      const stdlibArray = stdlib_ndarray.ndarray(
        'uint8',
        data,
        Array.from(this.shape),
        this._computeStrides(this.shape),
        0,
        'row-major'
      );
      return new NDArray(stdlibArray);
    } else {
      return this._comparisonOp(other, (a, b) => a >= b, 'greater_equal');
    }
  }

  /**
   * Element-wise less than comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  less(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      const data = new Uint8Array(this.size);
      const thisData = this.data;
      for (let i = 0; i < this.size; i++) {
        data[i] = thisData[i]! < other ? 1 : 0;
      }
      const stdlibArray = stdlib_ndarray.ndarray(
        'uint8',
        data,
        Array.from(this.shape),
        this._computeStrides(this.shape),
        0,
        'row-major'
      );
      return new NDArray(stdlibArray);
    } else {
      return this._comparisonOp(other, (a, b) => a < b, 'less');
    }
  }

  /**
   * Element-wise less than or equal comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  less_equal(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      const data = new Uint8Array(this.size);
      const thisData = this.data;
      for (let i = 0; i < this.size; i++) {
        data[i] = thisData[i]! <= other ? 1 : 0;
      }
      const stdlibArray = stdlib_ndarray.ndarray(
        'uint8',
        data,
        Array.from(this.shape),
        this._computeStrides(this.shape),
        0,
        'row-major'
      );
      return new NDArray(stdlibArray);
    } else {
      return this._comparisonOp(other, (a, b) => a <= b, 'less_equal');
    }
  }

  /**
   * Element-wise equality comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  equal(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      const data = new Uint8Array(this.size);
      const thisData = this.data;
      for (let i = 0; i < this.size; i++) {
        data[i] = thisData[i]! === other ? 1 : 0;
      }
      const stdlibArray = stdlib_ndarray.ndarray(
        'uint8',
        data,
        Array.from(this.shape),
        this._computeStrides(this.shape),
        0,
        'row-major'
      );
      return new NDArray(stdlibArray);
    } else {
      return this._comparisonOp(other, (a, b) => a === b, 'equal');
    }
  }

  /**
   * Element-wise not equal comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  not_equal(other: NDArray | number): NDArray {
    if (typeof other === 'number') {
      const data = new Uint8Array(this.size);
      const thisData = this.data;
      for (let i = 0; i < this.size; i++) {
        data[i] = thisData[i]! !== other ? 1 : 0;
      }
      const stdlibArray = stdlib_ndarray.ndarray(
        'uint8',
        data,
        Array.from(this.shape),
        this._computeStrides(this.shape),
        0,
        'row-major'
      );
      return new NDArray(stdlibArray);
    } else {
      return this._comparisonOp(other, (a, b) => a !== b, 'not_equal');
    }
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
    if (typeof other === 'number') {
      // Scalar comparison
      const data = new Uint8Array(this.size);
      const thisData = this.data;
      for (let i = 0; i < this.size; i++) {
        const a = thisData[i]!;
        const diff = Math.abs(a - other);
        const threshold = atol + rtol * Math.abs(other);
        data[i] = diff <= threshold ? 1 : 0;
      }
      const stdlibArray = stdlib_ndarray.ndarray(
        'uint8',
        data,
        Array.from(this.shape),
        this._computeStrides(this.shape),
        0,
        'row-major'
      );
      return new NDArray(stdlibArray);
    } else {
      // Array comparison with broadcasting
      return this._iscloseOp(other, rtol, atol);
    }
  }

  /**
   * Returns True if all elements are close within tolerance
   * Equivalent to all(isclose(a, b, rtol, atol))
   * @param other - Value or array to compare with
   * @param rtol - Relative tolerance (default: 1e-5)
   * @param atol - Absolute tolerance (default: 1e-8)
   * @returns True if all elements are close
   */
  allclose(other: NDArray | number, rtol: number = 1e-5, atol: number = 1e-8): boolean {
    const iscloseResult = this.isclose(other, rtol, atol);
    const data = iscloseResult.data;

    // Check if all elements are 1 (true)
    for (let i = 0; i < iscloseResult.size; i++) {
      if (data[i] === 0) {
        return false;
      }
    }
    return true;
  }

  /**
   * Helper method for isclose operation with broadcasting
   * @private
   */
  private _iscloseOp(other: NDArray, rtol: number, atol: number): NDArray {
    // Check if shapes are broadcast-compatible
    const outputShape = computeBroadcastShape([Array.from(this.shape), Array.from(other.shape)]);

    if (outputShape === null) {
      throw new Error(
        broadcastErrorMessage([Array.from(this.shape), Array.from(other.shape)], 'isclose')
      );
    }

    // Broadcast both arrays to the output shape
    const [broadcastThis, broadcastOther] = broadcastStdlibArrays([this._data, other._data]);

    // Create result array with uint8 dtype for boolean result
    const size = outputShape.reduce((a, b) => a * b, 1);
    const resultData = new Uint8Array(size);

    // Perform element-wise comparison
    for (let i = 0; i < size; i++) {
      const a = broadcastThis.iget(i);
      const b = broadcastOther.iget(i);
      const diff = Math.abs(a - b);
      const threshold = atol + rtol * Math.abs(b);
      resultData[i] = diff <= threshold ? 1 : 0;
    }

    // Create NDArray with uint8 dtype
    const stdlibArray = stdlib_ndarray.ndarray(
      'uint8',
      resultData,
      outputShape,
      this._computeStrides(outputShape),
      0,
      'row-major'
    );

    return new NDArray(stdlibArray);
  }

  /**
   * Helper method for comparison operations with broadcasting
   * @private
   */
  private _comparisonOp(
    other: NDArray,
    op: (a: number, b: number) => boolean,
    opName: string
  ): NDArray {
    // Check if shapes are broadcast-compatible
    const outputShape = computeBroadcastShape([Array.from(this.shape), Array.from(other.shape)]);

    if (outputShape === null) {
      throw new Error(
        broadcastErrorMessage([Array.from(this.shape), Array.from(other.shape)], opName)
      );
    }

    // Broadcast both arrays to the output shape
    const [broadcastThis, broadcastOther] = broadcastStdlibArrays([this._data, other._data]);

    // Create result array with uint8 dtype for boolean result
    const size = outputShape.reduce((a, b) => a * b, 1);
    const resultData = new Uint8Array(size);

    // Perform element-wise comparison
    for (let i = 0; i < size; i++) {
      const a = broadcastThis.iget(i);
      const b = broadcastOther.iget(i);
      resultData[i] = op(a, b) ? 1 : 0;
    }

    // Create NDArray with uint8 dtype
    const stdlibArray = stdlib_ndarray.ndarray(
      'uint8',
      resultData,
      outputShape,
      this._computeStrides(outputShape),
      0,
      'row-major'
    );

    return new NDArray(stdlibArray);
  }

  // Reductions
  /**
   * Sum array elements over a given axis
   * @param axis - Axis along which to sum. If undefined, sum all elements
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Sum of array elements, or array of sums along axis
   */
  sum(axis?: number, keepdims: boolean = false): NDArray | number {
    if (axis === undefined) {
      // Sum all elements
      const data = this.data;
      let total = 0;
      for (let i = 0; i < this.size; i++) {
        total += data[i]!;
      }
      return total;
    }

    // Validate axis
    if (axis < 0) {
      axis = this.ndim + axis;
    }
    if (axis < 0 || axis >= this.ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${this.ndim}`);
    }

    // Compute output shape
    const outputShape = Array.from(this.shape).filter((_, i) => i !== axis);
    if (outputShape.length === 0) {
      // Result is scalar
      const data = this.data;
      let total = 0;
      for (let i = 0; i < this.size; i++) {
        total += data[i]!;
      }
      return total;
    }

    // Create result array
    const result = zeros(outputShape);
    const resultData = result.data;

    // Perform reduction along axis
    const axisSize = this.shape[axis]!;
    const outerSize = outputShape.reduce((a, b) => a * b, 1);

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let sum = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        // Convert output index to input multi-index
        const inputIndices = this._outerIndexToMultiIndex(outerIdx, axis, axisIdx);
        const linearIdx = this._multiIndexToLinear(inputIndices);
        sum += this.data[linearIdx]!;
      }
      resultData[outerIdx] = sum;
    }

    // Handle keepdims
    if (keepdims) {
      const keepdimsShape = [...this.shape];
      keepdimsShape[axis] = 1;
      // Reshape result to have size-1 dimension at axis position
      const reshapedData = resultData;
      const stdlibArray = stdlib_ndarray.ndarray(
        'float64',
        reshapedData,
        keepdimsShape,
        this._computeStrides(keepdimsShape),
        0,
        'row-major'
      );
      return new NDArray(stdlibArray);
    }

    return result;
  }

  /**
   * Compute the arithmetic mean along the specified axis
   * @param axis - Axis along which to compute mean. If undefined, compute mean of all elements
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Mean of array elements, or array of means along axis
   */
  mean(axis?: number, keepdims: boolean = false): NDArray | number {
    if (axis === undefined) {
      return (this.sum() as number) / this.size;
    }

    // Normalize negative axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = this.ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= this.ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${this.ndim}`);
    }

    const sumResult = this.sum(axis, keepdims);
    if (typeof sumResult === 'number') {
      return sumResult / this.shape[normalizedAxis]!;
    }

    // Divide by the size of the reduced axis
    const divisor = this.shape[normalizedAxis]!;
    const result = zeros(sumResult.shape as number[]);
    const resultData = result.data;
    const sumData = sumResult.data;

    for (let i = 0; i < resultData.length; i++) {
      resultData[i] = sumData[i]! / divisor;
    }

    return result;
  }

  /**
   * Return the maximum along a given axis
   * @param axis - Axis along which to compute maximum. If undefined, compute maximum of all elements
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Maximum of array elements, or array of maximums along axis
   */
  max(axis?: number, keepdims: boolean = false): NDArray | number {
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
    }

    // Validate axis
    if (axis < 0) {
      axis = this.ndim + axis;
    }
    if (axis < 0 || axis >= this.ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${this.ndim}`);
    }

    // Compute output shape
    const outputShape = Array.from(this.shape).filter((_, i) => i !== axis);
    if (outputShape.length === 0) {
      // Result is scalar
      return this.max();
    }

    // Create result array
    const result = zeros(outputShape);
    const resultData = result.data;

    // Perform reduction along axis
    const axisSize = this.shape[axis]!;
    const outerSize = outputShape.reduce((a, b) => a * b, 1);

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let maxVal = -Infinity;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = this._outerIndexToMultiIndex(outerIdx, axis, axisIdx);
        const linearIdx = this._multiIndexToLinear(inputIndices);
        const val = this.data[linearIdx]!;
        if (val > maxVal) {
          maxVal = val;
        }
      }
      resultData[outerIdx] = maxVal;
    }

    // Handle keepdims
    if (keepdims) {
      const keepdimsShape = [...this.shape];
      keepdimsShape[axis] = 1;
      const stdlibArray = stdlib_ndarray.ndarray(
        'float64',
        resultData,
        keepdimsShape,
        this._computeStrides(keepdimsShape),
        0,
        'row-major'
      );
      return new NDArray(stdlibArray);
    }

    return result;
  }

  /**
   * Return the minimum along a given axis
   * @param axis - Axis along which to compute minimum. If undefined, compute minimum of all elements
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Minimum of array elements, or array of minimums along axis
   */
  min(axis?: number, keepdims: boolean = false): NDArray | number {
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
    }

    // Validate axis
    if (axis < 0) {
      axis = this.ndim + axis;
    }
    if (axis < 0 || axis >= this.ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${this.ndim}`);
    }

    // Compute output shape
    const outputShape = Array.from(this.shape).filter((_, i) => i !== axis);
    if (outputShape.length === 0) {
      // Result is scalar
      return this.min();
    }

    // Create result array
    const result = zeros(outputShape);
    const resultData = result.data;

    // Perform reduction along axis
    const axisSize = this.shape[axis]!;
    const outerSize = outputShape.reduce((a, b) => a * b, 1);

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let minVal = Infinity;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = this._outerIndexToMultiIndex(outerIdx, axis, axisIdx);
        const linearIdx = this._multiIndexToLinear(inputIndices);
        const val = this.data[linearIdx]!;
        if (val < minVal) {
          minVal = val;
        }
      }
      resultData[outerIdx] = minVal;
    }

    // Handle keepdims
    if (keepdims) {
      const keepdimsShape = [...this.shape];
      keepdimsShape[axis] = 1;
      const stdlibArray = stdlib_ndarray.ndarray(
        'float64',
        resultData,
        keepdimsShape,
        this._computeStrides(keepdimsShape),
        0,
        'row-major'
      );
      return new NDArray(stdlibArray);
    }

    return result;
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

    // Check if -1 is in the shape (infer dimension)
    const negIndex = newShape.indexOf(-1);
    let finalShape: number[];

    if (negIndex !== -1) {
      // Infer the dimension at negIndex
      const knownSize = newShape.reduce((acc, dim, i) => (i === negIndex ? acc : acc * dim), 1);
      const inferredDim = this.size / knownSize;

      if (!Number.isInteger(inferredDim)) {
        throw new Error(
          `cannot reshape array of size ${this.size} into shape ${JSON.stringify(newShape)}`
        );
      }

      finalShape = newShape.map((dim, i) => (i === negIndex ? inferredDim : dim));
    } else {
      finalShape = newShape;
    }

    // Validate that the new shape has the same total size
    const newSize = finalShape.reduce((a, b) => a * b, 1);
    if (newSize !== this.size) {
      throw new Error(
        `cannot reshape array of size ${this.size} into shape ${JSON.stringify(finalShape)}`
      );
    }

    // Create new array with same data but different shape
    const stdlibArray = stdlib_ndarray.ndarray(
      this.dtype as StdlibNDArray,
      this.data,
      finalShape,
      this._computeStrides(finalShape),
      0,
      'row-major'
    );

    return new NDArray(stdlibArray);
  }

  /**
   * Return a flattened copy of the array
   * @returns 1D array containing all elements
   */
  flatten(): NDArray {
    // Create a 1D copy in row-major (C) order
    const data = new Float64Array(this.size);
    let idx = 0;

    // Helper function to recursively iterate through all indices
    const flattenRecursive = (indices: number[], dim: number) => {
      if (dim === this.ndim) {
        // At leaf, copy the value using stdlib's get method which respects strides
        data[idx++] = this._data.get(...indices);
        return;
      }

      // Iterate through current dimension
      for (let i = 0; i < this.shape[dim]!; i++) {
        indices[dim] = i;
        flattenRecursive(indices, dim + 1);
      }
    };

    flattenRecursive(new Array(this.ndim), 0);

    const stdlibArray = stdlib_ndarray.ndarray('float64', data, [this.size], [1], 0, 'row-major');
    return new NDArray(stdlibArray);
  }

  /**
   * Return a flattened array (view when possible, otherwise copy)
   * Currently always returns a copy like flatten()
   * @returns 1D array containing all elements
   */
  ravel(): NDArray {
    // For now, always return a copy like flatten()
    // In the future, could return a view if the array is C-contiguous
    return this.flatten();
  }

  /**
   * Transpose array (permute dimensions)
   * @param axes - Permutation of axes. If undefined, reverse the dimensions
   * @returns Transposed array
   */
  transpose(axes?: number[]): NDArray {
    let permutation: number[];

    if (axes === undefined) {
      // Default: reverse all dimensions
      permutation = Array.from({ length: this.ndim }, (_, i) => this.ndim - 1 - i);
    } else {
      // Validate axes
      if (axes.length !== this.ndim) {
        throw new Error(`axes must have length ${this.ndim}, got ${axes.length}`);
      }

      // Check that axes is a valid permutation
      const seen = new Set<number>();
      for (const axis of axes) {
        const normalizedAxis = axis < 0 ? this.ndim + axis : axis;
        if (normalizedAxis < 0 || normalizedAxis >= this.ndim) {
          throw new Error(`axis ${axis} is out of bounds for array of dimension ${this.ndim}`);
        }
        if (seen.has(normalizedAxis)) {
          throw new Error(`repeated axis in transpose`);
        }
        seen.add(normalizedAxis);
      }

      permutation = axes.map((ax) => (ax < 0 ? this.ndim + ax : ax));
    }

    // Compute new shape and strides
    const newShape = permutation.map((i) => this.shape[i]!);
    const oldStrides = Array.from(this.strides);
    const newStrides = permutation.map((i) => oldStrides[i]!);

    // Create transposed view
    const stdlibArray = stdlib_ndarray.ndarray(
      this.dtype as StdlibNDArray,
      this.data,
      newShape,
      newStrides,
      stdlib_ndarray.offset(this._data),
      'row-major'
    );

    return new NDArray(stdlibArray);
  }

  /**
   * Remove axes of length 1
   * @param axis - Axis to squeeze. If undefined, squeeze all axes of length 1
   * @returns Array with specified dimensions removed
   */
  squeeze(axis?: number): NDArray {
    if (axis === undefined) {
      // Remove all axes with size 1
      const newShape = Array.from(this.shape).filter((dim) => dim !== 1);

      // If all dimensions were 1, result would be a scalar (0-d array)
      // For now, keep at least one dimension since stdlib may not fully support 0-d arrays
      if (newShape.length === 0) {
        newShape.push(1);
      }

      const newStrides = this._computeStrides(newShape);
      const stdlibArray = stdlib_ndarray.ndarray(
        this.dtype as StdlibNDArray,
        this.data,
        newShape,
        newStrides,
        0,
        'row-major'
      );

      return new NDArray(stdlibArray);
    } else {
      // Normalize axis
      const normalizedAxis = axis < 0 ? this.ndim + axis : axis;

      if (normalizedAxis < 0 || normalizedAxis >= this.ndim) {
        throw new Error(`axis ${axis} is out of bounds for array of dimension ${this.ndim}`);
      }

      // Check that the axis has size 1
      if (this.shape[normalizedAxis] !== 1) {
        throw new Error(
          `cannot select an axis which has size not equal to one (axis ${axis} has size ${this.shape[normalizedAxis]})`
        );
      }

      // Remove the specified axis
      const newShape = Array.from(this.shape).filter((_, i) => i !== normalizedAxis);

      const newStrides = this._computeStrides(newShape);
      const stdlibArray = stdlib_ndarray.ndarray(
        this.dtype as StdlibNDArray,
        this.data,
        newShape,
        newStrides,
        0,
        'row-major'
      );

      return new NDArray(stdlibArray);
    }
  }

  /**
   * Expand the shape by inserting a new axis of length 1
   * @param axis - Position where new axis is placed
   * @returns Array with additional dimension
   */
  expand_dims(axis: number): NDArray {
    // Normalize axis (can be from -ndim-1 to ndim)
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = this.ndim + axis + 1;
    }

    if (normalizedAxis < 0 || normalizedAxis > this.ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${this.ndim + 1}`);
    }

    // Insert 1 at the specified position
    const newShape = [...Array.from(this.shape)];
    newShape.splice(normalizedAxis, 0, 1);

    const newStrides = this._computeStrides(newShape);
    const stdlibArray = stdlib_ndarray.ndarray(
      this.dtype as StdlibNDArray,
      this.data,
      newShape,
      newStrides,
      0,
      'row-major'
    );

    return new NDArray(stdlibArray);
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

        // @ts-expect-error - stdlib Slice type definitions don't match actual behavior
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

  /**
   * Helper: Convert outer index and axis index to full multi-index
   * @private
   */
  private _outerIndexToMultiIndex(outerIdx: number, axis: number, axisIdx: number): number[] {
    const indices = new Array(this.ndim);
    const outputShape = this.shape.filter((_, i) => i !== axis);

    // Convert outerIdx to multi-index in the output shape
    let remaining = outerIdx;
    for (let i = outputShape.length - 1; i >= 0; i--) {
      indices[i >= axis ? i + 1 : i] = remaining % outputShape[i]!;
      remaining = Math.floor(remaining / outputShape[i]!);
    }

    // Insert the axis index
    indices[axis] = axisIdx;
    return indices;
  }

  /**
   * Helper: Convert multi-index to linear index
   * @private
   */
  private _multiIndexToLinear(indices: number[]): number {
    let linearIdx = 0;
    let stride = 1;
    for (let i = this.ndim - 1; i >= 0; i--) {
      linearIdx += indices[i]! * stride;
      stride *= this.shape[i]!;
    }
    return linearIdx;
  }

  /**
   * Helper: Compute strides for a given shape (row-major order)
   * @private
   */
  private _computeStrides(shape: readonly number[]): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i]!;
    }
    return strides;
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
