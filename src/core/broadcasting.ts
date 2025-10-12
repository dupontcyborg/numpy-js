/**
 * Broadcasting utilities for NumPy-compatible array operations
 *
 * Wraps @stdlib broadcasting functions with NumPy-compatible API
 */

import broadcastShapes from '@stdlib/ndarray/base/broadcast-shapes';
import broadcastArrays from '@stdlib/ndarray/broadcast-arrays';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type StdlibNDArray = any;

/**
 * Check if two or more shapes are broadcast-compatible
 * and compute the resulting output shape
 *
 * @param shapes - Array of shapes to broadcast
 * @returns The broadcast output shape, or null if incompatible
 *
 * @example
 * ```typescript
 * computeBroadcastShape([[3, 4], [4]]);     // [3, 4]
 * computeBroadcastShape([[3, 4], [3, 1]]);  // [3, 4]
 * computeBroadcastShape([[3, 4], [5]]);     // null (incompatible)
 * ```
 */
export function computeBroadcastShape(shapes: readonly number[][]): number[] | null {
  if (shapes.length === 0) {
    return [];
  }

  if (shapes.length === 1) {
    return Array.from(shapes[0]!);
  }

  // Use @stdlib's broadcastShapes
  // It returns null if shapes are incompatible
  const result = broadcastShapes(shapes.map((s) => Array.from(s)));
  return result;
}

/**
 * Check if two shapes are broadcast-compatible
 *
 * @param shape1 - First shape
 * @param shape2 - Second shape
 * @returns true if shapes can be broadcast together, false otherwise
 *
 * @example
 * ```typescript
 * areBroadcastable([3, 4], [4]);      // true
 * areBroadcastable([3, 4], [3, 1]);   // true
 * areBroadcastable([3, 4], [5]);      // false
 * ```
 */
export function areBroadcastable(shape1: readonly number[], shape2: readonly number[]): boolean {
  return computeBroadcastShape([shape1, shape2]) !== null;
}

/**
 * Broadcast multiple stdlib ndarrays to a common shape
 *
 * Returns read-only views of the input arrays broadcast to the same shape.
 * Views share memory with the original arrays.
 *
 * @param arrays - Stdlib ndarrays to broadcast
 * @returns Array of broadcast stdlib ndarrays
 * @throws Error if arrays have incompatible shapes
 *
 * @example
 * ```typescript
 * const [a, b] = broadcastStdlibArrays([arr1._data, arr2._data]);
 * ```
 */
export function broadcastStdlibArrays(arrays: StdlibNDArray[]): StdlibNDArray[] {
  if (arrays.length === 0) {
    return [];
  }

  if (arrays.length === 1) {
    return arrays;
  }

  // Use @stdlib's broadcastArrays
  // This returns read-only views, which is perfect for efficient broadcasting
  try {
    const result = broadcastArrays(arrays);
    return result;
  } catch (error: unknown) {
    // @stdlib throws for incompatible shapes
    const err = error as Error;
    throw new Error(`operands could not be broadcast together: ${err.message}`);
  }
}

/**
 * Generate a descriptive error message for broadcasting failures
 *
 * @param shapes - The incompatible shapes
 * @param operation - The operation being attempted (e.g., 'add', 'multiply')
 * @returns Error message string
 */
export function broadcastErrorMessage(shapes: readonly number[][], operation?: string): string {
  const opStr = operation ? ` for ${operation}` : '';
  const shapeStrs = shapes.map((s) => `(${s.join(',')})`).join(' ');
  return `operands could not be broadcast together${opStr} with shapes ${shapeStrs}`;
}
