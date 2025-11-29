/**
 * Python NumPy validation tests for array manipulation operations
 *
 * Tests: swapaxes, moveaxis, concatenate, stack, vstack, hstack, dstack,
 *        split, array_split, vsplit, hsplit, tile, repeat
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  swapaxes,
  moveaxis,
  concatenate,
  stack,
  vstack,
  hstack,
  dstack,
  split,
  array_split,
  vsplit,
  hsplit,
  tile,
  repeat,
} from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Array Manipulation', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  // ========================================
  // swapaxes
  // ========================================
  describe('swapaxes()', () => {
    it('validates swapaxes 2D', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = swapaxes(arr, 0, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.swapaxes(arr, 0, 1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates swapaxes 3D', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = swapaxes(arr, 0, 2);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = np.swapaxes(arr, 0, 2)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates swapaxes with negative axes', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = swapaxes(arr, -2, -1);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.swapaxes(arr, -2, -1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // moveaxis
  // ========================================
  describe('moveaxis()', () => {
    it('validates moveaxis single axis', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = moveaxis(arr, 0, -1);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = np.moveaxis(arr, 0, -1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates moveaxis multiple axes', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = moveaxis(arr, [0, 1], [-1, -2]);

      const npResult = runNumPy(`
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = np.moveaxis(arr, [0, 1], [-1, -2])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // concatenate
  // ========================================
  describe('concatenate()', () => {
    it('validates concatenate 1D', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = concatenate([a, b]);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.concatenate([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates concatenate 2D axis=0', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concatenate([a, b], 0);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.concatenate([a, b], axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates concatenate 2D axis=1', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = concatenate([a, b], 1);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.concatenate([a, b], axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates concatenate multiple arrays', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const c = array([5, 6]);
      const result = concatenate([a, b, c]);

      const npResult = runNumPy(`
a = np.array([1, 2])
b = np.array([3, 4])
c = np.array([5, 6])
result = np.concatenate([a, b, c])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // stack
  // ========================================
  describe('stack()', () => {
    it('validates stack 1D arrays axis=0', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = stack([a, b], 0);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.stack([a, b], axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates stack 1D arrays axis=1', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = stack([a, b], 1);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.stack([a, b], axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates stack 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = stack([a, b], 0);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.stack([a, b], axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // vstack
  // ========================================
  describe('vstack()', () => {
    it('validates vstack 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = vstack([a, b]);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.vstack([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates vstack 2D arrays', () => {
      const a = array([[1, 2, 3]]);
      const b = array([[4, 5, 6]]);
      const result = vstack([a, b]);

      const npResult = runNumPy(`
a = np.array([[1, 2, 3]])
b = np.array([[4, 5, 6]])
result = np.vstack([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // hstack
  // ========================================
  describe('hstack()', () => {
    it('validates hstack 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = hstack([a, b]);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.hstack([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates hstack 2D arrays', () => {
      const a = array([[1], [2]]);
      const b = array([[3], [4]]);
      const result = hstack([a, b]);

      const npResult = runNumPy(`
a = np.array([[1], [2]])
b = np.array([[3], [4]])
result = np.hstack([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // dstack
  // ========================================
  describe('dstack()', () => {
    it('validates dstack 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = dstack([a, b]);

      const npResult = runNumPy(`
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.dstack([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates dstack 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = dstack([a, b]);

      const npResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.dstack([a, b])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // split
  // ========================================
  describe('split()', () => {
    it('validates split into equal parts', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = split(arr, 3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5, 6])
result = np.split(arr, 3)
result = np.array([r.tolist() for r in result])
`);

      expect(result.length).toBe(3);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
      expect(arraysClose(result[2]!.toArray(), npResult.value[2])).toBe(true);
    });

    it('validates split at indices', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = split(arr, [2, 4]);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5, 6])
result = np.split(arr, [2, 4])
result = np.array([r.tolist() for r in result])
`);

      expect(result.length).toBe(3);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
      expect(arraysClose(result[2]!.toArray(), npResult.value[2])).toBe(true);
    });

    it('validates split 2D along axis', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const result = split(arr, 2, 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
result = np.split(arr, 2, axis=0)
result = np.array([r.tolist() for r in result])
`);

      expect(result.length).toBe(2);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
    });
  });

  // ========================================
  // array_split
  // ========================================
  describe('array_split()', () => {
    it('validates array_split unequal parts', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = array_split(arr, 3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
result = np.array_split(arr, 3)
result = [r.tolist() for r in result]
`);

      expect(result.length).toBe(3);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
      expect(arraysClose(result[2]!.toArray(), npResult.value[2])).toBe(true);
    });
  });

  // ========================================
  // vsplit
  // ========================================
  describe('vsplit()', () => {
    it('validates vsplit 2D array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ]);
      const result = vsplit(arr, 2);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
result = np.vsplit(arr, 2)
result = np.array([r.tolist() for r in result])
`);

      expect(result.length).toBe(2);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
    });
  });

  // ========================================
  // hsplit
  // ========================================
  describe('hsplit()', () => {
    it('validates hsplit 1D array', () => {
      const arr = array([1, 2, 3, 4, 5, 6]);
      const result = hsplit(arr, 3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5, 6])
result = np.hsplit(arr, 3)
result = np.array([r.tolist() for r in result])
`);

      expect(result.length).toBe(3);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
      expect(arraysClose(result[2]!.toArray(), npResult.value[2])).toBe(true);
    });

    it('validates hsplit 2D array', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const result = hsplit(arr, 2);

      const npResult = runNumPy(`
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
result = np.hsplit(arr, 2)
result = np.array([r.tolist() for r in result])
`);

      expect(result.length).toBe(2);
      expect(arraysClose(result[0]!.toArray(), npResult.value[0])).toBe(true);
      expect(arraysClose(result[1]!.toArray(), npResult.value[1])).toBe(true);
    });
  });

  // ========================================
  // tile
  // ========================================
  describe('tile()', () => {
    it('validates tile 1D array', () => {
      const arr = array([1, 2, 3]);
      const result = tile(arr, 3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.tile(arr, 3)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates tile 1D with 2D reps', () => {
      const arr = array([1, 2]);
      const result = tile(arr, [2, 3]);

      const npResult = runNumPy(`
arr = np.array([1, 2])
result = np.tile(arr, (2, 3))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates tile 2D array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = tile(arr, [2, 2]);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.tile(arr, (2, 2))
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  // ========================================
  // repeat
  // ========================================
  describe('repeat()', () => {
    it('validates repeat 1D array', () => {
      const arr = array([1, 2, 3]);
      const result = repeat(arr, 2);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.repeat(arr, 2)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates repeat 2D array (flattened)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(arr, 2);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.repeat(arr, 2)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates repeat along axis 0', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(arr, 2, 0);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.repeat(arr, 2, axis=0)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates repeat along axis 1', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = repeat(arr, 2, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = np.repeat(arr, 2, axis=1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates repeat with different counts', () => {
      const arr = array([1, 2, 3]);
      const result = repeat(arr, [1, 2, 3]);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.repeat(arr, [1, 2, 3])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });
});
