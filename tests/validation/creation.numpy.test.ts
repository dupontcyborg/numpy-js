/**
 * Python NumPy validation tests for array creation functions
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  zeros,
  ones,
  array,
  arange,
  linspace,
  eye,
  empty,
  full,
  identity,
  asarray,
  copy,
  zeros_like,
  ones_like,
  empty_like,
  full_like,
} from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Array Creation', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error(
        '❌ Python NumPy not available!\n\n' +
          '   This test suite requires Python with NumPy installed.\n\n' +
          '   Setup options:\n' +
          '   1. Using system Python: pip install numpy\n' +
          '   2. Using conda: conda install numpy\n' +
          '   3. Set custom Python: NUMPY_PYTHON="conda run -n myenv python" npm test\n\n' +
          '   Current Python command: ' +
          (process.env.NUMPY_PYTHON || 'python3') +
          '\n'
      );
    }
  });

  describe('zeros', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = zeros([5]);
      const pyResult = runNumPy('result = np.zeros(5)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = zeros([3, 4]);
      const pyResult = runNumPy('result = np.zeros((3, 4))');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for 3D array', () => {
      const jsResult = zeros([2, 3, 4]);
      const pyResult = runNumPy('result = np.zeros((2, 3, 4))');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.size).toBe(2 * 3 * 4);
    });
  });

  describe('ones', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = ones([5]);
      const pyResult = runNumPy('result = np.ones(5)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = ones([3, 4]);
      const pyResult = runNumPy('result = np.ones((3, 4))');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });
  });

  describe('array', () => {
    it('matches NumPy for 1D array from list', () => {
      const jsResult = array([1, 2, 3, 4, 5]);
      const pyResult = runNumPy('result = np.array([1, 2, 3, 4, 5])');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for 2D array from nested lists', () => {
      const jsResult = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const pyResult = runNumPy('result = np.array([[1, 2, 3], [4, 5, 6]])');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for 3D array', () => {
      const jsResult = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const pyResult = runNumPy('result = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.ndim).toBe(pyResult.shape.length);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });
  });

  describe('arange', () => {
    it('matches NumPy for arange(n)', () => {
      const jsResult = arange(10);
      const pyResult = runNumPy('result = np.arange(10)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for arange(start, stop)', () => {
      const jsResult = arange(5, 15);
      const pyResult = runNumPy('result = np.arange(5, 15)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for arange(start, stop, step)', () => {
      const jsResult = arange(0, 20, 3);
      const pyResult = runNumPy('result = np.arange(0, 20, 3)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative range', () => {
      const jsResult = arange(10, 0, -2);
      const pyResult = runNumPy('result = np.arange(10, 0, -2)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for float step', () => {
      const jsResult = arange(0, 1, 0.1);
      const pyResult = runNumPy('result = np.arange(0, 1, 0.1)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('linspace', () => {
    it('matches NumPy for linspace(start, stop, num)', () => {
      const jsResult = linspace(0, 10, 11);
      const pyResult = runNumPy('result = np.linspace(0, 10, 11)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for default 50 points', () => {
      const jsResult = linspace(0, 1);
      const pyResult = runNumPy('result = np.linspace(0, 1)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative range', () => {
      const jsResult = linspace(-5, 5, 11);
      const pyResult = runNumPy('result = np.linspace(-5, 5, 11)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for single point', () => {
      const jsResult = linspace(5, 10, 1);
      const pyResult = runNumPy('result = np.linspace(5, 10, 1)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('eye', () => {
    it('matches NumPy for square identity', () => {
      const jsResult = eye(3);
      const pyResult = runNumPy('result = np.eye(3)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for rectangular identity', () => {
      const jsResult = eye(3, 5);
      const pyResult = runNumPy('result = np.eye(3, 5)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for diagonal offset k=1', () => {
      const jsResult = eye(4, 4, 1);
      const pyResult = runNumPy('result = np.eye(4, 4, k=1)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for diagonal offset k=-1', () => {
      const jsResult = eye(4, 4, -1);
      const pyResult = runNumPy('result = np.eye(4, 4, k=-1)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });
  });

  describe('empty', () => {
    it('matches NumPy for 1D array shape', () => {
      const jsResult = empty([5]);
      const pyResult = runNumPy('result = np.empty(5)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });

    it('matches NumPy for 2D array shape', () => {
      const jsResult = empty([3, 4]);
      const pyResult = runNumPy('result = np.empty((3, 4))');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.size).toBe(3 * 4);
    });

    it('matches NumPy for 3D array shape', () => {
      const jsResult = empty([2, 3, 4]);
      const pyResult = runNumPy('result = np.empty((2, 3, 4))');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.size).toBe(2 * 3 * 4);
    });
  });

  describe('full', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = full([5], 7);
      const pyResult = runNumPy('result = np.full(5, 7)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = full([3, 4], 2.5);
      const pyResult = runNumPy('result = np.full((3, 4), 2.5)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative fill value', () => {
      const jsResult = full([2, 3], -9);
      const pyResult = runNumPy('result = np.full((2, 3), -9)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for float fill value', () => {
      const jsResult = full([2, 2], 3.14);
      const pyResult = runNumPy('result = np.full((2, 2), 3.14)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for int32 dtype', () => {
      const jsResult = full([2, 2], 42, 'int32');
      const pyResult = runNumPy('result = np.full((2, 2), 42, dtype=np.int32)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });
  });

  describe('identity', () => {
    it('matches NumPy for square identity', () => {
      const jsResult = identity(3);
      const pyResult = runNumPy('result = np.identity(3)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for large identity', () => {
      const jsResult = identity(5);
      const pyResult = runNumPy('result = np.identity(5)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for 1x1 identity', () => {
      const jsResult = identity(1);
      const pyResult = runNumPy('result = np.identity(1)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });
  });

  describe('asarray', () => {
    it('matches NumPy for nested array conversion', () => {
      const jsResult = asarray([
        [1, 2],
        [3, 4],
      ]);
      const pyResult = runNumPy('result = np.asarray([[1, 2], [3, 4]])');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for 1D array conversion', () => {
      const jsResult = asarray([1, 2, 3, 4, 5]);
      const pyResult = runNumPy('result = np.asarray([1, 2, 3, 4, 5])');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('preserves existing array like NumPy', () => {
      const original = array([
        [1, 2],
        [3, 4],
      ]);
      const jsResult = asarray(original);

      // NumPy asarray also returns the same object if no conversion needed
      expect(jsResult).toBe(original);
    });
  });

  describe('copy', () => {
    it('matches NumPy for 2D array copy', () => {
      const original = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const jsResult = copy(original);
      const pyResult = runNumPy('arr = np.array([[1, 2, 3], [4, 5, 6]]); result = np.copy(arr)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
      expect(jsResult).not.toBe(original);
    });

    it('matches NumPy for 1D array copy', () => {
      const original = array([1, 2, 3, 4, 5]);
      const jsResult = copy(original);
      const pyResult = runNumPy('arr = np.array([1, 2, 3, 4, 5]); result = np.copy(arr)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('copy is independent like NumPy', () => {
      const original = zeros([2, 2]);
      const copied = copy(original);

      original.set([0, 0], 99);
      expect(original.get([0, 0])).toBe(99);
      expect(copied.get([0, 0])).toBe(0);
    });
  });

  describe('zeros_like', () => {
    it('matches NumPy for 2D array', () => {
      const template = array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        'float64'
      );
      const jsResult = zeros_like(template);
      const pyResult = runNumPy(
        'template = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64); result = np.zeros_like(template)'
      );

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });

    it('matches NumPy for 1D array', () => {
      const template = ones([5]);
      const jsResult = zeros_like(template);
      const pyResult = runNumPy('template = np.ones(5); result = np.zeros_like(template)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy with dtype conversion', () => {
      const template = array([[1, 2]], 'int32');
      const jsResult = zeros_like(template, 'float64');
      const pyResult = runNumPy(
        'template = np.array([[1, 2]], dtype=np.int32); result = np.zeros_like(template, dtype=np.float64)'
      );

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });
  });

  describe('ones_like', () => {
    it('matches NumPy for 2D array', () => {
      const template = zeros([3, 4]);
      const jsResult = ones_like(template);
      const pyResult = runNumPy('template = np.zeros((3, 4)); result = np.ones_like(template)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for 1D array', () => {
      const template = zeros([7]);
      const jsResult = ones_like(template);
      const pyResult = runNumPy('template = np.zeros(7); result = np.ones_like(template)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });
  });

  describe('empty_like', () => {
    it('matches NumPy for shape and dtype', () => {
      const template = array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        'float64'
      );
      const jsResult = empty_like(template);
      const pyResult = runNumPy(
        'template = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64); result = np.empty_like(template)'
      );

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });

    it('matches NumPy with dtype conversion', () => {
      const template = zeros([2, 2], 'float64');
      const jsResult = empty_like(template, 'int32');
      const pyResult = runNumPy(
        'template = np.zeros((2, 2), dtype=np.float64); result = np.empty_like(template, dtype=np.int32)'
      );

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });
  });

  describe('full_like', () => {
    it('matches NumPy for 2D array', () => {
      const template = zeros([2, 3]);
      const jsResult = full_like(template, 42);
      const pyResult = runNumPy('template = np.zeros((2, 3)); result = np.full_like(template, 42)');

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });

    it('matches NumPy for negative fill value', () => {
      const template = ones([3, 2]);
      const jsResult = full_like(template, -7.5);
      const pyResult = runNumPy(
        'template = np.ones((3, 2)); result = np.full_like(template, -7.5)'
      );

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with dtype conversion', () => {
      const template = zeros([2, 2], 'float64');
      const jsResult = full_like(template, 99, 'int32');
      const pyResult = runNumPy(
        'template = np.zeros((2, 2), dtype=np.float64); result = np.full_like(template, 99, dtype=np.int32)'
      );

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(jsResult.toArray()).toEqual(pyResult.value);
    });
  });
});
