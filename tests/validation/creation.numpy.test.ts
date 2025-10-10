/**
 * Python NumPy validation tests for array creation functions
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { zeros, ones, array, arange, linspace, eye } from '../../src/core/ndarray';
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
});
