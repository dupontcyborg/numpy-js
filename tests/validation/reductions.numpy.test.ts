/**
 * Python NumPy validation tests for reduction operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array, zeros } from '../../src/core/ndarray';
import { runNumPy, closeEnough, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Reductions', () => {
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

  describe('sum', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = array([1, 2, 3, 4, 5]).sum();
      const pyResult = runNumPy(`
result = np.array([1, 2, 3, 4, 5]).sum()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = array([
        [1, 2, 3],
        [4, 5, 6],
      ]).sum();
      const pyResult = runNumPy(`
result = np.array([[1, 2, 3], [4, 5, 6]]).sum()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });

    it('matches NumPy for empty array', () => {
      const jsResult = zeros([0]).sum();
      const pyResult = runNumPy(`
result = np.zeros(0).sum()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative numbers', () => {
      const jsResult = array([-1, -2, -3, 4, 5]).sum();
      const pyResult = runNumPy(`
result = np.array([-1, -2, -3, 4, 5]).sum()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });
  });

  describe('mean', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = array([1, 2, 3, 4, 5]).mean();
      const pyResult = runNumPy(`
result = np.array([1, 2, 3, 4, 5]).mean()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = array([
        [1, 2, 3],
        [4, 5, 6],
      ]).mean();
      const pyResult = runNumPy(`
result = np.array([[1, 2, 3], [4, 5, 6]]).mean()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });

    it('matches NumPy for floating point precision', () => {
      const jsResult = array([1.5, 2.7, 3.9, 4.1]).mean();
      const pyResult = runNumPy(`
result = np.array([1.5, 2.7, 3.9, 4.1]).mean()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });
  });

  describe('max', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = array([3, 1, 4, 1, 5, 9, 2, 6]).max();
      const pyResult = runNumPy(`
result = np.array([3, 1, 4, 1, 5, 9, 2, 6]).max()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = array([
        [1, 5, 3],
        [8, 2, 7],
      ]).max();
      const pyResult = runNumPy(`
result = np.array([[1, 5, 3], [8, 2, 7]]).max()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative numbers', () => {
      const jsResult = array([-5, -1, -10, -3]).max();
      const pyResult = runNumPy(`
result = np.array([-5, -1, -10, -3]).max()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });

    it('matches NumPy for single element', () => {
      const jsResult = array([42]).max();
      const pyResult = runNumPy(`
result = np.array([42]).max()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });
  });

  describe('min', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = array([3, 1, 4, 1, 5, 9, 2, 6]).min();
      const pyResult = runNumPy(`
result = np.array([3, 1, 4, 1, 5, 9, 2, 6]).min()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = array([
        [5, 3, 8],
        [2, 7, 1],
      ]).min();
      const pyResult = runNumPy(`
result = np.array([[5, 3, 8], [2, 7, 1]]).min()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative numbers', () => {
      const jsResult = array([-5, -1, -10, -3]).min();
      const pyResult = runNumPy(`
result = np.array([-5, -1, -10, -3]).min()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });

    it('matches NumPy for single element', () => {
      const jsResult = array([42]).min();
      const pyResult = runNumPy(`
result = np.array([42]).min()
      `);

      expect(closeEnough(jsResult as number, pyResult.value)).toBe(true);
    });
  });
});
