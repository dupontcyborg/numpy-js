/**
 * Python NumPy validation tests for arithmetic operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array } from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Arithmetic Operations', () => {
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

  describe('add', () => {
    it('matches NumPy for scalar addition', () => {
      const jsResult = array([1, 2, 3, 4, 5]).add(10);
      const pyResult = runNumPy(`
result = np.array([1, 2, 3, 4, 5]) + 10
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for array addition', () => {
      const jsResult = array([1, 2, 3]).add(array([4, 5, 6]));
      const pyResult = runNumPy(`
result = np.array([1, 2, 3]) + np.array([4, 5, 6])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array addition', () => {
      const jsResult = array([
        [1, 2],
        [3, 4],
      ]).add(
        array([
          [5, 6],
          [7, 8],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[1, 2], [3, 4]]) + np.array([[5, 6], [7, 8]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('subtract', () => {
    it('matches NumPy for scalar subtraction', () => {
      const jsResult = array([10, 20, 30]).subtract(5);
      const pyResult = runNumPy(`
result = np.array([10, 20, 30]) - 5
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for array subtraction', () => {
      const jsResult = array([10, 20, 30]).subtract(array([1, 2, 3]));
      const pyResult = runNumPy(`
result = np.array([10, 20, 30]) - np.array([1, 2, 3])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('multiply', () => {
    it('matches NumPy for scalar multiplication', () => {
      const jsResult = array([1, 2, 3]).multiply(5);
      const pyResult = runNumPy(`
result = np.array([1, 2, 3]) * 5
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for element-wise multiplication', () => {
      const jsResult = array([1, 2, 3]).multiply(array([4, 5, 6]));
      const pyResult = runNumPy(`
result = np.array([1, 2, 3]) * np.array([4, 5, 6])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D element-wise multiplication', () => {
      const jsResult = array([
        [1, 2],
        [3, 4],
      ]).multiply(
        array([
          [2, 3],
          [4, 5],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[1, 2], [3, 4]]) * np.array([[2, 3], [4, 5]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('divide', () => {
    it('matches NumPy for scalar division', () => {
      const jsResult = array([10, 20, 30]).divide(10);
      const pyResult = runNumPy(`
result = np.array([10, 20, 30]) / 10
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for element-wise division', () => {
      const jsResult = array([10, 20, 30]).divide(array([2, 4, 5]));
      const pyResult = runNumPy(`
result = np.array([10, 20, 30]) / np.array([2, 4, 5])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('chained operations', () => {
    it('matches NumPy for complex expression', () => {
      const jsResult = array([1, 2, 3]).add(10).multiply(2).subtract(5);
      const pyResult = runNumPy(`
result = (np.array([1, 2, 3]) + 10) * 2 - 5
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('cbrt', () => {
    it('matches NumPy for positive numbers', () => {
      const jsResult = array([1, 8, 27, 64, 125]).cbrt();
      const pyResult = runNumPy(`
result = np.cbrt([1, 8, 27, 64, 125])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative numbers', () => {
      const jsResult = array([-1, -8, -27]).cbrt();
      const pyResult = runNumPy(`
result = np.cbrt([-1, -8, -27])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D arrays', () => {
      const jsResult = array([
        [8, 27],
        [64, 125],
      ]).cbrt();
      const pyResult = runNumPy(`
result = np.cbrt([[8, 27], [64, 125]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for fractional cubes', () => {
      const jsResult = array([0.001, 0.125, 1.728]).cbrt();
      const pyResult = runNumPy(`
result = np.cbrt([0.001, 0.125, 1.728])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fabs', () => {
    it('matches NumPy for positive numbers', () => {
      const jsResult = array([1, 2, 3]).fabs();
      const pyResult = runNumPy(`
result = np.fabs([1, 2, 3])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative numbers', () => {
      const jsResult = array([-1, -2, -3]).fabs();
      const pyResult = runNumPy(`
result = np.fabs([-1, -2, -3])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for mixed numbers', () => {
      const jsResult = array([-1.5, 0, 2.5, -3.5]).fabs();
      const pyResult = runNumPy(`
result = np.fabs([-1.5, 0, 2.5, -3.5])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D arrays', () => {
      const jsResult = array([
        [-1, 2],
        [-3, 4],
      ]).fabs();
      const pyResult = runNumPy(`
result = np.fabs([[-1, 2], [-3, 4]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('divmod', () => {
    it('matches NumPy for positive integers with scalar (quotient)', () => {
      const [jsQuotient] = array([10, 20, 30]).divmod(7);
      const pyResult = runNumPy(`
result = np.floor_divide([10, 20, 30], 7)
      `);

      expect(jsQuotient.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsQuotient.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for positive integers with scalar (remainder)', () => {
      const [, jsRemainder] = array([10, 20, 30]).divmod(7);
      const pyResult = runNumPy(`
result = np.mod([10, 20, 30], 7)
      `);

      expect(jsRemainder.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsRemainder.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for array divisor (quotient)', () => {
      const [jsQuotient] = array([10, 20, 30]).divmod(array([3, 4, 7]));
      const pyResult = runNumPy(`
result = np.floor_divide([10, 20, 30], [3, 4, 7])
      `);

      expect(jsQuotient.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsQuotient.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for array divisor (remainder)', () => {
      const [, jsRemainder] = array([10, 20, 30]).divmod(array([3, 4, 7]));
      const pyResult = runNumPy(`
result = np.mod([10, 20, 30], [3, 4, 7])
      `);

      expect(jsRemainder.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsRemainder.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative dividend (quotient)', () => {
      const [jsQuotient] = array([-10, -20, -30]).divmod(7);
      const pyResult = runNumPy(`
result = np.floor_divide([-10, -20, -30], 7)
      `);

      expect(jsQuotient.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsQuotient.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative dividend (remainder)', () => {
      const [, jsRemainder] = array([-10, -20, -30]).divmod(7);
      const pyResult = runNumPy(`
result = np.mod([-10, -20, -30], 7)
      `);

      expect(jsRemainder.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsRemainder.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D arrays (quotient)', () => {
      const [jsQuotient] = array([
        [10, 20],
        [15, 25],
      ]).divmod(7);
      const pyResult = runNumPy(`
result = np.floor_divide([[10, 20], [15, 25]], 7)
      `);

      expect(jsQuotient.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsQuotient.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D arrays (remainder)', () => {
      const [, jsRemainder] = array([
        [10, 20],
        [15, 25],
      ]).divmod(7);
      const pyResult = runNumPy(`
result = np.mod([[10, 20], [15, 25]], 7)
      `);

      expect(jsRemainder.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsRemainder.toArray(), pyResult.value)).toBe(true);
    });
  });
});
