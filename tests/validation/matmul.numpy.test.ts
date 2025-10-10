/**
 * Python NumPy validation tests for matrix operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array } from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Matrix Operations', () => {
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

  describe('matmul', () => {
    it('matches NumPy for 2x2 @ 2x2', () => {
      const jsResult = array([
        [1, 2],
        [3, 4],
      ]).matmul(
        array([
          [5, 6],
          [7, 8],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2x3 @ 3x2', () => {
      const jsResult = array([
        [1, 2, 3],
        [4, 5, 6],
      ]).matmul(
        array([
          [7, 8],
          [9, 10],
          [11, 12],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[1, 2, 3], [4, 5, 6]]) @ np.array([[7, 8], [9, 10], [11, 12]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 3x3 @ 3x3', () => {
      const jsResult = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]).matmul(
        array([
          [9, 8, 7],
          [6, 5, 4],
          [3, 2, 1],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) @ np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for identity matrix multiplication', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const I = array([
        [1, 0],
        [0, 1],
      ]);
      const jsResult = A.matmul(I);

      const pyResult = runNumPy(`
A = np.array([[1, 2], [3, 4]])
I = np.array([[1, 0], [0, 1]])
result = A @ I
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for matrix with zeros', () => {
      const jsResult = array([
        [1, 0],
        [0, 1],
      ]).matmul(
        array([
          [2, 3],
          [4, 5],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[1, 0], [0, 1]]) @ np.array([[2, 3], [4, 5]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative numbers', () => {
      const jsResult = array([
        [-1, 2],
        [3, -4],
      ]).matmul(
        array([
          [5, -6],
          [-7, 8],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[-1, 2], [3, -4]]) @ np.array([[5, -6], [-7, 8]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for larger matrices (5x5)', () => {
      const A = array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
      ]);
      const B = array([
        [25, 24, 23, 22, 21],
        [20, 19, 18, 17, 16],
        [15, 14, 13, 12, 11],
        [10, 9, 8, 7, 6],
        [5, 4, 3, 2, 1],
      ]);
      const jsResult = A.matmul(B);

      const pyResult = runNumPy(`
A = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]])
B = np.array([[25,24,23,22,21],[20,19,18,17,16],[15,14,13,12,11],[10,9,8,7,6],[5,4,3,2,1]])
result = A @ B
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });
});
