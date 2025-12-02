/**
 * Python NumPy validation tests for linear algebra operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array, dot, trace, transpose, inner, outer, tensordot, diagonal, kron } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Linear Algebra', () => {
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

  describe('dot()', () => {
    it('matches NumPy for 1D · 1D (inner product)', () => {
      const a = array([1, 2, 3, 4, 5]);
      const b = array([6, 7, 8, 9, 10]);
      const jsResult = dot(a, b);

      const pyResult = runNumPy(`
result = np.dot([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for 2D · 2D (matrix multiplication)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = dot(a, b) as any;

      const pyResult = runNumPy(`
result = np.dot([[1, 2], [3, 4]], [[5, 6], [7, 8]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D · 1D (matrix-vector)', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = array([7, 8, 9]);
      const jsResult = dot(a, b) as any;

      const pyResult = runNumPy(`
result = np.dot([[1, 2, 3], [4, 5, 6]], [7, 8, 9])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 1D · 2D (vector-matrix)', () => {
      const a = array([1, 2, 3]);
      const b = array([
        [4, 5],
        [6, 7],
        [8, 9],
      ]);
      const jsResult = dot(a, b) as any;

      const pyResult = runNumPy(`
result = np.dot([1, 2, 3], [[4, 5], [6, 7], [8, 9]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with float arrays', () => {
      const a = array([1.5, 2.5, 3.5]);
      const b = array([4.2, 5.3, 6.4]);
      const jsResult = dot(a, b);

      const pyResult = runNumPy(`
result = np.dot([1.5, 2.5, 3.5], [4.2, 5.3, 6.4])
      `);

      expect(Math.abs((jsResult as number) - pyResult.value)).toBeLessThan(1e-10);
    });

    it('matches NumPy for large vectors', () => {
      const aData = Array.from({ length: 100 }, (_, i) => i);
      const bData = Array.from({ length: 100 }, (_, i) => 99 - i);
      const a = array(aData);
      const b = array(bData);
      const jsResult = dot(a, b);

      const pyResult = runNumPy(`
a = np.arange(100)
b = np.arange(100)[::-1]
result = np.dot(a, b)
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    describe('scalar (0D) cases', () => {
      it('matches NumPy for 0D · 0D (scalar multiplication)', () => {
        const a = array(5);
        const b = array(3);
        const jsResult = dot(a, b);

        const pyResult = runNumPy(`
result = np.dot(np.array(5), np.array(3))
        `);

        expect(jsResult).toBe(pyResult.value);
      });

      it('matches NumPy for 0D · 1D (scalar times vector)', () => {
        const a = array(2);
        const b = array([1, 2, 3, 4]);
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
result = np.dot(np.array(2), np.array([1, 2, 3, 4]))
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 1D · 0D (vector times scalar)', () => {
        const a = array([1, 2, 3, 4]);
        const b = array(2);
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
result = np.dot(np.array([1, 2, 3, 4]), np.array(2))
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 0D · 2D (scalar times matrix)', () => {
        const a = array(3);
        const b = array([
          [1, 2],
          [3, 4],
        ]);
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
result = np.dot(np.array(3), np.array([[1, 2], [3, 4]]))
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 2D · 0D (matrix times scalar)', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array(3);
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
result = np.dot(np.array([[1, 2], [3, 4]]), np.array(3))
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('higher dimensional cases', () => {
      it('matches NumPy for 3D · 1D', () => {
        const a = array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ]); // Shape: (2, 2, 2)
        const b = array([10, 20]); // Shape: (2,)
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = np.array([10, 20])
result = np.dot(a, b)
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 1D · 3D', () => {
        const a = array([1, 2]); // Shape: (2,)
        const b = array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ]); // Shape: (2, 2, 2)
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
a = np.array([1, 2])
b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = np.dot(a, b)
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 3D · 2D (general tensor)', () => {
        const a = array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ]); // Shape: (2, 2, 2)
        const b = array([
          [10, 20, 30],
          [40, 50, 60],
        ]); // Shape: (2, 3)
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = np.array([[10, 20, 30], [40, 50, 60]])
result = np.dot(a, b)
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 2D · 3D (general tensor)', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]); // Shape: (2, 2)
        const b = array([
          [
            [10, 20],
            [30, 40],
          ],
          [
            [50, 60],
            [70, 80],
          ],
        ]); // Shape: (2, 2, 2)
        const jsResult = dot(a, b) as any;

        const pyResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]])
result = np.dot(a, b)
        `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });
  });

  describe('trace()', () => {
    it('matches NumPy for square matrix', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
result = np.trace([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for 2x2 matrix', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
result = np.trace([[1, 2], [3, 4]])
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for non-square matrix (wide)', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
result = np.trace([[1, 2, 3], [4, 5, 6]])
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for non-square matrix (tall)', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
result = np.trace([[1, 2], [3, 4], [5, 6]])
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for identity matrix', () => {
      const a = array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
result = np.trace(np.eye(3))
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy with float values', () => {
      const a = array([
        [1.5, 2.3],
        [3.7, 4.9],
      ]);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
result = np.trace([[1.5, 2.3], [3.7, 4.9]])
      `);

      expect(Math.abs((jsResult as number) - pyResult.value)).toBeLessThan(1e-10);
    });

    it('matches NumPy for large matrix', () => {
      const size = 50;
      const data = Array.from({ length: size }, (_, i) =>
        Array.from({ length: size }, (_, j) => i * size + j)
      );
      const a = array(data);
      const jsResult = trace(a);

      const pyResult = runNumPy(`
a = np.arange(${size * size}).reshape(${size}, ${size})
result = np.trace(a)
      `);

      expect(jsResult).toBe(pyResult.value);
    });
  });

  describe('transpose()', () => {
    it('matches NumPy for 2D array', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const jsResult = transpose(a);

      const pyResult = runNumPy(`
result = np.transpose([[1, 2, 3], [4, 5, 6]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for square matrix', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const jsResult = transpose(a);

      const pyResult = runNumPy(`
result = np.transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 3D array with custom axes', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const jsResult = transpose(a, [1, 0, 2]);

      const pyResult = runNumPy(`
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = np.transpose(a, (1, 0, 2))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 1D array (no-op)', () => {
      const a = array([1, 2, 3, 4, 5]);
      const jsResult = transpose(a);

      const pyResult = runNumPy(`
result = np.transpose([1, 2, 3, 4, 5])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('inner()', () => {
    it('matches NumPy for 1D vectors', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const jsResult = inner(a, b);

      const pyResult = runNumPy(`
result = np.inner([1, 2, 3], [4, 5, 6])
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = inner(a, b) as any;

      const pyResult = runNumPy(`
result = np.inner([[1, 2], [3, 4]], [[5, 6], [7, 8]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D · 1D', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = array([7, 8, 9]);
      const jsResult = inner(a, b) as any;

      const pyResult = runNumPy(`
result = np.inner([[1, 2, 3], [4, 5, 6]], [7, 8, 9])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for different shapes', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]); // (3, 2)
      const b = array([
        [7, 8],
        [9, 10],
      ]); // (2, 2)
      const jsResult = inner(a, b) as any;

      const pyResult = runNumPy(`
result = np.inner([[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('outer()', () => {
    it('matches NumPy for 1D vectors', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5]);
      const jsResult = outer(a, b);

      const pyResult = runNumPy(`
result = np.outer([1, 2, 3], [4, 5])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with 2D inputs (flattens)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([5, 6]);
      const jsResult = outer(a, b);

      const pyResult = runNumPy(`
result = np.outer([[1, 2], [3, 4]], [5, 6])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('tensordot()', () => {
    it('matches NumPy for axes=0 (outer product)', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const jsResult = tensordot(a, b, 0) as any;

      const pyResult = runNumPy(`
result = np.tensordot([1, 2], [3, 4], axes=0)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for axes=1 (dot product)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = tensordot(a, b, 1) as any;

      const pyResult = runNumPy(`
result = np.tensordot([[1, 2], [3, 4]], [[5, 6], [7, 8]], axes=1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for axes=2 (full contraction -> scalar)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = tensordot(a, b, 2);

      const pyResult = runNumPy(`
result = np.tensordot([[1, 2], [3, 4]], [[5, 6], [7, 8]], axes=2)
      `);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for custom axis pairs', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]); // (2, 2, 2)
      const b = array([[[9, 10]], [[11, 12]]]); // (2, 1, 2)
      const jsResult = tensordot(a, b, [[2], [2]]) as any;

      const pyResult = runNumPy(`
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = np.array([[[9, 10]], [[11, 12]]])
result = np.tensordot(a, b, axes=([2], [2]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for multiple axis contraction', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]); // (2, 2, 2)
      const b = array([
        [
          [9, 10],
          [11, 12],
        ],
        [
          [13, 14],
          [15, 16],
        ],
      ]); // (2, 2, 2)
      const jsResult = tensordot(a, b, [
        [1, 2],
        [0, 1],
      ]) as any;

      const pyResult = runNumPy(`
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
result = np.tensordot(a, b, axes=([1, 2], [0, 1]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('diagonal()', () => {
    it('matches NumPy diagonal for 2D array', () => {
      const a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const jsResult = diagonal(a);

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = np.diagonal(a)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy diagonal with positive offset', () => {
      const a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const jsResult = diagonal(a, 1);

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = np.diagonal(a, 1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy diagonal with negative offset', () => {
      const a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const jsResult = diagonal(a, -1);

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = np.diagonal(a, -1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy diagonal for non-square matrix', () => {
      const a = array([[1, 2, 3, 4], [5, 6, 7, 8]]);
      const jsResult = diagonal(a);

      const pyResult = runNumPy(`
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
result = np.diagonal(a)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('kron()', () => {
    it('matches NumPy kron for 1D vectors', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const jsResult = kron(a, b);

      const pyResult = runNumPy(`
a = np.array([1, 2])
b = np.array([3, 4])
result = np.kron(a, b)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy kron for 2D matrices', () => {
      const a = array([[1, 2], [3, 4]]);
      const b = array([[5, 6], [7, 8]]);
      const jsResult = kron(a, b);

      const pyResult = runNumPy(`
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.kron(a, b)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy kron with identity matrix', () => {
      const a = array([[0, 1], [1, 0]]);
      const b = array([[1, 2], [3, 4]]);
      const jsResult = kron(a, b);

      const pyResult = runNumPy(`
a = np.array([[0, 1], [1, 0]])
b = np.array([[1, 2], [3, 4]])
result = np.kron(a, b)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy kron with different dimensions', () => {
      const a = array([[1, 2]]);
      const b = array([3, 4, 5]);
      const jsResult = kron(a, b);

      const pyResult = runNumPy(`
a = np.array([[1, 2]])
b = np.array([3, 4, 5])
result = np.kron(a, b)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });
});
