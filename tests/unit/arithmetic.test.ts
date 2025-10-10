import { describe, it, expect } from 'vitest';
import { array, zeros } from '../../src/core/ndarray';

describe('Arithmetic Operations', () => {
  describe('add', () => {
    it('adds scalar to array', () => {
      const arr = array([1, 2, 3]);
      const result = arr.add(10);
      expect(result.toArray()).toEqual([11, 12, 13]);
    });

    it('adds two arrays element-wise', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = a.add(b);
      expect(result.toArray()).toEqual([5, 7, 9]);
    });

    it('adds 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = a.add(b);
      expect(result.toArray()).toEqual([
        [6, 8],
        [10, 12],
      ]);
    });
  });

  describe('subtract', () => {
    it('subtracts scalar from array', () => {
      const arr = array([10, 20, 30]);
      const result = arr.subtract(5);
      expect(result.toArray()).toEqual([5, 15, 25]);
    });

    it('subtracts two arrays element-wise', () => {
      const a = array([10, 20, 30]);
      const b = array([1, 2, 3]);
      const result = a.subtract(b);
      expect(result.toArray()).toEqual([9, 18, 27]);
    });
  });

  describe('multiply', () => {
    it('multiplies array by scalar', () => {
      const arr = array([1, 2, 3]);
      const result = arr.multiply(3);
      expect(result.toArray()).toEqual([3, 6, 9]);
    });

    it('multiplies two arrays element-wise', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = a.multiply(b);
      expect(result.toArray()).toEqual([4, 10, 18]);
    });
  });

  describe('divide', () => {
    it('divides array by scalar', () => {
      const arr = array([10, 20, 30]);
      const result = arr.divide(10);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('divides two arrays element-wise', () => {
      const a = array([10, 20, 30]);
      const b = array([2, 4, 5]);
      const result = a.divide(b);
      expect(result.toArray()).toEqual([5, 5, 6]);
    });
  });

  describe('chained operations', () => {
    it('chains multiple operations', () => {
      const arr = array([1, 2, 3]);
      const result = arr.add(10).multiply(2).subtract(5);
      expect(result.toArray()).toEqual([17, 19, 21]);
    });
  });
});

describe('Reduction Operations', () => {
  describe('sum', () => {
    it('sums all elements in 1D array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.sum()).toBe(15);
    });

    it('sums all elements in 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(arr.sum()).toBe(21);
    });

    it('sums empty array', () => {
      const arr = zeros([0]);
      expect(arr.sum()).toBe(0);
    });
  });

  describe('mean', () => {
    it('computes mean of 1D array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.mean()).toBe(3);
    });

    it('computes mean of 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(arr.mean()).toBe(3.5);
    });
  });

  describe('max', () => {
    it('finds maximum in 1D array', () => {
      const arr = array([3, 1, 4, 1, 5, 9, 2, 6]);
      expect(arr.max()).toBe(9);
    });

    it('finds maximum in 2D array', () => {
      const arr = array([
        [1, 5, 3],
        [8, 2, 7],
      ]);
      expect(arr.max()).toBe(8);
    });

    it('handles negative numbers', () => {
      const arr = array([-5, -1, -10, -3]);
      expect(arr.max()).toBe(-1);
    });
  });

  describe('min', () => {
    it('finds minimum in 1D array', () => {
      const arr = array([3, 1, 4, 1, 5, 9, 2, 6]);
      expect(arr.min()).toBe(1);
    });

    it('finds minimum in 2D array', () => {
      const arr = array([
        [5, 3, 8],
        [2, 7, 1],
      ]);
      expect(arr.min()).toBe(1);
    });

    it('handles negative numbers', () => {
      const arr = array([-5, -1, -10, -3]);
      expect(arr.min()).toBe(-10);
    });
  });
});
