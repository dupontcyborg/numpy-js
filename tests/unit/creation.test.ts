import { describe, it, expect } from 'vitest';
import { arange, linspace, eye } from '../../src/core/ndarray';

describe('Array Creation Functions', () => {
  describe('arange', () => {
    it('creates array from 0 to n', () => {
      const arr = arange(5);
      expect(arr.shape).toEqual([5]);
      expect(arr.toArray()).toEqual([0, 1, 2, 3, 4]);
    });

    it('creates array from start to stop', () => {
      const arr = arange(2, 7);
      expect(arr.shape).toEqual([5]);
      expect(arr.toArray()).toEqual([2, 3, 4, 5, 6]);
    });

    it('creates array with custom step', () => {
      const arr = arange(0, 10, 2);
      expect(arr.shape).toEqual([5]);
      expect(arr.toArray()).toEqual([0, 2, 4, 6, 8]);
    });

    it('creates array with negative step', () => {
      const arr = arange(10, 0, -2);
      expect(arr.shape).toEqual([5]);
      expect(arr.toArray()).toEqual([10, 8, 6, 4, 2]);
    });

    it('creates array with float step', () => {
      const arr = arange(0, 1, 0.25);
      expect(arr.shape).toEqual([4]);
      expect(arr.toArray()).toEqual([0, 0.25, 0.5, 0.75]);
    });

    it('creates empty array for invalid range', () => {
      const arr = arange(5, 2);
      expect(arr.shape).toEqual([0]);
      expect(arr.toArray()).toEqual([]);
    });
  });

  describe('linspace', () => {
    it('creates 50 points by default', () => {
      const arr = linspace(0, 1);
      expect(arr.shape).toEqual([50]);
      expect(arr.toArray()[0]).toBe(0);
      expect(arr.toArray()[49]).toBeCloseTo(1);
    });

    it('creates specified number of points', () => {
      const arr = linspace(0, 10, 5);
      expect(arr.shape).toEqual([5]);
      expect(arr.toArray()).toEqual([0, 2.5, 5, 7.5, 10]);
    });

    it('creates single point', () => {
      const arr = linspace(5, 10, 1);
      expect(arr.shape).toEqual([1]);
      expect(arr.toArray()).toEqual([5]);
    });

    it('creates empty array for num=0', () => {
      const arr = linspace(0, 1, 0);
      expect(arr.shape).toEqual([0]);
    });

    it('works with negative range', () => {
      const arr = linspace(-5, 5, 11);
      expect(arr.shape).toEqual([11]);
      expect(arr.toArray()[0]).toBe(-5);
      expect(arr.toArray()[5]).toBe(0);
      expect(arr.toArray()[10]).toBe(5);
    });
  });

  describe('eye', () => {
    it('creates square identity matrix', () => {
      const arr = eye(3);
      expect(arr.shape).toEqual([3, 3]);
      expect(arr.toArray()).toEqual([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
    });

    it('creates rectangular identity matrix', () => {
      const arr = eye(2, 3);
      expect(arr.shape).toEqual([2, 3]);
      expect(arr.toArray()).toEqual([
        [1, 0, 0],
        [0, 1, 0],
      ]);
    });

    it('creates identity matrix with offset diagonal (k=1)', () => {
      const arr = eye(3, 3, 1);
      expect(arr.shape).toEqual([3, 3]);
      expect(arr.toArray()).toEqual([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
      ]);
    });

    it('creates identity matrix with offset diagonal (k=-1)', () => {
      const arr = eye(3, 3, -1);
      expect(arr.shape).toEqual([3, 3]);
      expect(arr.toArray()).toEqual([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
      ]);
    });
  });
});
