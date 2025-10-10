/**
 * Python NumPy API Comparison Tests
 *
 * Validates that our TypeScript exports match Python NumPy's API:
 * - Tests each NumPy function individually (will fail for unimplemented)
 * - Tests each ndarray method individually (will fail for unimplemented)
 * - Validates TS exports don't add invalid functions
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../src/index';
import { runNumPy, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

// Get NumPy exports once at module level
let numpyExports: string[] = [];
let ndarrayAttrs: string[] = [];

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

  const pythonInfo = getPythonInfo();
  console.log(`\n🐍 Python ${pythonInfo.python} | NumPy ${pythonInfo.numpy}`);
  console.log(`   Command: ${pythonInfo.command}\n`);

  // Get all public exports from Python NumPy
  const pyExportsResult = runNumPy(`
import numpy
result = sorted([name for name in dir(numpy) if not name.startswith('_')])
  `);
  numpyExports = pyExportsResult.value;

  // Get all public attributes from Python ndarray
  const pyAttrsResult = runNumPy(`
arr = np.zeros([2, 3])
result = sorted([name for name in dir(arr) if not name.startswith('_')])
  `);
  ndarrayAttrs = pyAttrsResult.value;

  console.log(`\n📊 NumPy has ${numpyExports.length} public module exports`);
  console.log(`📊 ndarray has ${ndarrayAttrs.length} public attributes\n`);
});

describe('NumPy API Comparison', () => {
  describe('Module-level exports - test each NumPy function', () => {
    // Priority functions we're tracking
    const priorityFunctions = [
      // Array Creation
      'zeros',
      'ones',
      'array',
      'empty',
      'full',
      'arange',
      'linspace',
      'logspace',
      'eye',
      'identity',
      'zeros_like',
      'ones_like',
      'empty_like',
      'full_like',
      // Arithmetic
      'add',
      'subtract',
      'multiply',
      'divide',
      'power',
      'mod',
      'negative',
      'positive',
      'absolute',
      'abs',
      'sign',
      'sqrt',
      'square',
      'reciprocal',
      // Trigonometric
      'sin',
      'cos',
      'tan',
      'arcsin',
      'arccos',
      'arctan',
      'arctan2',
      'sinh',
      'cosh',
      'tanh',
      'arcsinh',
      'arccosh',
      'arctanh',
      'degrees',
      'radians',
      'hypot',
      // Exponential/Log
      'exp',
      'exp2',
      'expm1',
      'log',
      'log2',
      'log10',
      'log1p',
      // Reductions
      'sum',
      'prod',
      'mean',
      'median',
      'std',
      'var',
      'min',
      'max',
      'argmin',
      'argmax',
      'all',
      'any',
      'ptp',
      'cumsum',
      'cumprod',
      // Linear Algebra
      'dot',
      'matmul',
      'inner',
      'outer',
      'tensordot',
      'trace',
      // Array Manipulation
      'reshape',
      'ravel',
      'flatten',
      'transpose',
      'swapaxes',
      'moveaxis',
      'concatenate',
      'stack',
      'vstack',
      'hstack',
      'split',
      'tile',
      'repeat',
      'expand_dims',
      'squeeze',
      // Comparison
      'equal',
      'not_equal',
      'greater',
      'greater_equal',
      'less',
      'less_equal',
      'allclose',
      'isclose',
      // Logic
      'logical_and',
      'logical_or',
      'logical_not',
      'logical_xor',
      // Sorting
      'sort',
      'argsort',
      'searchsorted',
      // Other
      'where',
      'clip',
      'round',
      'floor',
      'ceil',
      'trunc',
    ];

    // Generate a test for each priority function
    priorityFunctions.forEach((funcName) => {
      it(`numpy.${funcName} should be exported`, () => {
        const tsExports = Object.keys(np);

        // Check if it exists in Python NumPy
        const existsInPython = numpyExports.includes(funcName);
        if (!existsInPython) {
          // Skip if not in Python (maybe wrong name or doesn't exist)
          console.log(`⚠️  ${funcName} not found in Python NumPy`);
          return;
        }

        // Check if we have it in TypeScript
        expect(tsExports).toContain(funcName);
      });
    });
  });

  describe('ndarray attributes - test each Python ndarray attribute', () => {
    // Priority properties and methods
    const priorityAttrs = {
      properties: [
        'shape',
        'ndim',
        'size',
        'dtype',
        'strides',
        'data',
        'T',
        'flat',
        'itemsize',
        'nbytes',
      ],
      methods: [
        'sum',
        'mean',
        'std',
        'var',
        'min',
        'max',
        'argmin',
        'argmax',
        'reshape',
        'transpose',
        'flatten',
        'ravel',
        'squeeze',
        'astype',
        'copy',
        'tolist',
        'fill',
        'clip',
        'round',
        'sort',
        'argsort',
        'all',
        'any',
        'cumsum',
        'cumprod',
        'prod',
        'dot',
        'trace',
      ],
    };

    const allPriorityAttrs = [...priorityAttrs.properties, ...priorityAttrs.methods];

    // Generate a test for each priority attribute
    allPriorityAttrs.forEach((attrName) => {
      it(`ndarray.${attrName} should exist`, () => {
        // Check if it exists in Python
        const existsInPython = ndarrayAttrs.includes(attrName);
        if (!existsInPython) {
          console.log(`⚠️  ${attrName} not found in Python ndarray`);
          return;
        }

        // Check if we have it in TypeScript
        const tsArr = np.zeros([2, 3]);
        const tsAttrs = Object.getOwnPropertyNames(Object.getPrototypeOf(tsArr))
          .concat(Object.keys(tsArr))
          .filter((name, index, self) => self.indexOf(name) === index);

        expect(tsAttrs).toContain(attrName);
      });
    });
  });

  describe('Validate TypeScript exports are valid NumPy functions', () => {
    it('all TS module exports should exist in NumPy (or be allowed custom exports)', () => {
      const tsExports = Object.keys(np).filter((name) => !name.startsWith('_'));

      // Custom exports that are allowed (not in Python NumPy)
      const allowedCustomExports = [
        'NDArray', // Our class name for ndarray
      ];

      const invalidExports: string[] = [];

      for (const tsExport of tsExports) {
        if (allowedCustomExports.includes(tsExport)) {
          continue; // This is an allowed custom export
        }

        if (!numpyExports.includes(tsExport)) {
          invalidExports.push(tsExport);
        }
      }

      if (invalidExports.length > 0) {
        console.log(`\n❌ TS exports not in NumPy: ${invalidExports.join(', ')}\n`);
      }

      expect(invalidExports).toEqual([]);
    });

    it('all TS ndarray attributes should exist in Python ndarray (or be allowed custom)', () => {
      const tsArr = np.zeros([2, 3]);
      const tsAttrs = Object.getOwnPropertyNames(Object.getPrototypeOf(tsArr))
        .concat(Object.keys(tsArr))
        .filter((name) => !name.startsWith('_'))
        .filter((name, index, self) => self.indexOf(name) === index);

      // Custom attributes that are allowed (not in Python ndarray)
      const allowedCustomAttrs = [
        'toArray', // Our custom method to convert to nested JS arrays
        '_data', // Internal stdlib ndarray reference
        'constructor', // JavaScript built-in
        // Note: add, subtract, multiply, divide exist in Python but are ufuncs,
        // not ndarray methods. We implement them as methods for now (will add standalone later)
        'add',
        'subtract',
        'multiply',
        'divide',
        // matmul exists in Python as both function and @ operator, but not as method in old NumPy
        // (it is a method in newer versions though)
        'matmul',
        // toString is JavaScript built-in
        'toString',
      ];

      const invalidAttrs: string[] = [];

      for (const tsAttr of tsAttrs) {
        if (allowedCustomAttrs.includes(tsAttr)) {
          continue; // This is an allowed custom attribute
        }

        if (!ndarrayAttrs.includes(tsAttr)) {
          invalidAttrs.push(tsAttr);
        }
      }

      if (invalidAttrs.length > 0) {
        console.log(`\n❌ TS ndarray attributes not in Python: ${invalidAttrs.join(', ')}\n`);
      }

      expect(invalidAttrs).toEqual([]);
    });
  });

  describe('Summary statistics', () => {
    it('prints implementation progress', () => {
      const tsExports = Object.keys(np).filter((name) => !name.startsWith('_'));
      const tsImplemented = tsExports.filter((name) => numpyExports.includes(name));

      console.log(`\n📈 Implementation Progress:`);
      console.log(
        `   Module exports: ${tsImplemented.length}/${numpyExports.length} (${Math.round((tsImplemented.length / numpyExports.length) * 100)}%)`
      );
      console.log(`   Implemented: ${tsImplemented.join(', ')}\n`);

      const tsArr = np.zeros([2, 3]);
      const tsAttrs = Object.getOwnPropertyNames(Object.getPrototypeOf(tsArr))
        .concat(Object.keys(tsArr))
        .filter((name) => !name.startsWith('_'))
        .filter((name, index, self) => self.indexOf(name) === index);
      const tsAttrsImplemented = tsAttrs.filter((name) => ndarrayAttrs.includes(name));

      console.log(
        `   NDArray attributes: ${tsAttrsImplemented.length}/${ndarrayAttrs.length} (${Math.round((tsAttrsImplemented.length / ndarrayAttrs.length) * 100)}%)`
      );
      console.log(`   Implemented: ${tsAttrsImplemented.join(', ')}\n`);

      expect(true).toBe(true); // Always pass, just for reporting
    });
  });
});
