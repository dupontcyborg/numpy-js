/**
 * Validation module for benchmark correctness
 * Validates that numpy-ts produces the same results as NumPy
 */

import { spawn } from 'child_process';
import { resolve } from 'path';
import * as np from '../../src/index';
import type { BenchmarkCase } from './types';

const FLOAT_TOLERANCE = 1e-10;

/**
 * Compare two arrays or scalars for equality with tolerance
 */
function resultsMatch(numpytsResult: any, numpyResult: any): boolean {
  // Both scalars
  if (typeof numpytsResult === 'number' && typeof numpyResult === 'number') {
    return Math.abs(numpytsResult - numpyResult) < FLOAT_TOLERANCE;
  }

  if (typeof numpytsResult === 'boolean' && typeof numpyResult === 'boolean') {
    return numpytsResult === numpyResult;
  }

  // Both arrays (both should be {shape, data} format at this point)
  if (numpytsResult?.shape && numpyResult?.shape) {
    // Check shapes match
    if (numpytsResult.shape.length !== numpyResult.shape.length) {
      return false;
    }
    if (!numpytsResult.shape.every((dim: number, i: number) => dim === numpyResult.shape[i])) {
      return false;
    }

    // Compare values element by element
    // Both are already in {shape, data} format
    const tsData = numpytsResult.data;
    const npData = numpyResult.data;

    return arraysEqual(tsData, npData);
  }

  return false;
}

/**
 * Recursively compare nested arrays with tolerance
 */
function arraysEqual(a: any, b: any): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((val, i) => arraysEqual(val, b[i]));
  }

  // Compare numbers with tolerance
  if (typeof a === 'number' && typeof b === 'number') {
    if (isNaN(a) && isNaN(b)) return true;
    if (!isFinite(a) && !isFinite(b)) return a === b; // Both inf or -inf
    return Math.abs(a - b) < FLOAT_TOLERANCE;
  }

  // Compare booleans
  if (typeof a === 'boolean' && typeof b === 'boolean') {
    return a === b;
  }

  return a === b;
}

/**
 * Run a single benchmark operation with numpy-ts
 */
function runNumpyTsOperation(spec: BenchmarkCase): any {
  // Setup arrays
  const arrays: Record<string, any> = {};

  for (const [key, config] of Object.entries(spec.setup)) {
    const { shape, dtype = 'float64', fill = 'zeros', value } = config;

    // Handle scalar values
    if (['n', 'axis', 'new_shape', 'shape', 'fill_value', 'target_shape'].includes(key)) {
      arrays[key] = shape[0];
      if (key === 'new_shape' || key === 'shape' || key === 'target_shape') {
        arrays[key] = shape;
      }
      continue;
    }

    // Handle indices array
    if (key === 'indices') {
      arrays[key] = shape;
      continue;
    }

    // Create arrays
    if (value !== undefined) {
      arrays[key] = np.full(shape, value, dtype);
    } else if (fill === 'zeros') {
      arrays[key] = np.zeros(shape, dtype);
    } else if (fill === 'ones') {
      arrays[key] = np.ones(shape, dtype);
    } else if (fill === 'random' || fill === 'arange') {
      const size = shape.reduce((a, b) => a * b, 1);
      const flat = np.arange(0, size, 1, dtype);
      arrays[key] = flat.reshape(...shape);
    }
  }

  // Execute operation
  switch (spec.operation) {
    // Creation
    case 'zeros':
      return np.zeros(arrays.shape);
    case 'ones':
      return np.ones(arrays.shape);
    case 'arange':
      return np.arange(0, arrays.n);
    case 'linspace':
      return np.linspace(0, 100, arrays.n);
    case 'logspace':
      return np.logspace(0, 3, arrays.n);
    case 'geomspace':
      return np.geomspace(1, 1000, arrays.n);
    case 'eye':
      return np.eye(arrays.n);
    case 'identity':
      return np.identity(arrays.n);
    case 'empty':
      return np.empty(arrays.shape);
    case 'full':
      return np.full(arrays.shape, arrays.fill_value);
    case 'copy':
      return arrays.a.copy();
    case 'zeros_like':
      return np.zeros_like(arrays.a);

    // Arithmetic
    case 'add':
      return arrays.b !== undefined ? arrays.a.add(arrays.b) : arrays.a.add(arrays.scalar);
    case 'multiply':
      return arrays.b !== undefined
        ? arrays.a.multiply(arrays.b)
        : arrays.a.multiply(arrays.scalar);
    case 'mod':
      return np.mod(arrays.a, arrays.b || arrays.scalar);
    case 'floor_divide':
      return np.floor_divide(arrays.a, arrays.b || arrays.scalar);
    case 'reciprocal':
      return np.reciprocal(arrays.a);

    // Math
    case 'sqrt':
      return np.sqrt(arrays.a);
    case 'power':
      return np.power(arrays.a, 2);
    case 'absolute':
      return np.absolute(arrays.a);
    case 'negative':
      return np.negative(arrays.a);
    case 'sign':
      return np.sign(arrays.a);

    // Linalg
    case 'dot':
      return arrays.a.dot(arrays.b);
    case 'inner':
      return arrays.a.inner(arrays.b);
    case 'outer':
      return arrays.a.outer(arrays.b);
    case 'tensordot':
      return arrays.a.tensordot(arrays.b, arrays.axes ?? 2);
    case 'matmul':
      return arrays.a.matmul(arrays.b);
    case 'trace':
      return arrays.a.trace();
    case 'transpose':
      return arrays.a.transpose();

    // Reductions
    case 'sum':
      return arrays.axis !== undefined ? arrays.a.sum(arrays.axis) : arrays.a.sum();
    case 'mean':
      return arrays.a.mean();
    case 'max':
      return arrays.a.max();
    case 'min':
      return arrays.a.min();
    case 'prod':
      return arrays.a.prod();
    case 'argmin':
      return arrays.a.argmin();
    case 'argmax':
      return arrays.a.argmax();
    case 'var':
      return arrays.a.var();
    case 'std':
      return arrays.a.std();
    case 'all':
      return arrays.a.all();
    case 'any':
      return arrays.a.any();

    // Reshape
    case 'reshape':
      return arrays.a.reshape(...arrays.new_shape);
    case 'flatten':
      return arrays.a.flatten();
    case 'ravel':
      return arrays.a.ravel();

    // Array manipulation
    case 'swapaxes':
      return np.swapaxes(arrays.a, 0, 1);
    case 'concatenate':
      return np.concatenate([arrays.a, arrays.b], 0);
    case 'stack':
      return np.stack([arrays.a, arrays.b], 0);
    case 'vstack':
      return np.vstack([arrays.a, arrays.b]);
    case 'hstack':
      return np.hstack([arrays.a, arrays.b]);
    case 'tile':
      return np.tile(arrays.a, [2, 2]);
    case 'repeat':
      return arrays.a.repeat(2);

    // Advanced
    case 'broadcast_to':
      return np.broadcast_to(arrays.a, arrays.target_shape);
    case 'take':
      return arrays.a.take(arrays.indices);

    default:
      throw new Error(`Unknown operation: ${spec.operation}`);
  }
}

/**
 * Validate all benchmarks produce correct results
 */
export async function validateBenchmarks(specs: BenchmarkCase[]): Promise<void> {
  const scriptPath = resolve(__dirname, '../scripts/validation.py');

  return new Promise((resolve, reject) => {
    const python = spawn('python3', [scriptPath]);

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    python.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Validation script failed: ${stderr}`));
        return;
      }

      try {
        const numpyResults = JSON.parse(stdout);

        // Run numpy-ts operations and compare
        let passed = 0;
        let failed = 0;

        for (let i = 0; i < specs.length; i++) {
          const spec = specs[i]!;
          const numpyResult = numpyResults[i];

          try {
            const numpytsResult = runNumpyTsOperation(spec);

            // Convert numpy-ts result to comparable format
            let tsValue: any;
            if (typeof numpytsResult === 'number' || typeof numpytsResult === 'boolean') {
              tsValue = numpytsResult;
            } else if (
              numpytsResult &&
              typeof numpytsResult === 'object' &&
              'shape' in numpytsResult
            ) {
              // It's an NDArray - check if it has toArray method
              if (typeof numpytsResult.toArray !== 'function') {
                throw new Error(
                  `NDArray missing toArray method. Type: ${typeof numpytsResult}, keys: ${Object.keys(numpytsResult).join(', ')}`
                );
              }
              tsValue = {
                shape: Array.from(numpytsResult.shape),
                data: numpytsResult.toArray(),
              };
            } else {
              tsValue = numpytsResult;
            }

            if (resultsMatch(tsValue, numpyResult)) {
              passed++;
            } else {
              failed++;
              console.error(`  ❌ ${spec.name}: Results don't match`);
              console.error(`     numpy-ts: ${JSON.stringify(tsValue).substring(0, 100)}`);
              console.error(`     NumPy:    ${JSON.stringify(numpyResult).substring(0, 100)}`);
            }
          } catch (err) {
            failed++;
            console.error(`  ❌ ${spec.name}: Error - ${err}`);
          }
        }

        if (failed > 0) {
          reject(
            new Error(
              `Validation failed: ${failed}/${specs.length} benchmarks produced incorrect results`
            )
          );
        } else {
          console.log(`  ✓ ${passed}/${specs.length} operations validated`);
          resolve();
        }
      } catch (err) {
        reject(new Error(`Failed to parse validation output: ${err}`));
      }
    });

    python.on('error', (err) => {
      reject(new Error(`Failed to spawn Python: ${err.message}`));
    });

    // Send specs to Python
    python.stdin.write(JSON.stringify({ specs }));
    python.stdin.end();
  });
}
