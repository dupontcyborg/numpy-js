/**
 * Python NumPy Test Oracle
 *
 * Executes Python code with NumPy and returns results for comparison
 *
 * Environment Variables:
 * - NUMPY_PYTHON: Python command to use (default: 'python3')
 *   Examples:
 *     NUMPY_PYTHON='python3' npm test
 *     NUMPY_PYTHON='conda run -n myenv python' npm test
 *     NUMPY_PYTHON='python' npm test
 */

import { execSync, execFileSync } from 'child_process';
import { writeFileSync, unlinkSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

// Get Python command from environment or use default
const PYTHON_CMD = process.env.NUMPY_PYTHON || 'python3';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export interface NumPyResult {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  value: any;
  dtype: string;
  shape: number[];
}

/**
 * Execute Python NumPy code and return the result
 */
export function runNumPy(code: string): NumPyResult {
  // Indent user code properly for try block
  const indentedCode = code
    .trim()
    .split('\n')
    .map((line) => `    ${line}`)
    .join('\n');

  const pythonCode = `import numpy as np
import json
import sys
try:
${indentedCode}
    if isinstance(result, np.ndarray):
        output = {'value': result.tolist(), 'dtype': str(result.dtype), 'shape': list(result.shape)}
    elif isinstance(result, (np.integer, np.floating)):
        output = {'value': float(result), 'dtype': str(type(result).__name__), 'shape': []}
    else:
        output = {'value': result, 'dtype': str(type(result).__name__), 'shape': []}
    print(json.dumps(output))
except Exception as e:
    print(json.dumps({'error': str(e)}), file=sys.stderr)
    sys.exit(1)`;

  // Write to temp file to avoid shell escaping issues
  const tmpFile = join(
    tmpdir(),
    `numpy-test-${Date.now()}-${Math.random().toString(36).slice(2)}.py`
  );

  try {
    writeFileSync(tmpFile, pythonCode, 'utf-8');
    const result = execSync(`${PYTHON_CMD} ${tmpFile}`, {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    const parsed = JSON.parse(result);
    if ('error' in parsed) {
      throw new Error(`NumPy error: ${parsed.error}`);
    }
    return parsed as NumPyResult;
  } catch (error: any) {
    // eslint-disable-line @typescript-eslint/no-explicit-any
    if (error.stderr) {
      const stderrStr = error.stderr.toString();
      try {
        const parsed = JSON.parse(stderrStr);
        if ('error' in parsed) {
          throw new Error(`NumPy error: ${parsed.error}`);
        }
      } catch {
        // Not JSON, throw original error
      }
    }
    throw new Error(`Failed to run Python: ${error.message}`);
  } finally {
    try {
      unlinkSync(tmpFile);
    } catch {
      // Ignore cleanup errors
    }
  }
}

/**
 * Compare two values with tolerance for floating point
 */
export function closeEnough(
  a: number,
  b: number,
  rtol: number = 1e-5,
  atol: number = 1e-8
): boolean {
  return Math.abs(a - b) <= atol + rtol * Math.abs(b);
}

/**
 * Recursively compare arrays/nested arrays with tolerance
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function arraysClose(a: any, b: any, rtol: number = 1e-5, atol: number = 1e-8): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((val, i) => arraysClose(val, b[i], rtol, atol));
  } else if (typeof a === 'number' && typeof b === 'number') {
    return closeEnough(a, b, rtol, atol);
  } else {
    return a === b;
  }
}

/**
 * Check if Python NumPy is available
 */
export function checkNumPyAvailable(): boolean {
  try {
    execSync(`${PYTHON_CMD} -c "import numpy"`, { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

/**
 * Get information about the Python/NumPy environment being used
 */
export function getPythonInfo(): { python: string; numpy: string; command: string } {
  try {
    const info = execSync(
      `${PYTHON_CMD} -c "import sys; import numpy; print(f'{sys.version.split()[0]}|{numpy.__version__}')"`,
      { encoding: 'utf-8' }
    ).trim();
    const [python, numpy] = info.split('|');
    return { python, numpy, command: PYTHON_CMD };
  } catch {
    return { python: 'unknown', numpy: 'unknown', command: PYTHON_CMD };
  }
}
