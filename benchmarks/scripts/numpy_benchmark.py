#!/usr/bin/env python3
"""
NumPy benchmark script
Receives benchmark specifications via stdin as JSON
Returns timing results as JSON
"""

import numpy as np
import json
import sys
import time
from typing import Dict, Any, List


def setup_arrays(setup: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Create arrays based on setup specification"""
    arrays = {}

    for key, spec in setup.items():
        shape = spec['shape']
        dtype = spec.get('dtype', 'float64')
        fill_type = spec.get('fill', 'zeros')

        # Handle scalar values (n, axis, new_shape, shape, fill_value)
        if key in ['n', 'axis', 'new_shape', 'shape', 'fill_value']:
            if len(shape) == 1:
                arrays[key] = shape[0]
            else:
                arrays[key] = tuple(shape)
            continue

        if fill_type == 'zeros':
            arrays[key] = np.zeros(shape, dtype=dtype)
        elif fill_type == 'ones':
            arrays[key] = np.ones(shape, dtype=dtype)
        elif fill_type == 'random':
            arrays[key] = np.random.randn(*shape).astype(dtype)
        elif fill_type == 'arange':
            arrays[key] = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        elif 'value' in spec:
            arrays[key] = np.full(shape, spec['value'], dtype=dtype)

    return arrays


def execute_operation(operation: str, arrays: Dict[str, np.ndarray]) -> Any:
    """Execute the benchmark operation"""
    # Array creation
    if operation == 'zeros':
        return np.zeros(arrays['shape'])
    elif operation == 'ones':
        return np.ones(arrays['shape'])
    elif operation == 'empty':
        return np.empty(arrays['shape'])
    elif operation == 'full':
        return np.full(arrays['shape'], arrays['fill_value'])
    elif operation == 'arange':
        return np.arange(arrays['n'])
    elif operation == 'linspace':
        return np.linspace(0, 100, arrays['n'])
    elif operation == 'logspace':
        return np.logspace(0, 3, arrays['n'])
    elif operation == 'geomspace':
        return np.geomspace(1, 1000, arrays['n'])
    elif operation == 'eye':
        return np.eye(arrays['n'])
    elif operation == 'identity':
        return np.identity(arrays['n'])
    elif operation == 'copy':
        return np.copy(arrays['a'])
    elif operation == 'zeros_like':
        return np.zeros_like(arrays['a'])
    elif operation == 'ones_like':
        return np.ones_like(arrays['a'])
    elif operation == 'empty_like':
        return np.empty_like(arrays['a'])
    elif operation == 'full_like':
        return np.full_like(arrays['a'], 7)

    # Arithmetic
    elif operation == 'add':
        return arrays['a'] + arrays['b']
    elif operation == 'subtract':
        return arrays['a'] - arrays['b']
    elif operation == 'multiply':
        return arrays['a'] * arrays['b']
    elif operation == 'divide':
        return arrays['a'] / arrays['b']

    # Linear algebra
    elif operation == 'matmul':
        return arrays['a'] @ arrays['b']
    elif operation == 'transpose':
        return arrays['a'].T

    # Reductions
    elif operation == 'sum':
        axis = arrays.get('axis')
        return arrays['a'].sum(axis=axis)
    elif operation == 'mean':
        axis = arrays.get('axis')
        return arrays['a'].mean(axis=axis)
    elif operation == 'max':
        axis = arrays.get('axis')
        return arrays['a'].max(axis=axis)
    elif operation == 'min':
        axis = arrays.get('axis')
        return arrays['a'].min(axis=axis)
    elif operation == 'prod':
        axis = arrays.get('axis')
        return arrays['a'].prod(axis=axis)
    elif operation == 'argmin':
        axis = arrays.get('axis')
        return arrays['a'].argmin(axis=axis)
    elif operation == 'argmax':
        axis = arrays.get('axis')
        return arrays['a'].argmax(axis=axis)
    elif operation == 'var':
        axis = arrays.get('axis')
        return arrays['a'].var(axis=axis)
    elif operation == 'std':
        axis = arrays.get('axis')
        return arrays['a'].std(axis=axis)
    elif operation == 'all':
        axis = arrays.get('axis')
        return arrays['a'].all(axis=axis)
    elif operation == 'any':
        axis = arrays.get('axis')
        return arrays['a'].any(axis=axis)

    # Reshape
    elif operation == 'reshape':
        return arrays['a'].reshape(arrays['new_shape'])
    elif operation == 'flatten':
        return arrays['a'].flatten()
    elif operation == 'ravel':
        return arrays['a'].ravel()
    elif operation == 'squeeze':
        return arrays['a'].squeeze()

    # Slicing
    elif operation == 'slice':
        return arrays['a'][:100, :100]

    else:
        raise ValueError(f"Unknown operation: {operation}")


def run_benchmark(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single benchmark and return timing results"""
    name = spec['name']
    operation = spec['operation']
    setup = spec['setup']
    iterations = spec['iterations']
    warmup = spec['warmup']

    # Setup arrays
    arrays = setup_arrays(setup)

    # Warmup
    for _ in range(warmup):
        execute_operation(operation, arrays)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = execute_operation(operation, arrays)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

        # Keep reference to prevent optimization
        _ = result

    times_array = np.array(times)

    return {
        'name': name,
        'mean_ms': float(np.mean(times_array)),
        'median_ms': float(np.median(times_array)),
        'min_ms': float(np.min(times_array)),
        'max_ms': float(np.max(times_array)),
        'std_ms': float(np.std(times_array))
    }


def main():
    """Main entry point - read specs from stdin, output results to stdout"""
    try:
        # Read benchmark specifications from stdin
        specs = json.loads(sys.stdin.read())

        results = []

        # Print environment info to stderr
        print(f"Python {sys.version.split()[0]}", file=sys.stderr)
        print(f"NumPy {np.__version__}", file=sys.stderr)
        print(f"Running {len(specs)} benchmarks...", file=sys.stderr)

        for i, spec in enumerate(specs, 1):
            result = run_benchmark(spec)
            results.append(result)

            # Print progress to stderr
            print(f"  [{i}/{len(specs)}] {spec['name']}: {result['mean_ms']:.3f}ms", file=sys.stderr)

        # Output results as JSON to stdout
        print(json.dumps(results, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
