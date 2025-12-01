#!/usr/bin/env python3
"""
Validation script for benchmark correctness
Runs operations with NumPy and returns results for comparison
"""

import json
import sys
import numpy as np


def setup_arrays(setup_config):
    """Create NumPy arrays from setup configuration"""
    arrays = {}

    for key, config in setup_config.items():
        shape = config["shape"]
        dtype = config.get("dtype", "float64")
        fill = config.get("fill", "zeros")
        value = config.get("value")

        # Handle scalar values
        if key in ["n", "axis", "new_shape", "shape", "fill_value", "target_shape"]:
            arrays[key] = shape[0]
            if key in ["new_shape", "shape", "target_shape"]:
                arrays[key] = shape
            continue

        # Handle indices array
        if key == "indices":
            arrays[key] = shape
            continue

        # Map dtype names
        np_dtype = dtype
        if dtype == "bool":
            np_dtype = "bool_"

        # Create arrays
        if value is not None:
            arrays[key] = np.full(shape, value, dtype=np_dtype)
        elif fill == "zeros":
            arrays[key] = np.zeros(shape, dtype=np_dtype)
        elif fill == "ones":
            arrays[key] = np.ones(shape, dtype=np_dtype)
        elif fill in ["random", "arange"]:
            size = np.prod(shape)
            flat = np.arange(0, size, 1, dtype=np_dtype)
            arrays[key] = flat.reshape(shape)

    return arrays


def run_operation(spec):
    """Run a single operation and return result"""
    arrays = setup_arrays(spec["setup"])
    operation = spec["operation"]

    # Execute operation
    if operation == "zeros":
        result = np.zeros(arrays["shape"])
    elif operation == "ones":
        result = np.ones(arrays["shape"])
    elif operation == "arange":
        result = np.arange(0, arrays["n"])
    elif operation == "linspace":
        result = np.linspace(0, 100, arrays["n"])
    elif operation == "logspace":
        result = np.logspace(0, 3, arrays["n"])
    elif operation == "geomspace":
        result = np.geomspace(1, 1000, arrays["n"])
    elif operation == "eye":
        result = np.eye(arrays["n"])
    elif operation == "identity":
        result = np.identity(arrays["n"])
    elif operation == "empty":
        result = np.empty(arrays["shape"])
        # For empty, just return zeros (we can't compare uninitialized data)
        result = np.zeros(arrays["shape"])
    elif operation == "full":
        result = np.full(arrays["shape"], arrays["fill_value"])
    elif operation == "copy":
        result = arrays["a"].copy()
    elif operation == "zeros_like":
        result = np.zeros_like(arrays["a"])

    # Arithmetic
    elif operation == "add":
        result = arrays["a"] + (
            arrays.get("b") if "b" in arrays else arrays.get("scalar")
        )
    elif operation == "multiply":
        result = arrays["a"] * (
            arrays.get("b") if "b" in arrays else arrays.get("scalar")
        )
    elif operation == "mod":
        divisor = arrays.get("b") if "b" in arrays else arrays.get("scalar")
        result = np.mod(arrays["a"], divisor)
    elif operation == "floor_divide":
        divisor = arrays.get("b") if "b" in arrays else arrays.get("scalar")
        result = np.floor_divide(arrays["a"], divisor)
    elif operation == "reciprocal":
        result = np.reciprocal(arrays["a"])

    # Math
    elif operation == "sqrt":
        result = np.sqrt(arrays["a"])
    elif operation == "power":
        result = np.power(arrays["a"], 2)
    elif operation == "absolute":
        result = np.absolute(arrays["a"])
    elif operation == "negative":
        result = np.negative(arrays["a"])
    elif operation == "sign":
        result = np.sign(arrays["a"])

    # Trigonometric
    elif operation == "sin":
        result = np.sin(arrays["a"])
    elif operation == "cos":
        result = np.cos(arrays["a"])
    elif operation == "tan":
        result = np.tan(arrays["a"])
    elif operation == "arctan2":
        result = np.arctan2(arrays["a"], arrays["b"])
    elif operation == "hypot":
        result = np.hypot(arrays["a"], arrays["b"])

    # Hyperbolic
    elif operation == "sinh":
        result = np.sinh(arrays["a"])
    elif operation == "cosh":
        result = np.cosh(arrays["a"])
    elif operation == "tanh":
        result = np.tanh(arrays["a"])

    # Linalg
    elif operation == "dot":
        result = np.dot(arrays["a"], arrays["b"])
    elif operation == "inner":
        result = np.inner(arrays["a"], arrays["b"])
    elif operation == "outer":
        result = np.outer(arrays["a"], arrays["b"])
    elif operation == "matmul":
        result = arrays["a"] @ arrays["b"]
    elif operation == "trace":
        result = np.trace(arrays["a"])
    elif operation == "transpose":
        result = arrays["a"].T

    # Reductions
    elif operation == "sum":
        result = arrays["a"].sum(axis=arrays.get("axis"))
    elif operation == "mean":
        result = arrays["a"].mean()
    elif operation == "max":
        result = arrays["a"].max()
    elif operation == "min":
        result = arrays["a"].min()
    elif operation == "prod":
        result = arrays["a"].prod()
    elif operation == "argmin":
        result = arrays["a"].argmin()
    elif operation == "argmax":
        result = arrays["a"].argmax()
    elif operation == "var":
        result = arrays["a"].var()
    elif operation == "std":
        result = arrays["a"].std()
    elif operation == "all":
        result = arrays["a"].all()
    elif operation == "any":
        result = arrays["a"].any()

    # Reshape
    elif operation == "reshape":
        result = arrays["a"].reshape(*arrays["new_shape"])
    elif operation == "flatten":
        result = arrays["a"].flatten()
    elif operation == "ravel":
        result = arrays["a"].ravel()

    # Array manipulation
    elif operation == "swapaxes":
        result = np.swapaxes(arrays["a"], 0, 1)
    elif operation == "concatenate":
        result = np.concatenate([arrays["a"], arrays["b"]], axis=0)
    elif operation == "stack":
        result = np.stack([arrays["a"], arrays["b"]], axis=0)
    elif operation == "vstack":
        result = np.vstack([arrays["a"], arrays["b"]])
    elif operation == "hstack":
        result = np.hstack([arrays["a"], arrays["b"]])
    elif operation == "tile":
        result = np.tile(arrays["a"], [2, 2])
    elif operation == "repeat":
        result = np.repeat(arrays["a"], 2)

    # Advanced
    elif operation == "broadcast_to":
        result = np.broadcast_to(arrays["a"], arrays["target_shape"])
    elif operation == "take":
        result = np.take(arrays["a"], arrays["indices"])

    else:
        raise ValueError(f"Unknown operation: {operation}")

    # Convert result to JSON-serializable format
    if isinstance(result, np.ndarray):
        return {"shape": result.shape, "data": result.tolist()}
    elif isinstance(result, (np.integer, np.floating)):
        return float(result)
    elif isinstance(result, np.bool_):
        return bool(result)
    else:
        return result


import math

def serialize_value(val):
    """Recursively serialize values, handling Infinity and NaN"""
    if isinstance(val, dict):
        return {k: serialize_value(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [serialize_value(v) for v in val]
    elif isinstance(val, float):
        if math.isnan(val):
            return "__NaN__"
        elif math.isinf(val):
            return "__Infinity__" if val > 0 else "__-Infinity__"
        return val
    return val


def main():
    # Read specs from stdin
    input_data = json.loads(sys.stdin.read())
    specs = input_data["specs"]

    results = []

    for spec in specs:
        try:
            result = run_operation(spec)
            # Serialize to handle Infinity/NaN
            result = serialize_value(result)
            results.append(result)
        except Exception as e:
            print(f"Error running {spec['name']}: {e}", file=sys.stderr)
            results.append(None)

    # Output results as JSON
    print(json.dumps(results))


if __name__ == "__main__":
    main()
