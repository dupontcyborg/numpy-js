/**
 * numpy-ts - Complete NumPy implementation for TypeScript and JavaScript
 *
 * @module numpy-ts
 */

// Core array functions
export {
  NDArray,
  zeros,
  ones,
  array,
  arange,
  linspace,
  logspace,
  geomspace,
  eye,
  empty,
  full,
  identity,
  asarray,
  copy,
  zeros_like,
  ones_like,
  empty_like,
  full_like,
  sqrt,
  power,
  absolute,
  negative,
  sign,
  mod,
  floor_divide,
  positive,
  reciprocal,
  dot,
  trace,
  diagonal,
  kron,
  transpose,
  inner,
  outer,
  tensordot,
  // Trigonometric functions
  sin,
  cos,
  tan,
  arcsin,
  arccos,
  arctan,
  arctan2,
  hypot,
  degrees,
  radians,
  deg2rad,
  rad2deg,
  // Hyperbolic functions
  sinh,
  cosh,
  tanh,
  arcsinh,
  arccosh,
  arctanh,
  // Array manipulation
  swapaxes,
  moveaxis,
  concatenate,
  stack,
  vstack,
  hstack,
  dstack,
  split,
  array_split,
  vsplit,
  hsplit,
  tile,
  repeat,
  // Advanced
  broadcast_to,
  broadcast_arrays,
  broadcast_shapes,
  take,
  put,
  choose,
  array_equal,
  array_equiv,
} from './core/ndarray';

// IO functions (environment-agnostic parsing/serialization)
// These work with bytes (ArrayBuffer/Uint8Array), not files
export {
  // NPY format
  parseNpy,
  serializeNpy,
  parseNpyHeader,
  parseNpyData,
  UnsupportedDTypeError,
  InvalidNpyError,
  SUPPORTED_DTYPES,
  DTYPE_TO_DESCR,
  type NpyHeader,
  type NpyMetadata,
  type NpyVersion,
  // NPZ format
  parseNpz,
  parseNpzSync,
  loadNpz,
  loadNpzSync,
  serializeNpz,
  serializeNpzSync,
  type NpzParseOptions,
  type NpzParseResult,
  type NpzSerializeOptions,
} from './io';

// Version (replaced at build time from package.json)
// In development/tests, use package.json directly; in production, use the replaced value
declare const __VERSION_PLACEHOLDER__: string;
export const __version__ =
  typeof __VERSION_PLACEHOLDER__ !== 'undefined' ? __VERSION_PLACEHOLDER__ : '0.5.0'; // Fallback for development/tests
