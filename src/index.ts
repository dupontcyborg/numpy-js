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
  transpose,
} from './core/ndarray';

// Version (replaced at build time from package.json)
// In development/tests, use package.json directly; in production, use the replaced value
declare const __VERSION_PLACEHOLDER__: string;
export const __version__ =
  typeof __VERSION_PLACEHOLDER__ !== 'undefined' ? __VERSION_PLACEHOLDER__ : '0.2.0'; // Fallback for development/tests
