/**
 * NumPy.js - Complete NumPy implementation for TypeScript and JavaScript
 *
 * @module numpy-js
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
} from './core/ndarray';

// Version (replaced at build time from package.json)
declare const __VERSION_PLACEHOLDER__: string;
export const __version__ = __VERSION_PLACEHOLDER__;
