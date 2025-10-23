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

// Version
export const __version__ = '0.0.1';
