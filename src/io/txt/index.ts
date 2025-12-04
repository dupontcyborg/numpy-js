/**
 * Text I/O module for numpy-ts
 *
 * Provides parsing and serialization for delimited text formats (CSV, TSV, etc.).
 * These functions work with strings and are environment-agnostic.
 *
 * For file system operations, use the Node.js-specific entry point:
 *   import { loadtxt, savetxt } from 'numpy-ts/node';
 */

export { parseTxt, genfromtxt, fromregex, type ParseTxtOptions } from './parser';
export { serializeTxt, type SerializeTxtOptions } from './serializer';
