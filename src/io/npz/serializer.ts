/**
 * NPZ file serializer
 *
 * Serializes multiple NDArrays to NPZ format (ZIP archive of .npy files).
 */

import { NDArray } from '../../core/ndarray';
import { serializeNpy } from '../npy/serializer';
import { writeZip, writeZipSync } from '../zip/writer';

/**
 * Options for serializing NPZ files
 */
export interface NpzSerializeOptions {
  /**
   * Whether to compress the NPZ file using DEFLATE.
   * Default: false (matches np.savez behavior; use true for np.savez_compressed behavior)
   */
  compress?: boolean;
}

/**
 * Serialize multiple arrays to NPZ format
 *
 * @param arrays - Map or object of array names to NDArrays
 * @param options - Serialization options
 * @returns Promise resolving to NPZ file as Uint8Array
 */
export async function serializeNpz(
  arrays: Map<string, NDArray> | Record<string, NDArray>,
  options: NpzSerializeOptions = {}
): Promise<Uint8Array> {
  const files = prepareNpzFiles(arrays);
  return writeZip(files, { compress: options.compress ?? false });
}

/**
 * Synchronously serialize multiple arrays to NPZ format (no compression)
 *
 * @param arrays - Map or object of array names to NDArrays
 * @returns NPZ file as Uint8Array
 */
export function serializeNpzSync(
  arrays: Map<string, NDArray> | Record<string, NDArray>
): Uint8Array {
  const files = prepareNpzFiles(arrays);
  return writeZipSync(files);
}

/**
 * Prepare NPY files for ZIP packaging
 */
function prepareNpzFiles(
  arrays: Map<string, NDArray> | Record<string, NDArray>
): Map<string, Uint8Array> {
  const files = new Map<string, Uint8Array>();

  // Handle both Map and plain object
  const entries =
    arrays instanceof Map ? arrays.entries() : Object.entries(arrays);

  for (const [name, arr] of entries) {
    // Validate array name
    if (typeof name !== 'string' || name.length === 0) {
      throw new Error('Array names must be non-empty strings');
    }

    // Serialize to NPY format
    const npyData = serializeNpy(arr);

    // Add .npy extension
    const fileName = name.endsWith('.npy') ? name : `${name}.npy`;
    files.set(fileName, npyData);
  }

  return files;
}
