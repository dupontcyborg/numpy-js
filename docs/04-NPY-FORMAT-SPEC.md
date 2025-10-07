# .npy and .npz File Format Specification

## Overview

NumPy's `.npy` and `.npz` file formats provide efficient, portable binary storage for arrays. Our implementation must support full read/write compatibility with Python NumPy.

---

## .npy File Format

### Format Structure

```
┌─────────────────────────────────────────────────────────┐
│ Magic Number (6 bytes): \x93NUMPY                       │
├─────────────────────────────────────────────────────────┤
│ Version (2 bytes): Major (1 byte) + Minor (1 byte)      │
├─────────────────────────────────────────────────────────┤
│ Header Length (2 or 4 bytes, depending on version)      │
├─────────────────────────────────────────────────────────┤
│ Header (ASCII or UTF-8 encoded dictionary)              │
│   - "descr": dtype description                           │
│   - "fortran_order": boolean                             │
│   - "shape": tuple of integers                           │
├─────────────────────────────────────────────────────────┤
│ Array Data (contiguous bytes in C or Fortran order)     │
└─────────────────────────────────────────────────────────┘
```

### Version Specifications

#### Version 1.0
- **Header Length**: 2 bytes (little-endian uint16)
- **Maximum Header Size**: 65,535 bytes
- **Header Encoding**: ASCII
- **Introduced**: NumPy 1.0

#### Version 2.0
- **Header Length**: 4 bytes (little-endian uint32)
- **Maximum Header Size**: 4 GiB
- **Header Encoding**: ASCII
- **Use Case**: Large headers (many dimensions, long dtype names)
- **Introduced**: NumPy 1.9

#### Version 3.0
- **Header Length**: 4 bytes (little-endian uint32)
- **Maximum Header Size**: 4 GiB
- **Header Encoding**: UTF-8
- **Use Case**: Unicode field names in structured dtypes
- **Introduced**: NumPy 1.17 (planned for future)

### Magic Number

```typescript
const NPY_MAGIC = new Uint8Array([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]); // \x93NUMPY
```

The first byte (0x93) is chosen to:
- Be outside ASCII printable range
- Identify binary files
- Avoid conflicts with other formats

### Header Format

#### Header Dictionary
The header is a Python dictionary literal string containing:

```python
{
  'descr': '<f8',           # dtype descriptor
  'fortran_order': False,   # memory layout
  'shape': (100, 100),      # array dimensions
}
```

Requirements:
- Dictionary must be a valid Python literal
- Padded with spaces to make total header length divisible by 64 (for alignment)
- Ends with '\n'
- ASCII or UTF-8 encoded (depending on version)

#### Dtype Descriptor Format

The `descr` field describes the array's data type using NumPy's dtype notation:

**Scalar Types:**
```
'<f8'     # little-endian float64
'>i4'     # big-endian int32
'|u1'     # byte-order independent uint8
'<c16'    # little-endian complex128
```

**Byte Order Prefixes:**
- `<` - little-endian
- `>` - big-endian
- `|` - not applicable (single-byte types)
- `=` - native byte order

**Type Codes:**
- `b` - signed byte
- `B` - unsigned byte
- `i` - signed integer
- `u` - unsigned integer
- `f` - floating point
- `c` - complex floating point
- `m` - timedelta
- `M` - datetime
- `O` - object (Python pickle)
- `S` - zero-terminated bytes
- `U` - Unicode string
- `V` - raw data (void)

**Structured Types:**
```python
[('x', '<f4'), ('y', '<f4'), ('z', '<f4')]  # 3D point
```

#### Shape Tuple
- Empty tuple `()` for 0-D (scalar) arrays
- Single-element tuple `(n,)` for 1-D arrays
- Multi-element tuple `(n, m, ...)` for N-D arrays

#### Fortran Order
- `False`: C-contiguous (row-major)
- `True`: Fortran-contiguous (column-major)

### Detailed Write Algorithm

```typescript
function writeNPY(array: NDArray, path: string, version: [number, number] = [1, 0]): void {
  // 1. Create header dictionary
  const header = {
    descr: array.dtype.str,  // e.g., '<f8'
    fortran_order: array.flags.f_contiguous,
    shape: array.shape,
  };

  // 2. Serialize header to Python dict literal
  let headerStr = '{';
  headerStr += `'descr': '${header.descr}', `;
  headerStr += `'fortran_order': ${header.fortran_order}, `;
  headerStr += `'shape': (${header.shape.join(', ')}${header.shape.length === 1 ? ',' : ''}), `;
  headerStr += '}';

  // 3. Pad header to multiple of 64 bytes
  const [major, minor] = version;
  const headerLengthBytes = major === 1 ? 2 : 4;
  const prefixLength = 6 + 2 + headerLengthBytes; // magic + version + header_len

  let totalHeaderLength = prefixLength + headerStr.length + 1; // +1 for '\n'
  const padding = (64 - (totalHeaderLength % 64)) % 64;
  headerStr += ' '.repeat(padding) + '\n';

  const headerLength = headerStr.length;

  // 4. Write to buffer
  const buffer = new ArrayBuffer(prefixLength + headerLength + array.nbytes);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  // Magic number
  bytes.set(NPY_MAGIC, 0);

  // Version
  bytes[6] = major;
  bytes[7] = minor;

  // Header length
  if (major === 1) {
    view.setUint16(8, headerLength, true); // little-endian
  } else {
    view.setUint32(8, headerLength, true); // little-endian
  }

  // Header string
  const headerBytes = new TextEncoder().encode(headerStr);
  bytes.set(headerBytes, prefixLength);

  // Array data
  bytes.set(new Uint8Array(array.data.buffer), prefixLength + headerLength);

  // Write to file
  writeFile(path, bytes);
}
```

### Detailed Read Algorithm

```typescript
function readNPY(path: string): NDArray {
  // 1. Read file
  const bytes = readFile(path);
  const view = new DataView(bytes.buffer);

  // 2. Verify magic number
  for (let i = 0; i < 6; i++) {
    if (bytes[i] !== NPY_MAGIC[i]) {
      throw new Error('Invalid .npy file: bad magic number');
    }
  }

  // 3. Read version
  const major = bytes[6];
  const minor = bytes[7];

  if (major > 3) {
    throw new Error(`Unsupported .npy version: ${major}.${minor}`);
  }

  // 4. Read header length
  const headerLengthBytes = major === 1 ? 2 : 4;
  const headerLength = headerLengthBytes === 2
    ? view.getUint16(8, true)
    : view.getUint32(8, true);

  // 5. Parse header
  const prefixLength = 6 + 2 + headerLengthBytes;
  const headerBytes = bytes.slice(prefixLength, prefixLength + headerLength);
  const headerStr = new TextDecoder(major >= 3 ? 'utf-8' : 'ascii').decode(headerBytes);

  // Parse Python dict literal (need safe eval)
  const header = parsePythonDict(headerStr);

  // 6. Extract metadata
  const dtype = DType.fromString(header.descr);
  const shape = header.shape;
  const fortranOrder = header.fortran_order;

  // 7. Read array data
  const dataOffset = prefixLength + headerLength;
  const dataBytes = bytes.slice(dataOffset);

  // 8. Create array
  const array = new NDArray({
    buffer: dataBytes.buffer,
    dtype,
    shape,
    order: fortranOrder ? 'F' : 'C',
  });

  return array;
}
```

### Python Dict Parser

Since we need to parse Python dict literals, we have several options:

**Option 1: Safe Regex Parser**
```typescript
function parsePythonDict(s: string): any {
  // Remove trailing newline and whitespace
  s = s.trim();

  // Extract key-value pairs
  const descMatch = s.match(/'descr':\s*'([^']+)'/);
  const fortranMatch = s.match(/'fortran_order':\s*(True|False)/);
  const shapeMatch = s.match(/'shape':\s*\(([^)]*)\)/);

  if (!descMatch || !fortranMatch || !shapeMatch) {
    throw new Error('Invalid header format');
  }

  // Parse shape tuple
  const shapeStr = shapeMatch[1].trim();
  const shape = shapeStr === ''
    ? []
    : shapeStr.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x));

  return {
    descr: descMatch[1],
    fortran_order: fortranMatch[1] === 'True',
    shape: shape,
  };
}
```

**Option 2: Python Subprocess (Most Reliable)**
```typescript
async function parsePythonDict(s: string): Promise<any> {
  const result = await execPython(`
import json
import sys
header = ${s}
print(json.dumps({
    'descr': header['descr'],
    'fortran_order': header['fortran_order'],
    'shape': list(header['shape']),
}))
  `);
  return JSON.parse(result);
}
```

---

## .npz File Format

### Format Structure

An `.npz` file is a standard ZIP archive containing:
- One or more `.npy` files
- Each `.npy` file represents a named array
- File names inside ZIP correspond to array names

### Structure

```
archive.npz (ZIP file)
├── arr_0.npy          # First array
├── arr_1.npy          # Second array
├── array_name.npy     # Named array
└── ...
```

### Compressed vs Uncompressed

**Uncompressed (.npz):**
- Created with `np.savez()`
- Uses ZIP's STORED method (no compression)
- Faster to write/read
- Larger file size

**Compressed (.npz):**
- Created with `np.savez_compressed()`
- Uses ZIP's DEFLATE compression
- Slower to write/read
- Smaller file size
- Good for sparse data or integers

### Write Algorithm

```typescript
function saveNPZ(path: string, arrays: Record<string, NDArray>, compressed: boolean = false): void {
  const zip = new JSZip(); // Or use node's zlib + zip functionality

  for (const [name, array] of Object.entries(arrays)) {
    // Convert array to .npy format
    const npyBytes = arrayToNPYBytes(array);

    // Add to ZIP archive
    zip.file(`${name}.npy`, npyBytes, {
      compression: compressed ? 'DEFLATE' : 'STORE',
    });
  }

  // Write ZIP file
  zip.generateAsync({ type: 'uint8array' }).then(content => {
    writeFile(path, content);
  });
}
```

### Read Algorithm

```typescript
function loadNPZ(path: string): Record<string, NDArray> {
  const zipBytes = readFile(path);
  const zip = new JSZip();

  return zip.loadAsync(zipBytes).then(zip => {
    const arrays: Record<string, NDArray> = {};

    for (const [filename, file] of Object.entries(zip.files)) {
      // Extract .npy file
      const npyBytes = await file.async('uint8array');

      // Parse .npy format
      const array = parseNPYBytes(npyBytes);

      // Store with name (remove .npy extension)
      const name = filename.replace(/\.npy$/, '');
      arrays[name] = array;
    }

    return arrays;
  });
}
```

---

## Memory-Mapped Files

NumPy supports memory-mapped .npy files for efficient access to large arrays without loading into RAM.

### Requirements
- File must be in .npy format
- Only works with regular (non-object) dtypes
- Provides array-like interface backed by file

### Implementation Strategy

For Node.js, we can use:
```typescript
import * as fs from 'fs';
import * as mmap from 'mmap-io'; // or similar library

function memmap(path: string, mode: 'r' | 'r+' | 'w+' = 'r'): NDArray {
  // Read header
  const fd = fs.openSync(path, mode);
  const header = readNPYHeader(fd);

  // Memory-map the data region
  const dataOffset = header.offset;
  const dataLength = header.nbytes;

  const buffer = mmap.map(
    dataLength,
    mmap.PROT_READ | (mode !== 'r' ? mmap.PROT_WRITE : 0),
    mmap.MAP_SHARED,
    fd,
    dataOffset
  );

  // Create array backed by mmapped buffer
  return new NDArray({
    buffer: buffer,
    dtype: header.dtype,
    shape: header.shape,
    order: header.order,
  });
}
```

---

## Special Cases and Edge Cases

### 1. Zero-Dimensional Arrays
```python
arr = np.array(42)  # shape = ()
```
Header: `'shape': ()`

### 2. Empty Arrays
```python
arr = np.array([])  # shape = (0,)
```
Header: `'shape': (0,)`

### 3. Object Arrays
Arrays with dtype `object` contain Python pickles:
```python
arr = np.array(['hello', {'key': 'value'}, [1, 2, 3]])
```

Our implementation options:
- **Full support**: Include a JavaScript pickle parser (complex)
- **Limited support**: Warn and skip object arrays
- **Hybrid**: Support simple object types (strings), error on complex pickles

### 4. Structured Arrays
```python
dt = np.dtype([('x', 'f4'), ('y', 'f4')])
arr = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=dt)
```

Header: `'descr': [('x', '<f4'), ('y', '<f4')]`

### 5. Byte Order Conversion
If file byte order doesn't match system:
- Parse byte order from dtype descriptor
- Perform byte swapping during read

### 6. Unicode Field Names (Version 3.0)
```python
dt = np.dtype([('时间', 'f4'), ('数据', 'f4')])
```

Requires UTF-8 header encoding (version 3.0+)

---

## Testing Strategy

### Compatibility Tests

Create test files with Python NumPy:
```python
# test_files/generate_test_npys.py
import numpy as np

# Various dtypes
np.save('int32.npy', np.array([1, 2, 3], dtype='int32'))
np.save('float64.npy', np.array([1.5, 2.5], dtype='float64'))
np.save('complex128.npy', np.array([1+2j, 3+4j], dtype='complex128'))

# Various shapes
np.save('scalar.npy', np.array(42))
np.save('empty.npy', np.array([]))
np.save('3d.npy', np.arange(24).reshape(2, 3, 4))

# Fortran order
arr = np.array([[1, 2], [3, 4]], order='F')
np.save('fortran_order.npy', arr)

# Structured dtype
dt = np.dtype([('x', 'f4'), ('y', 'f4')])
np.save('structured.npy', np.array([(1, 2), (3, 4)], dtype=dt))

# NPZ files
np.savez('archive.npz', a=np.ones(5), b=np.zeros(3))
np.savez_compressed('compressed.npz', x=np.arange(1000))
```

Then test round-trip compatibility:
```typescript
describe('.npy format compatibility', () => {
  it('should read Python-generated .npy files', () => {
    const arr = np.load('test_files/int32.npy');
    expect(arr.tolist()).toEqual([1, 2, 3]);
    expect(arr.dtype.name).toBe('int32');
  });

  it('should write .npy files readable by Python', async () => {
    const arr = np.array([1, 2, 3], { dtype: 'int32' });
    np.save('output.npy', arr);

    // Validate with Python
    const result = await execPython(`
import numpy as np
arr = np.load('output.npy')
print(arr.tolist())
print(arr.dtype)
    `);

    expect(result).toContain('[1, 2, 3]');
    expect(result).toContain('int32');
  });
});
```

---

## Implementation Priority

### Phase 1: Basic .npy Support
- Read/write version 1.0 format
- Support common dtypes (int, float)
- C-order only
- Test compatibility with Python NumPy

### Phase 2: Extended .npy Support
- Version 2.0 support (large headers)
- Fortran order
- All numeric dtypes
- Structured dtypes

### Phase 3: .npz Support
- Read/write uncompressed .npz
- Compressed .npz with zlib

### Phase 4: Advanced Features
- Memory-mapped files
- Object arrays (limited support)
- Version 3.0 (UTF-8 headers)

---

## Dependencies

### Required
- **TypedArray support**: For efficient binary I/O
- **TextEncoder/TextDecoder**: For header encoding
- **DataView**: For byte order handling

### Optional
- **ZIP library**: JSZip (browser), node-zip, or built-in zlib (Node.js)
- **Memory mapping**: mmap-io or mmap.js (Node.js only)

---

## Reference Implementation

Python NumPy's format module:
- [numpy/lib/format.py](https://github.com/numpy/numpy/blob/main/numpy/lib/format.py)
- Authoritative source for .npy format

---

**Last Updated**: 2025-10-07
**Status**: Complete specification for .npy/.npz support
