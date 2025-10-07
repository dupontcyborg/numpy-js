# DType Specification

## Overview

NumPy.js will support all major NumPy data types. This document specifies which dtypes we support, how they're implemented in JavaScript/TypeScript, and any limitations.

---

## DType Categories

NumPy organizes dtypes into categories by "kind":
- **b** - boolean
- **i** - signed integer
- **u** - unsigned integer
- **f** - floating-point
- **c** - complex floating-point
- **m** - timedelta
- **M** - datetime
- **O** - object (Python objects)
- **S** - byte string (fixed length)
- **U** - Unicode string (fixed length)
- **V** - void (raw data)

---

## Supported DTypes

### 1. Boolean

| NumPy dtype | JS Implementation | Size | Notes |
|-------------|-------------------|------|-------|
| `bool` | `Uint8Array` (0 or 1) | 1 byte | Use uint8 with 0/1 values |

**Implementation:**
```typescript
class BoolDType extends DType {
  readonly name = 'bool';
  readonly kind = 'b';
  readonly itemsize = 1;
  readonly arrayType = Uint8Array;
  readonly byteorder = '|'; // not applicable

  cast(value: any): number {
    return value ? 1 : 0;
  }
}
```

**Usage:**
```typescript
const arr = np.array([true, false, true], { dtype: 'bool' });
// Stored as Uint8Array([1, 0, 1])
```

---

### 2. Integers (Signed)

| NumPy dtype | JS Implementation | Size | Range | Notes |
|-------------|-------------------|------|-------|-------|
| `int8` | `Int8Array` | 1 byte | -128 to 127 | ✅ Full support |
| `int16` | `Int16Array` | 2 bytes | -32,768 to 32,767 | ✅ Full support |
| `int32` | `Int32Array` | 4 bytes | -2³¹ to 2³¹-1 | ✅ Full support |
| `int64` | `BigInt64Array` | 8 bytes | -2⁶³ to 2⁶³-1 | ⚠️ BigInt required |

**int64 Implementation Strategy:**

**Option A (Recommended): Use BigInt**
```typescript
class Int64DType extends DType {
  readonly name = 'int64';
  readonly kind = 'i';
  readonly itemsize = 8;
  readonly arrayType = BigInt64Array;

  cast(value: any): bigint {
    return BigInt(value);
  }
}

// Usage
const arr = np.array([1n, 2n, 3n], { dtype: 'int64' });
arr.item(0); // 1n (bigint)
```

**Pros:**
- Exact representation of all int64 values
- Native JavaScript support
- No precision loss

**Cons:**
- Different type (bigint vs number)
- Some operations need special handling
- Slightly slower than number

**Option B (Fallback): Float64 with warnings**
```typescript
class Int64DType extends DType {
  cast(value: any): number {
    const num = Number(value);
    if (!Number.isSafeInteger(num)) {
      console.warn(`int64 value ${value} cannot be exactly represented`);
    }
    return num;
  }
}
```

**Recommendation**: Use BigInt (Option A) as primary, with configuration option to fallback to Float64

---

### 3. Integers (Unsigned)

| NumPy dtype | JS Implementation | Size | Range | Notes |
|-------------|-------------------|------|-------|-------|
| `uint8` | `Uint8Array` | 1 byte | 0 to 255 | ✅ Full support |
| `uint16` | `Uint16Array` | 2 bytes | 0 to 65,535 | ✅ Full support |
| `uint32` | `Uint32Array` | 4 bytes | 0 to 2³²-1 | ✅ Full support |
| `uint64` | `BigUint64Array` | 8 bytes | 0 to 2⁶⁴-1 | ⚠️ BigInt required |

**uint64 Implementation:**
Same approach as int64 - use BigUint64Array with BigInt.

---

### 4. Floating Point

| NumPy dtype | JS Implementation | Size | Precision | Notes |
|-------------|-------------------|------|-----------|-------|
| `float16` | `Uint16Array` + manual conversion | 2 bytes | ~3 decimal digits | ⚠️ No native support |
| `float32` | `Float32Array` | 4 bytes | ~7 decimal digits | ✅ Full support |
| `float64` | `Float64Array` | 8 bytes | ~16 decimal digits | ✅ Full support (default) |

**float16 (Half Precision) Implementation:**

JavaScript has no native float16, so we need manual conversion:

```typescript
class Float16DType extends DType {
  readonly name = 'float16';
  readonly kind = 'f';
  readonly itemsize = 2;
  readonly arrayType = Uint16Array; // Store as uint16

  // IEEE 754 half-precision conversion
  toFloat32(uint16: number): number {
    const sign = (uint16 & 0x8000) >> 15;
    const exponent = (uint16 & 0x7C00) >> 10;
    const fraction = uint16 & 0x03FF;

    if (exponent === 0) {
      // Subnormal or zero
      return (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
    } else if (exponent === 31) {
      // Inf or NaN
      return fraction ? NaN : (sign ? -Infinity : Infinity);
    } else {
      // Normal
      return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }
  }

  fromFloat32(float32: number): number {
    // Convert float32 to uint16 representation
    // ... (reverse of above)
  }
}
```

**Priority**: Lower priority - implement after core functionality working

---

### 5. Complex Numbers

| NumPy dtype | JS Implementation | Size | Component Type | Notes |
|-------------|-------------------|------|----------------|-------|
| `complex64` | `Float32Array` (interleaved) | 8 bytes | 2x float32 | ⚠️ Manual implementation |
| `complex128` | `Float64Array` (interleaved) | 16 bytes | 2x float64 | ⚠️ Manual implementation |

**Implementation Strategy: Interleaved Storage**

```typescript
class Complex128DType extends DType {
  readonly name = 'complex128';
  readonly kind = 'c';
  readonly itemsize = 16; // 2 * 8 bytes
  readonly arrayType = Float64Array;

  // Array layout: [real0, imag0, real1, imag1, real2, imag2, ...]

  getReal(data: Float64Array, index: number): number {
    return data[index * 2];
  }

  getImag(data: Float64Array, index: number): number {
    return data[index * 2 + 1];
  }

  setReal(data: Float64Array, index: number, value: number): void {
    data[index * 2] = value;
  }

  setImag(data: Float64Array, index: number, value: number): void {
    data[index * 2 + 1] = value;
  }

  // Complex operations
  add(a: Complex, b: Complex): Complex {
    return {
      real: a.real + b.real,
      imag: a.imag + b.imag,
    };
  }

  multiply(a: Complex, b: Complex): Complex {
    return {
      real: a.real * b.real - a.imag * b.imag,
      imag: a.real * b.imag + a.imag * b.real,
    };
  }
}
```

**Usage:**
```typescript
const arr = np.array([{real: 1, imag: 2}, {real: 3, imag: 4}], { dtype: 'complex128' });
// Stored as Float64Array([1, 2, 3, 4])

arr.real; // NDArray([1, 3])
arr.imag; // NDArray([2, 4])
np.abs(arr); // NDArray([sqrt(5), 5])
np.angle(arr); // NDArray([atan2(2,1), atan2(4,3)])
```

---

### 6. Datetime

| NumPy dtype | JS Implementation | Size | Notes |
|-------------|-------------------|------|-------|
| `datetime64` | `BigInt64Array` | 8 bytes | Time since epoch in specified unit |
| `timedelta64` | `BigInt64Array` | 8 bytes | Time difference in specified unit |

**Units Supported:**
- `Y` - Year
- `M` - Month
- `W` - Week
- `D` - Day
- `h` - Hour
- `m` - Minute
- `s` - Second
- `ms` - Millisecond
- `us` - Microsecond
- `ns` - Nanosecond (most precise)

**Implementation:**
```typescript
class Datetime64DType extends DType {
  readonly name: string; // 'datetime64[ns]', 'datetime64[ms]', etc.
  readonly kind = 'M';
  readonly itemsize = 8;
  readonly arrayType = BigInt64Array;
  readonly unit: DatetimeUnit;

  constructor(unit: DatetimeUnit = 'ns') {
    super();
    this.unit = unit;
    this.name = `datetime64[${unit}]`;
  }

  // Convert JavaScript Date to datetime64
  fromDate(date: Date): bigint {
    const ms = BigInt(date.getTime());
    return this.convertUnit(ms, 'ms', this.unit);
  }

  // Convert datetime64 to JavaScript Date
  toDate(value: bigint): Date {
    const ms = this.convertUnit(value, this.unit, 'ms');
    return new Date(Number(ms));
  }

  convertUnit(value: bigint, fromUnit: DatetimeUnit, toUnit: DatetimeUnit): bigint {
    // Conversion logic
  }
}
```

**Usage:**
```typescript
const dates = np.array(['2024-01-01', '2024-12-31'], { dtype: 'datetime64[D]' });
const delta = dates[1] - dates[0]; // timedelta64[D]
console.log(delta.item()); // 365 days
```

**Priority**: Medium - implement after core math functions

---

### 7. Strings

#### 7.1 Byte Strings (Fixed Length)

| NumPy dtype | JS Implementation | Size | Notes |
|-------------|-------------------|------|-------|
| `S<n>` (e.g., `S10`) | `Uint8Array` | n bytes | Fixed-length byte strings |

**Implementation:**
```typescript
class ByteStringDType extends DType {
  readonly kind = 'S';
  readonly itemsize: number;
  readonly arrayType = Uint8Array;

  constructor(length: number) {
    super();
    this.itemsize = length;
    this.name = `S${length}`;
  }

  toString(data: Uint8Array, index: number): string {
    const start = index * this.itemsize;
    const end = start + this.itemsize;
    const bytes = data.slice(start, end);

    // Null-terminated
    const nullIndex = bytes.indexOf(0);
    const validBytes = nullIndex >= 0 ? bytes.slice(0, nullIndex) : bytes;

    return new TextDecoder('latin1').decode(validBytes);
  }

  fromString(str: string): Uint8Array {
    const encoded = new TextEncoder().encode(str);
    const padded = new Uint8Array(this.itemsize);
    padded.set(encoded.slice(0, this.itemsize));
    return padded;
  }
}
```

#### 7.2 Unicode Strings (Fixed Length)

| NumPy dtype | JS Implementation | Size | Notes |
|-------------|-------------------|------|-------|
| `U<n>` (e.g., `U10`) | `Uint32Array` | n * 4 bytes | Fixed-length Unicode (UCS-4) |

**Implementation:**
```typescript
class UnicodeStringDType extends DType {
  readonly kind = 'U';
  readonly itemsize: number; // length * 4
  readonly arrayType = Uint32Array;
  readonly length: number;

  constructor(length: number) {
    super();
    this.length = length;
    this.itemsize = length * 4;
    this.name = `U${length}`;
  }

  toString(data: Uint32Array, index: number): string {
    const start = index * this.length;
    const end = start + this.length;
    const codePoints = Array.from(data.slice(start, end));

    // Stop at null terminator
    const nullIndex = codePoints.indexOf(0);
    const validCodes = nullIndex >= 0 ? codePoints.slice(0, nullIndex) : codePoints;

    return String.fromCodePoint(...validCodes);
  }

  fromString(str: string): Uint32Array {
    const codePoints = Array.from(str, c => c.codePointAt(0) || 0);
    const padded = new Uint32Array(this.length);
    padded.set(codePoints.slice(0, this.length));
    return padded;
  }
}
```

**Note**: NumPy 2.0 introduced variable-length `StringDType` which we may support later

**Priority**: Lower - implement after core functionality

---

### 8. Structured DTypes

Structured dtypes allow multiple fields with different types in a single array element.

**Example:**
```python
dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('name', 'U10')])
arr = np.array([(1.0, 2.0, 'point1'), (3.0, 4.0, 'point2')], dtype=dt)
```

**Implementation:**
```typescript
class StructuredDType extends DType {
  readonly kind = 'V';
  readonly name: string;
  readonly itemsize: number;
  readonly fields: Map<string, FieldDescriptor>;

  constructor(fields: FieldDescriptor[]) {
    super();
    this.fields = new Map();

    let offset = 0;
    for (const field of fields) {
      // Alignment
      const alignment = field.dtype.itemsize;
      if (offset % alignment !== 0) {
        offset += alignment - (offset % alignment);
      }

      this.fields.set(field.name, {
        ...field,
        offset,
      });

      offset += field.dtype.itemsize;
    }

    this.itemsize = offset;
    this.name = this.computeName();
  }

  getField(data: Uint8Array, index: number, fieldName: string): any {
    const field = this.fields.get(fieldName);
    if (!field) throw new Error(`Field ${fieldName} not found`);

    const byteOffset = index * this.itemsize + field.offset;
    return field.dtype.getValue(data, byteOffset);
  }

  setField(data: Uint8Array, index: number, fieldName: string, value: any): void {
    const field = this.fields.get(fieldName);
    if (!field) throw new Error(`Field ${fieldName} not found`);

    const byteOffset = index * this.itemsize + field.offset;
    field.dtype.setValue(data, byteOffset, value);
  }
}

interface FieldDescriptor {
  name: string;
  dtype: DType;
  offset?: number;
}
```

**Usage:**
```typescript
const dt = np.dtype([
  ['x', 'float32'],
  ['y', 'float32'],
  ['name', 'U10'],
]);

const arr = np.array([
  { x: 1.0, y: 2.0, name: 'point1' },
  { x: 3.0, y: 4.0, name: 'point2' },
], { dtype: dt });

arr['x']; // NDArray([1.0, 3.0])
arr['name']; // NDArray(['point1', 'point2'])
```

**Priority**: Medium - implement after basic dtypes working

---

### 9. Object DType

NumPy's `object` dtype stores arbitrary Python objects (via pickle).

**Implementation Strategy:**

**Option A: JavaScript Objects (Recommended)**
```typescript
class ObjectDType extends DType {
  readonly kind = 'O';
  readonly name = 'object';
  readonly itemsize = 8; // Pointer size (conceptual)
  readonly arrayType = Array; // Use regular JS array

  // Store references to JS objects
  // No pickle compatibility (different language)
}
```

**Limitations:**
- No pickle compatibility (Python-specific)
- Can't round-trip with Python NumPy object arrays
- Supports only JavaScript-serializable objects

**Option B: Limited Support with Warning**
```typescript
function createObjectArray(data: any[]): never {
  throw new Error(
    'Object arrays not fully supported. ' +
    'NumPy object dtype stores Python pickles which cannot be used in JavaScript. ' +
    'Consider using structured dtypes or JSON serialization.'
  );
}
```

**Recommendation**: Option A with clear documentation about limitations

**Priority**: Lower - implement after core functionality

---

### 10. Void (Raw Data)

| NumPy dtype | JS Implementation | Size | Notes |
|-------------|-------------------|------|-------|
| `V<n>` (e.g., `V8`) | `Uint8Array` | n bytes | Uninterpreted bytes |

**Implementation:**
```typescript
class VoidDType extends DType {
  readonly kind = 'V';
  readonly itemsize: number;
  readonly arrayType = Uint8Array;

  constructor(itemsize: number) {
    super();
    this.itemsize = itemsize;
    this.name = `V${itemsize}`;
  }
}
```

**Usage:** Primarily for structured dtypes and raw data.

**Priority**: Low

---

## DType Hierarchy

```
DType (abstract base class)
├── NumericDType
│   ├── IntegerDType
│   │   ├── Int8DType
│   │   ├── Int16DType
│   │   ├── Int32DType
│   │   ├── Int64DType
│   │   ├── UInt8DType
│   │   ├── UInt16DType
│   │   ├── UInt32DType
│   │   └── UInt64DType
│   ├── FloatDType
│   │   ├── Float16DType
│   │   ├── Float32DType
│   │   └── Float64DType
│   └── ComplexDType
│       ├── Complex64DType
│       └── Complex128DType
├── BoolDType
├── StringDType
│   ├── ByteStringDType (S)
│   └── UnicodeStringDType (U)
├── DatetimeDType
│   ├── Datetime64DType
│   └── Timedelta64DType
├── StructuredDType
├── VoidDType
└── ObjectDType
```

---

## Type Promotion Rules

When operating on arrays with different dtypes, NumPy promotes to a common type:

```typescript
function promoteTypes(dt1: DType, dt2: DType): DType {
  // Simplified promotion rules:
  // 1. If same type, return that type
  if (dt1.equals(dt2)) return dt1;

  // 2. Bool promotes to any other type
  if (dt1.kind === 'b') return dt2;
  if (dt2.kind === 'b') return dt1;

  // 3. Integer + Integer -> larger integer
  if (dt1.kind === 'i' && dt2.kind === 'i') {
    return dt1.itemsize >= dt2.itemsize ? dt1 : dt2;
  }

  // 4. Integer + Float -> Float
  if ((dt1.kind === 'i' && dt2.kind === 'f') || (dt1.kind === 'f' && dt2.kind === 'i')) {
    return dt1.kind === 'f' ? dt1 : dt2;
  }

  // 5. Float + Float -> larger float
  if (dt1.kind === 'f' && dt2.kind === 'f') {
    return dt1.itemsize >= dt2.itemsize ? dt1 : dt2;
  }

  // 6. Any + Complex -> Complex
  if (dt1.kind === 'c' || dt2.kind === 'c') {
    // Return complex with sufficient precision
    if (dt1.kind === 'c') return dt1;
    if (dt2.kind === 'c') return dt2;
  }

  // 7. Otherwise, try float64 as common ground
  return dtypes.float64;
}
```

**Important**: Our promotion rules must match NumPy exactly. Use extensive testing.

---

## DType String Descriptors

NumPy uses string descriptors to specify dtypes:

```typescript
function parseDTypeString(descriptor: string): DType {
  // Format: [byteorder]kind[size]
  // Examples:
  //   '<f8' - little-endian float64
  //   '>i4' - big-endian int32
  //   '|b1' - bool (byte order irrelevant)

  const match = descriptor.match(/^([<>|=])?([biufcmMOSUV])(\d+)$/);
  if (!match) throw new Error(`Invalid dtype descriptor: ${descriptor}`);

  const [, byteorder, kind, size] = match;
  const itemsize = parseInt(size);

  // Map to dtype
  const key = `${kind}${itemsize * 8}`; // e.g., 'f64', 'i32'
  return dtypes[key] || new GenericDType(kind, itemsize, byteorder);
}
```

**Byte Order:**
- `<` - little-endian
- `>` - big-endian
- `|` - not applicable (single-byte)
- `=` - native (system byte order)

**Note**: JavaScript TypedArrays use system byte order, so we may need DataView for cross-platform byte order handling.

---

## Implementation Priority

### Phase 1: Essential Types
- [x] `bool`
- [x] `int8`, `int16`, `int32`
- [x] `uint8`, `uint16`, `uint32`
- [x] `float32`, `float64`

### Phase 2: Extended Numeric
- [ ] `int64`, `uint64` (with BigInt)
- [ ] `complex64`, `complex128`

### Phase 3: Specialized
- [ ] `datetime64`, `timedelta64`
- [ ] Structured dtypes
- [ ] `float16`

### Phase 4: Strings and Advanced
- [ ] `S<n>` (byte strings)
- [ ] `U<n>` (Unicode strings)
- [ ] `object` (limited support)
- [ ] `V<n>` (void)

---

## Summary Table

| DType | JS Storage | Priority | Limitations |
|-------|-----------|----------|-------------|
| `bool` | `Uint8Array` | ✅ P1 | None |
| `int8` | `Int8Array` | ✅ P1 | None |
| `int16` | `Int16Array` | ✅ P1 | None |
| `int32` | `Int32Array` | ✅ P1 | None |
| `int64` | `BigInt64Array` | ⚠️ P2 | BigInt required |
| `uint8` | `Uint8Array` | ✅ P1 | None |
| `uint16` | `Uint16Array` | ✅ P1 | None |
| `uint32` | `Uint32Array` | ✅ P1 | None |
| `uint64` | `BigUint64Array` | ⚠️ P2 | BigInt required |
| `float16` | `Uint16Array` + conversion | ⚠️ P3 | Manual conversion |
| `float32` | `Float32Array` | ✅ P1 | None |
| `float64` | `Float64Array` | ✅ P1 | Default type |
| `complex64` | `Float32Array` (interleaved) | ⚠️ P2 | Manual operations |
| `complex128` | `Float64Array` (interleaved) | ⚠️ P2 | Manual operations |
| `datetime64` | `BigInt64Array` | ⚠️ P3 | Unit conversions |
| `timedelta64` | `BigInt64Array` | ⚠️ P3 | Unit conversions |
| `S<n>` | `Uint8Array` | 🔵 P4 | Fixed length |
| `U<n>` | `Uint32Array` | 🔵 P4 | Fixed length |
| Structured | Mixed | ⚠️ P3 | Complex |
| `object` | `Array` | 🔵 P4 | No pickle compatibility |
| `V<n>` | `Uint8Array` | 🔵 P4 | Rare usage |

---

**Last Updated**: 2025-10-07
**Status**: Complete dtype specification
