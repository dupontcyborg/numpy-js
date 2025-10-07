# NumPy 2.0+ Complete API Inventory

This document provides a comprehensive inventory of all NumPy 2.0+ Python API functions, organized by module and category. This inventory excludes deprecated/legacy functions and the C API.

**Total Estimated Functions**: ~600-800+

---

## 1. Core Array Object (ndarray)

### 1.1 NDArray Attributes

#### Memory Layout
- [ ] `ndarray.flags`: Information about memory layout
- [ ] `ndarray.shape`: Tuple of array dimensions
- [ ] `ndarray.strides`: Tuple of bytes to step in each dimension
- [ ] `ndarray.ndim`: Number of array dimensions
- [ ] `ndarray.data`: Buffer object pointing to array data
- [ ] `ndarray.size`: Number of elements in array
- [ ] `ndarray.itemsize`: Length of one array element in bytes
- [ ] `ndarray.nbytes`: Total bytes consumed by array elements
- [ ] `ndarray.base`: Base object if memory is from another object

#### Data Type
- [ ] `ndarray.dtype`: Data-type of array elements

#### Other Attributes
- [ ] `ndarray.T`: Transposed array view
- [ ] `ndarray.real`: Real part of the array
- [ ] `ndarray.imag`: Imaginary part of the array
- [ ] `ndarray.flat`: 1-D iterator over the array

### 1.2 NDArray Methods

#### Array Conversion
- [ ] `ndarray.item(*args)`: Copy element to standard Python scalar
- [ ] `ndarray.tolist()`: Return array as nested Python lists
- [ ] `ndarray.itemset(*args)`: Insert scalar into array
- [ ] `ndarray.tostring([order])`: Raw copy of array data as Python string (deprecated, use tobytes)
- [ ] `ndarray.tobytes([order])`: Construct Python bytes containing raw data bytes
- [ ] `ndarray.tofile(fid[, sep, format])`: Write array to file
- [ ] `ndarray.dump(file)`: Dump pickle to file
- [ ] `ndarray.dumps()`: Return pickle as string
- [ ] `ndarray.astype(dtype[, order, casting, ...])`: Copy of array cast to specified type
- [ ] `ndarray.byteswap([inplace])`: Swap bytes of array elements
- [ ] `ndarray.copy([order])`: Return copy of the array
- [ ] `ndarray.view([dtype][, type])`: New view of array with same data
- [ ] `ndarray.getfield(dtype[, offset])`: Return field of given array as certain type
- [ ] `ndarray.setflags([write, align, uic])`: Set array flags WRITEABLE, ALIGNED, WRITEBACKIFCOPY

#### Shape Manipulation
- [ ] `ndarray.reshape(shape[, order])`: Return array with new shape
- [ ] `ndarray.resize(new_shape[, refcheck])`: Change shape and size of array in-place
- [ ] `ndarray.transpose(*axes)`: Return view with axes transposed
- [ ] `ndarray.swapaxes(axis1, axis2)`: Return view with axis1 and axis2 interchanged
- [ ] `ndarray.flatten([order])`: Return copy of array collapsed into one dimension
- [ ] `ndarray.ravel([order])`: Return flattened array
- [ ] `ndarray.squeeze([axis])`: Remove axes of length one

#### Item Selection and Manipulation
- [ ] `ndarray.take(indices[, axis, out, mode])`: Return array formed from elements at given indices
- [ ] `ndarray.put(indices, values[, mode])`: Set array elements using flat indexing
- [ ] `ndarray.repeat(repeats[, axis])`: Repeat elements of array
- [ ] `ndarray.choose(choices[, out, mode])`: Use index array to construct new array from choice arrays
- [ ] `ndarray.sort([axis, kind, order])`: Sort array in-place
- [ ] `ndarray.argsort([axis, kind, order])`: Return indices that would sort array
- [ ] `ndarray.partition(kth[, axis, kind, order])`: Partial in-place sort
- [ ] `ndarray.argpartition(kth[, axis, kind, ...])`: Return indices that would partition array
- [ ] `ndarray.searchsorted(v[, side, sorter])`: Find indices to insert elements
- [ ] `ndarray.nonzero()`: Return indices of non-zero elements
- [ ] `ndarray.compress(condition[, axis, out])`: Return selected slices along axis
- [ ] `ndarray.diagonal([offset, axis1, axis2])`: Return specified diagonals

#### Calculation Methods
- [ ] `ndarray.max([axis, out, keepdims])`: Return maximum along given axis
- [ ] `ndarray.argmax([axis, out, keepdims])`: Return indices of maximum values
- [ ] `ndarray.min([axis, out, keepdims])`: Return minimum along given axis
- [ ] `ndarray.argmin([axis, out, keepdims])`: Return indices of minimum values
- [ ] `ndarray.ptp([axis, out, keepdims])`: Peak-to-peak (max: min) value
- [ ] `ndarray.clip([min, max, out])`: Return array with values limited to [min, max]
- [ ] `ndarray.conj()`: Complex conjugate
- [ ] `ndarray.conjugate()`: Complex conjugate
- [ ] `ndarray.round([decimals, out])`: Return array rounded to given number of decimals
- [ ] `ndarray.trace([offset, axis1, axis2, ...])`: Return sum along diagonals
- [ ] `ndarray.sum([axis, dtype, out, keepdims])`: Return sum over given axis
- [ ] `ndarray.cumsum([axis, dtype, out])`: Return cumulative sum
- [ ] `ndarray.mean([axis, dtype, out, keepdims])`: Return mean over given axis
- [ ] `ndarray.var([axis, dtype, out, ddof, ...])`: Return variance over given axis
- [ ] `ndarray.std([axis, dtype, out, ddof, ...])`: Return standard deviation
- [ ] `ndarray.prod([axis, dtype, out, keepdims])`: Return product over given axis
- [ ] `ndarray.cumprod([axis, dtype, out])`: Return cumulative product
- [ ] `ndarray.all([axis, out, keepdims])`: Return True if all elements evaluate to True
- [ ] `ndarray.any([axis, out, keepdims])`: Return True if any element evaluates to True

#### Arithmetic Operations (implement as methods and ufuncs)
- [ ] `ndarray.__add__(value)`: Addition
- [ ] `ndarray.__sub__(value)`: Subtraction
- [ ] `ndarray.__mul__(value)`: Multiplication
- [ ] `ndarray.__truediv__(value)`: True division
- [ ] `ndarray.__floordiv__(value)`: Floor division
- [ ] `ndarray.__mod__(value)`: Modulo
- [ ] `ndarray.__pow__(value[, modulo])`: Power
- [ ] `ndarray.__matmul__(value)`: Matrix multiplication
- [ ] `ndarray.__neg__()`: Unary negation
- [ ] `ndarray.__pos__()`: Unary positive
- [ ] `ndarray.__abs__()`: Absolute value
- [ ] `ndarray.__invert__()`: Bitwise inversion

#### Comparison Operations
- [ ] `ndarray.__lt__(value)`: Less than
- [ ] `ndarray.__le__(value)`: Less than or equal
- [ ] `ndarray.__gt__(value)`: Greater than
- [ ] `ndarray.__ge__(value)`: Greater than or equal
- [ ] `ndarray.__eq__(value)`: Equal
- [ ] `ndarray.__ne__(value)`: Not equal

#### Other Special Methods
- [ ] `ndarray.__len__()`: Length of first axis
- [ ] `ndarray.__getitem__(key)`: Get item
- [ ] `ndarray.__setitem__(key, value)`: Set item
- [ ] `ndarray.__contains__(key)`: Membership test
- [ ] `ndarray.__iter__()`: Iterator over first axis
- [ ] `ndarray.__str__()`: String representation
- [ ] `ndarray.__repr__()`: Official string representation

---

## 2. Array Creation Routines

### 2.1 From Shape or Value
- [ ] `empty(shape[, dtype, order, device])`: Return new array without initializing entries
- [ ] `empty_like(prototype[, dtype, order, ...])`: Return new array with same shape and type
- [ ] `eye(N[, M, k, dtype, order, device])`: Return 2-D array with ones on diagonal
- [ ] `identity(n[, dtype, device])`: Return identity array
- [ ] `ones(shape[, dtype, order, device])`: Return new array filled with ones
- [ ] `ones_like(a[, dtype, order, subok, ...])`: Return array of ones with same shape
- [ ] `zeros(shape[, dtype, order, device])`: Return new array filled with zeros
- [ ] `zeros_like(a[, dtype, order, subok, ...])`: Return array of zeros with same shape
- [ ] `full(shape, fill_value[, dtype, order, ...])`: Return new array filled with fill_value
- [ ] `full_like(a, fill_value[, dtype, order, ...])`: Return full array with same shape

### 2.2 From Existing Data
- [ ] `array(object[, dtype, copy, order, ...])`: Create array
- [ ] `asarray(a[, dtype, order, device, copy])`: Convert input to array
- [ ] `asanyarray(a[, dtype, order, device])`: Convert to ndarray, but pass subclasses through
- [ ] `ascontiguousarray(a[, dtype, device])`: Return contiguous array (C order)
- [ ] `asmatrix(data[, dtype])`: Interpret input as matrix
- [ ] `copy(a[, order])`: Return copy of array
- [ ] `frombuffer(buffer[, dtype, count, offset])`: Interpret buffer as 1-D array
- [ ] `from_dlpack(x)`: Create array from object supporting DLPack protocol
- [ ] `fromfile(file[, dtype, count, sep, offset])`: Construct array from file
- [ ] `fromfunction(function, shape[, dtype])`: Construct by executing function over each coordinate
- [ ] `fromiter(iter, dtype[, count])`: Create array from iterable
- [ ] `fromstring(string[, dtype, count, sep])`: New 1-D array initialized from text data
- [ ] `loadtxt(fname[, dtype, comments, ...])`: Load data from text file

### 2.3 Creating Record Arrays
- [ ] `rec.array(obj[, dtype, shape, offset, ...])`: Construct record array
- [ ] `rec.fromarrays(arrayList[, dtype, shape, ...])`: Create record array from arrays
- [ ] `rec.fromrecords(recList[, dtype, shape, ...])`: Create recarray from list of records
- [ ] `rec.fromstring(datastring[, dtype, ...])`: Create record array from binary data
- [ ] `rec.fromfile(fd[, dtype, shape, offset, ...])`: Create array from binary file data

### 2.4 Creating Character Arrays
- [ ] `char.array(obj[, itemsize, copy, ...])`: Create character array
- [ ] `char.asarray(obj[, itemsize])`: Convert input to character array

### 2.5 Numerical Ranges
- [ ] `arange([start,] stop[, step,][, dtype, ...])`: Return evenly spaced values within interval
- [ ] `linspace(start, stop[, num, endpoint, ...])`: Return evenly spaced numbers over interval
- [ ] `logspace(start, stop[, num, endpoint, ...])`: Return numbers spaced evenly on log scale
- [ ] `geomspace(start, stop[, num, endpoint, ...])`: Return numbers spaced evenly on log scale (geometric)
- [ ] `meshgrid(*xi[, copy, sparse, indexing])`: Return coordinate matrices from coordinate vectors
- [ ] `mgrid`: Instance returning dense multi-dimensional meshgrid
- [ ] `ogrid`: Instance returning open multi-dimensional meshgrid

### 2.6 Building Matrices
- [ ] `diag(v[, k])`: Extract diagonal or construct diagonal array
- [ ] `diagflat(v[, k])`: Create 2-D array with flattened input as diagonal
- [ ] `tri(N[, M, k, dtype])`: Array with ones at and below diagonal
- [ ] `tril(m[, k])`: Lower triangle of array
- [ ] `triu(m[, k])`: Upper triangle of array
- [ ] `vander(x[, N, increasing])`: Generate Vandermonde matrix

### 2.7 Matrix Class
- [ ] `bmat(obj[, ldict, gdict])`: Build matrix object from string, nested sequence, or array

---

## 3. Array Manipulation Routines

### 3.1 Basic Operations
- [ ] `copyto(dst, src[, casting, where])`: Copy values from one array to another
- [ ] `ndim(a)`: Return number of dimensions
- [ ] `shape(a)`: Return shape of array
- [ ] `size(a[, axis])`: Return number of elements

### 3.2 Changing Array Shape
- [ ] `reshape(a, newshape[, order])`: Give new shape to array
- [ ] `ravel(a[, order])`: Return contiguous flattened array
- [ ] `ndarray.flat`: 1-D iterator over array
- [ ] `ndarray.flatten([order])`: Return copy collapsed into one dimension

### 3.3 Transpose-like Operations
- [ ] `moveaxis(a, source, destination)`: Move axes to new positions
- [ ] `rollaxis(a, axis[, start])`: Roll specified axis backwards
- [ ] `swapaxes(a, axis1, axis2)`: Interchange two axes
- [ ] `ndarray.T`: Transposed array
- [ ] `transpose(a[, axes])`: Reverse or permute axes
- [ ] `permute_dims(a, axes)`: Permute dimensions of array
- [ ] `matrix_transpose(x)`: Transpose of matrix (or stack of matrices)

### 3.4 Changing Number of Dimensions
- [ ] `atleast_1d(*arys)`: Convert inputs to arrays with at least 1 dimension
- [ ] `atleast_2d(*arys)`: View inputs as arrays with at least 2 dimensions
- [ ] `atleast_3d(*arys)`: View inputs as arrays with at least 3 dimensions
- [ ] `broadcast`: Produce object that mimics broadcasting
- [ ] `broadcast_to(array, shape[, subok])`: Broadcast array to new shape
- [ ] `broadcast_arrays(*args[, subok])`: Broadcast arrays against each other
- [ ] `expand_dims(a, axis)`: Expand shape by inserting new axis
- [ ] `squeeze(a[, axis])`: Remove axes of length one

### 3.5 Changing Kind of Array
- [ ] `asarray(a[, dtype, order])`: Convert to array
- [ ] `asanyarray(a[, dtype, order])`: Convert to ndarray, pass subclasses
- [ ] `asmatrix(data[, dtype])`: Interpret as matrix
- [ ] `asfortranarray(a[, dtype])`: Return array laid out in Fortran order
- [ ] `ascontiguousarray(a[, dtype])`: Return contiguous array in C order
- [ ] `asarray_chkfinite(a[, dtype, order])`: Convert to array, check for NaNs/Infs
- [ ] `require(a[, dtype, requirements])`: Return array with requirements

### 3.6 Joining Arrays
- [ ] `concatenate(arrays[, axis, out, dtype, ...])`: Join sequence of arrays along existing axis
- [ ] `concat(arrays[, axis, out, dtype, casting])`: Join sequence of arrays
- [ ] `stack(arrays[, axis, out, dtype, casting])`: Join sequence along new axis
- [ ] `block(arrays)`: Assemble nd-array from nested lists of blocks
- [ ] `vstack(tup[, dtype, casting])`: Stack arrays vertically (row-wise)
- [ ] `hstack(tup[, dtype, casting])`: Stack arrays horizontally (column-wise)
- [ ] `dstack(tup)`: Stack arrays depth-wise (along third axis)
- [ ] `column_stack(tup)`: Stack 1-D arrays as columns into 2-D array

### 3.7 Splitting Arrays
- [ ] `split(ary, indices_or_sections[, axis])`: Split array into multiple sub-arrays
- [ ] `array_split(ary, indices_or_sections[, axis])`: Split into multiple sub-arrays (unequal)
- [ ] `dsplit(ary, indices_or_sections)`: Split along 3rd axis
- [ ] `hsplit(ary, indices_or_sections)`: Split horizontally
- [ ] `vsplit(ary, indices_or_sections)`: Split vertically

### 3.8 Tiling Arrays
- [ ] `tile(A, reps)`: Construct array by repeating
- [ ] `repeat(a, repeats[, axis])`: Repeat elements

### 3.9 Adding and Removing Elements
- [ ] `delete(arr, obj[, axis])`: Return new array with sub-arrays deleted
- [ ] `insert(arr, obj, values[, axis])`: Insert values along axis
- [ ] `append(arr, values[, axis])`: Append values to end of array
- [ ] `resize(a, new_shape)`: Return new array with specified shape
- [ ] `trim_zeros(filt[, trim])`: Trim leading/trailing zeros from 1-D array
- [ ] `unique(ar[, return_index, return_inverse, ...])`: Find unique elements
- [ ] `pad(array, pad_width[, mode])`: Pad array

### 3.10 Rearranging Elements
- [ ] `flip(m[, axis])`: Reverse order of elements along axis
- [ ] `fliplr(m)`: Reverse order of elements along axis 1 (left/right)
- [ ] `flipud(m)`: Reverse order of elements along axis 0 (up/down)
- [ ] `reshape(a, newshape[, order])`: Give new shape
- [ ] `roll(a, shift[, axis])`: Roll array elements along axis
- [ ] `rot90(m[, k, axes])`: Rotate array by 90 degrees

---

## 4. Binary Operations

### 4.1 Elementwise Bit Operations
- [ ] `bitwise_and(x1, x2[, out, where])`: Compute bitwise AND
- [ ] `bitwise_or(x1, x2[, out, where])`: Compute bitwise OR
- [ ] `bitwise_xor(x1, x2[, out, where])`: Compute bitwise XOR
- [ ] `bitwise_not(x[, out, where])`: Compute bitwise NOT
- [ ] `invert(x[, out, where])`: Compute bitwise inversion

### 4.2 Bit Packing
- [ ] `packbits(a[, axis, bitorder])`: Pack elements into bits
- [ ] `unpackbits(a[, axis, count, bitorder])`: Unpack bits into uint8 array

### 4.3 Output Formatting
- [ ] `binary_repr(num[, width])`: Return binary representation as string

---

## 5. String Operations

### 5.1 String Operations
- [ ] `add(x1, x2)`: Concatenate strings
- [ ] `multiply(a, i)`: Return string with multiple concatenation
- [ ] `mod(a, values)`: Return string formatted with values
- [ ] `capitalize(a)`: Capitalize first character
- [ ] `center(a, width[, fillchar])`: Center string
- [ ] `decode(a[, encoding, errors])`: Decode using codec
- [ ] `encode(a[, encoding, errors])`: Encode using codec
- [ ] `expandtabs(a[, tabsize])`: Replace tabs with spaces
- [ ] `join(sep, seq)`: Join sequence of strings
- [ ] `ljust(a, width[, fillchar])`: Left-justify string
- [ ] `lower(a)`: Convert to lowercase
- [ ] `lstrip(a[, chars])`: Remove leading characters
- [ ] `partition(a, sep)`: Partition around separator
- [ ] `replace(a, old, new[, count])`: Replace occurrences
- [ ] `rjust(a, width[, fillchar])`: Right-justify
- [ ] `rpartition(a, sep)`: Partition from right
- [ ] `rsplit(a[, sep, maxsplit])`: Split from right
- [ ] `rstrip(a[, chars])`: Remove trailing characters
- [ ] `split(a[, sep, maxsplit])`: Split string
- [ ] `splitlines(a[, keepends])`: Split at line breaks
- [ ] `strip(a[, chars])`: Remove leading and trailing characters
- [ ] `swapcase(a)`: Swap case
- [ ] `title(a)`: Convert to titlecase
- [ ] `translate(a, table[, deletechars])`: Translate characters
- [ ] `upper(a)`: Convert to uppercase
- [ ] `zfill(a, width)`: Left-fill with zeros

### 5.2 Comparison
- [ ] `equal(x1, x2)`: Element-wise equality
- [ ] `not_equal(x1, x2)`: Element-wise inequality
- [ ] `greater(x1, x2)`: Element-wise greater than
- [ ] `greater_equal(x1, x2)`: Element-wise greater or equal
- [ ] `less(x1, x2)`: Element-wise less than
- [ ] `less_equal(x1, x2)`: Element-wise less or equal
- [ ] `compare_chararrays(a, b, cmp_op, rstrip)`: Perform comparison

### 5.3 String Information
- [ ] `count(a, sub[, start, end])`: Count non-overlapping occurrences
- [ ] `endswith(a, suffix[, start, end])`: Check if ends with suffix
- [ ] `find(a, sub[, start, end])`: Find first occurrence
- [ ] `index(a, sub[, start, end])`: Like find, raises ValueError
- [ ] `isalpha(a)`: Check if all characters alphabetic
- [ ] `isalnum(a)`: Check if all characters alphanumeric
- [ ] `isdecimal(a)`: Check if all characters decimal
- [ ] `isdigit(a)`: Check if all characters digits
- [ ] `islower(a)`: Check if all cased characters lowercase
- [ ] `isnumeric(a)`: Check if all characters numeric
- [ ] `isspace(a)`: Check if all characters whitespace
- [ ] `istitle(a)`: Check if titlecased
- [ ] `isupper(a)`: Check if all cased characters uppercase
- [ ] `rfind(a, sub[, start, end])`: Find from right
- [ ] `rindex(a, sub[, start, end])`: Like rfind, raises ValueError
- [ ] `startswith(a, prefix[, start, end])`: Check if starts with prefix
- [ ] `str_len(a)`: Return length of each element

---

## 6. Datetime Support Functions

### 6.1 Business Day Functions
- [ ] `busday_count(begindates, enddates[, ...])`: Count valid business days
- [ ] `busday_offset(dates, offsets[, roll, ...])`: Apply offsets with business day calendar
- [ ] `is_busday(dates[, weekmask, holidays, ...])`: Check if dates are business days

### 6.2 Datetime Functions
- [ ] `datetime_as_string(arr[, unit, timezone, ...])`: Convert to string array
- [ ] `datetime_data(dtype)`: Get datetime metadata

---

## 7. Data Type Routines

### 7.1 Data Type Information
- [ ] `can_cast(from_, to[, casting])`: Check if cast is possible
- [ ] `promote_types(type1, type2)`: Find smallest type to hold both
- [ ] `min_scalar_type(a)`: Return type of smallest size/kind
- [ ] `result_type(*arrays_and_dtypes)`: Return result type
- [ ] `common_type(*arrays)`: Return common scalar type
- [ ] `obj2sctype(rep[, default])`: Return scalar dtype from object

### 7.2 Creating Data Types
- [ ] `dtype(dtype[, align, copy])`: Create data type object
- [ ] `format_parser(formats[, names, titles, ...])`: Parse format strings

### 7.3 Data Type Information Classes
- [ ] `finfo(dtype)`: Machine limits for floating point types
- [ ] `iinfo(dtype)`: Machine limits for integer types
- [ ] `MachAr`: Machine limits

### 7.4 Data Type Testing
- [ ] `isscalar(element)`: Check if element is scalar
- [ ] `issubdtype(arg1, arg2)`: Check if first is subtype of second
- [ ] `issubsctype(arg1, arg2)`: Check based on scalar type
- [ ] `issubclass_(arg1, arg2)`: Determine if class is subclass
- [ ] `find_common_type(array_types, scalar_types)`: Determine common type

---

## 8. Mathematical Functions

### 8.1 Trigonometric Functions
- [ ] `sin(x[, out, where])`: Trigonometric sine
- [ ] `cos(x[, out, where])`: Cosine
- [ ] `tan(x[, out, where])`: Tangent
- [ ] `arcsin(x[, out, where])`: Inverse sine
- [ ] `arccos(x[, out, where])`: Inverse cosine
- [ ] `arctan(x[, out, where])`: Inverse tangent
- [ ] `arctan2(x1, x2[, out, where])`: Element-wise arc tangent
- [ ] `hypot(x1, x2[, out, where])`: Hypotenuse
- [ ] `degrees(x[, out, where])`: Convert radians to degrees
- [ ] `radians(x[, out, where])`: Convert degrees to radians
- [ ] `unwrap(p[, discont, axis, period])`: Unwrap by changing deltas to period complement
- [ ] `deg2rad(x[, out, where])`: Convert degrees to radians
- [ ] `rad2deg(x[, out, where])`: Convert radians to degrees

### 8.2 Hyperbolic Functions
- [ ] `sinh(x[, out, where])`: Hyperbolic sine
- [ ] `cosh(x[, out, where])`: Hyperbolic cosine
- [ ] `tanh(x[, out, where])`: Hyperbolic tangent
- [ ] `arcsinh(x[, out, where])`: Inverse hyperbolic sine
- [ ] `arccosh(x[, out, where])`: Inverse hyperbolic cosine
- [ ] `arctanh(x[, out, where])`: Inverse hyperbolic tangent

### 8.3 Rounding
- [ ] `around(a[, decimals, out])`: Round to given number of decimals
- [ ] `round(a[, decimals, out])`: Round (alias for around)
- [ ] `rint(x[, out, where])`: Round to nearest integer
- [ ] `fix(x[, out])`: Round to nearest integer toward zero
- [ ] `floor(x[, out, where])`: Floor
- [ ] `ceil(x[, out, where])`: Ceiling
- [ ] `trunc(x[, out, where])`: Truncate

### 8.4 Sums, Products, Differences
- [ ] `prod(a[, axis, dtype, out, keepdims])`: Product of array elements
- [ ] `sum(a[, axis, dtype, out, keepdims])`: Sum of array elements
- [ ] `nanprod(a[, axis, dtype, out, keepdims])`: Product ignoring NaNs
- [ ] `nansum(a[, axis, dtype, out, keepdims])`: Sum ignoring NaNs
- [ ] `cumprod(a[, axis, dtype, out])`: Cumulative product
- [ ] `cumsum(a[, axis, dtype, out])`: Cumulative sum
- [ ] `nancumprod(a[, axis, dtype, out])`: Cumulative product ignoring NaNs
- [ ] `nancumsum(a[, axis, dtype, out])`: Cumulative sum ignoring NaNs
- [ ] `diff(a[, n, axis, prepend, append])`: Discrete difference
- [ ] `ediff1d(ary[, to_end, to_begin])`: Differences between consecutive elements
- [ ] `gradient(f, *varargs[, axis, edge_order])`: Gradient
- [ ] `cross(a, b[, axisa, axisb, axisc, axis])`: Cross product
- [ ] `trapz(y[, x, dx, axis])`: Integrate using trapezoidal rule (deprecated, use trapezoid)
- [ ] `trapezoid(y[, x, dx, axis])`: Integrate using trapezoidal rule

### 8.5 Exponents and Logarithms
- [ ] `exp(x[, out, where])`: Exponential
- [ ] `expm1(x[, out, where])`: exp(x): 1
- [ ] `exp2(x[, out, where])`: 2**x
- [ ] `log(x[, out, where])`: Natural logarithm
- [ ] `log10(x[, out, where])`: Base-10 logarithm
- [ ] `log2(x[, out, where])`: Base-2 logarithm
- [ ] `log1p(x[, out, where])`: log(1 + x)
- [ ] `logaddexp(x1, x2[, out, where])`: log(exp(x1) + exp(x2))
- [ ] `logaddexp2(x1, x2[, out, where])`: log2(2**x1 + 2**x2)

### 8.6 Other Special Functions
- [ ] `i0(x)`: Modified Bessel function of first kind, order 0
- [ ] `sinc(x)`: Sinc function

### 8.7 Floating Point Routines
- [ ] `signbit(x[, out, where])`: Check if sign bit is set
- [ ] `copysign(x1, x2[, out, where])`: Copy sign
- [ ] `frexp(x[, out1, out2, where])`: Decompose into mantissa and exponent
- [ ] `ldexp(x1, x2[, out, where])`: x1 * 2**x2
- [ ] `nextafter(x1, x2[, out, where])`: Next floating point value toward x2
- [ ] `spacing(x[, out, where])`: Distance to nearest adjacent number

### 8.8 Rational Routines
- [ ] `lcm(x1, x2[, out, where])`: Least common multiple
- [ ] `gcd(x1, x2[, out, where])`: Greatest common divisor

### 8.9 Arithmetic Operations
- [ ] `add(x1, x2[, out, where])`: Addition
- [ ] `reciprocal(x[, out, where])`: Reciprocal
- [ ] `positive(x[, out, where])`: Unary positive
- [ ] `negative(x[, out, where])`: Unary negative
- [ ] `multiply(x1, x2[, out, where])`: Multiplication
- [ ] `divide(x1, x2[, out, where])`: True division
- [ ] `power(x1, x2[, out, where])`: Exponentiation
- [ ] `subtract(x1, x2[, out, where])`: Subtraction
- [ ] `true_divide(x1, x2[, out, where])`: True division
- [ ] `floor_divide(x1, x2[, out, where])`: Floor division
- [ ] `float_power(x1, x2[, out, where])`: Float exponentiation
- [ ] `fmod(x1, x2[, out, where])`: Modulo
- [ ] `mod(x1, x2[, out, where])`: Remainder
- [ ] `modf(x[, out1, out2, where])`: Fractional and integer parts
- [ ] `remainder(x1, x2[, out, where])`: Remainder
- [ ] `divmod(x1, x2[, out1, out2, where])`: Quotient and remainder

### 8.10 Handling Complex Numbers
- [ ] `angle(z[, deg])`: Return angle of complex argument
- [ ] `real(val)`: Real part
- [ ] `imag(val)`: Imaginary part
- [ ] `conj(x[, out, where])`: Complex conjugate
- [ ] `conjugate(x[, out, where])`: Complex conjugate

### 8.11 Extrema Finding
- [ ] `maximum(x1, x2[, out, where])`: Element-wise maximum
- [ ] `max(a[, axis, out, keepdims])`: Maximum
- [ ] `amax(a[, axis, out, keepdims])`: Maximum (same as max)
- [ ] `fmax(x1, x2[, out, where])`: Element-wise maximum (ignoring NaNs)
- [ ] `nanmax(a[, axis, out, keepdims])`: Maximum ignoring NaNs
- [ ] `minimum(x1, x2[, out, where])`: Element-wise minimum
- [ ] `min(a[, axis, out, keepdims])`: Minimum
- [ ] `amin(a[, axis, out, keepdims])`: Minimum (same as min)
- [ ] `fmin(x1, x2[, out, where])`: Element-wise minimum (ignoring NaNs)
- [ ] `nanmin(a[, axis, out, keepdims])`: Minimum ignoring NaNs

### 8.12 Miscellaneous
- [ ] `convolve(a, v[, mode])`: Convolution
- [ ] `clip(a, a_min, a_max[, out])`: Clip values
- [ ] `sqrt(x[, out, where])`: Square root
- [ ] `cbrt(x[, out, where])`: Cube root
- [ ] `square(x[, out, where])`: Square
- [ ] `absolute(x[, out, where])`: Absolute value
- [ ] `abs(x[, out, where])`: Absolute value (alias)
- [ ] `fabs(x[, out, where])`: Absolute value (floating)
- [ ] `sign(x[, out, where])`: Sign
- [ ] `heaviside(x1, x2[, out, where])`: Heaviside step function
- [ ] `nan_to_num(x[, copy, nan, posinf, neginf])`: Replace NaN with zero and inf with large numbers
- [ ] `real_if_close(a[, tol])`: Return real array if imaginary part is close to zero
- [ ] `interp(x, xp, fp[, left, right, period])`: Linear interpolation

---

## 9. Linear Algebra (numpy.linalg)

### 9.1 Matrix and Vector Products
- [ ] `dot(a, b[, out])`: Dot product
- [ ] `linalg.multi_dot(arrays[, out])`: Chained dot product
- [ ] `vdot(a, b)`: Dot product treating vectors as 1-D
- [ ] `vecdot(x1, x2[, axis])`: Vector dot product
- [ ] `linalg.vecdot(x1, x2[, axis])`: Vector dot product
- [ ] `inner(a, b)`: Inner product
- [ ] `outer(a, b[, out])`: Outer product
- [ ] `matmul(x1, x2[, out])`: Matrix product
- [ ] `linalg.matmul(x1, x2[, out])`: Matrix product
- [ ] `tensordot(a, b[, axes])`: Tensor dot product
- [ ] `linalg.tensordot(a, b[, axes])`: Tensor dot product
- [ ] `einsum(subscripts, *operands[, out, dtype, ...])`: Einstein summation
- [ ] `einsum_path(subscripts, *operands[, optimize])`: Evaluate einsum strategy
- [ ] `linalg.matrix_power(a, n)`: Matrix power
- [ ] `kron(a, b)`: Kronecker product
- [ ] `linalg.cross(a, b[, axis])`: Cross product

### 9.2 Decompositions
- [ ] `linalg.cholesky(a)`: Cholesky decomposition
- [ ] `linalg.qr(a[, mode])`: QR decomposition
- [ ] `linalg.svd(a[, full_matrices, compute_uv, ...])`: Singular value decomposition
- [ ] `linalg.svdvals(x)`: Singular values only

### 9.3 Matrix Eigenvalues
- [ ] `linalg.eig(a)`: Eigenvalues and right eigenvectors
- [ ] `linalg.eigh(a[, UPLO])`: Eigenvalues and eigenvectors (Hermitian/symmetric)
- [ ] `linalg.eigvals(a)`: Eigenvalues only
- [ ] `linalg.eigvalsh(a[, UPLO])`: Eigenvalues only (Hermitian/symmetric)

### 9.4 Norms and Other Numbers
- [ ] `linalg.norm(x[, ord, axis, keepdims])`: Matrix or vector norm
- [ ] `linalg.matrix_norm(x[, ord, axis, keepdims])`: Matrix norm
- [ ] `linalg.vector_norm(x[, ord, axis, keepdims])`: Vector norm
- [ ] `linalg.cond(x[, p])`: Condition number
- [ ] `linalg.det(a)`: Determinant
- [ ] `linalg.matrix_rank(A[, tol, hermitian])`: Matrix rank
- [ ] `linalg.slogdet(a)`: Sign and log of determinant
- [ ] `trace(a[, offset, axis1, axis2, dtype, out])`: Trace
- [ ] `linalg.trace(x[, offset, axis1, axis2, dtype])`: Trace

### 9.5 Solving Equations and Inverting Matrices
- [ ] `linalg.solve(a, b)`: Solve linear system
- [ ] `linalg.tensorsolve(a, b[, axes])`: Solve tensor equation
- [ ] `linalg.lstsq(a, b[, rcond])`: Least-squares solution
- [ ] `linalg.inv(a)`: Matrix inverse
- [ ] `linalg.pinv(a[, rcond, hermitian])`: Moore-Penrose pseudoinverse
- [ ] `linalg.tensorinv(a[, ind])`: Inverse of tensor

### 9.6 Other Matrix Operations
- [ ] `diagonal(a[, offset, axis1, axis2])`: Diagonal
- [ ] `linalg.diagonal(x[, offset, axis1, axis2])`: Diagonal
- [ ] `linalg.matrix_transpose(x)`: Matrix transpose

### 9.7 Exceptions
- [ ] `linalg.LinAlgError`: Generic linear algebra error

---

## 10. Logic Functions

### 10.1 Truth Value Testing
- [ ] `all(a[, axis, out, keepdims, where])`: Test if all elements evaluate to True
- [ ] `any(a[, axis, out, keepdims, where])`: Test if any element evaluates to True

### 10.2 Array Contents
- [ ] `isfinite(x[, out, where])`: Test for finiteness
- [ ] `isinf(x[, out, where])`: Test for infinity
- [ ] `isnan(x[, out, where])`: Test for NaN
- [ ] `isnat(x[, out, where])`: Test for NaT (not-a-time)
- [ ] `isneginf(x[, out])`: Test for negative infinity
- [ ] `isposinf(x[, out])`: Test for positive infinity

### 10.3 Array Type Testing
- [ ] `iscomplex(x)`: Test for complex numbers
- [ ] `iscomplexobj(x)`: Test if input is complex type or array of complex
- [ ] `isfortran(a)`: Test if array is Fortran contiguous
- [ ] `isreal(x)`: Test for real numbers
- [ ] `isrealobj(x)`: Test if input is real type
- [ ] `isscalar(element)`: Test if element is scalar

### 10.4 Logical Operations
- [ ] `logical_and(x1, x2[, out, where])`: Logical AND
- [ ] `logical_or(x1, x2[, out, where])`: Logical OR
- [ ] `logical_not(x[, out, where])`: Logical NOT
- [ ] `logical_xor(x1, x2[, out, where])`: Logical XOR

### 10.5 Comparison
- [ ] `allclose(a, b[, rtol, atol, equal_nan])`: Test if two arrays are element-wise equal within tolerance
- [ ] `isclose(a, b[, rtol, atol, equal_nan])`: Test element-wise for equality within tolerance
- [ ] `array_equal(a1, a2[, equal_nan])`: Test if arrays have same shape and elements
- [ ] `array_equiv(a1, a2)`: Test if arrays are broadcastable and equal
- [ ] `greater(x1, x2[, out, where])`: Greater than
- [ ] `greater_equal(x1, x2[, out, where])`: Greater than or equal
- [ ] `less(x1, x2[, out, where])`: Less than
- [ ] `less_equal(x1, x2[, out, where])`: Less than or equal
- [ ] `equal(x1, x2[, out, where])`: Equal
- [ ] `not_equal(x1, x2[, out, where])`: Not equal

---

## 11. Masked Array Operations

### 11.1 Creation
- [ ] `ma.masked_array`: Alias for MaskedArray
- [ ] `ma.array(data[, dtype, copy, order, mask, ...])`: Create masked array
- [ ] `ma.copy(a[, order])`: Copy masked array
- [ ] `ma.frombuffer(buffer[, dtype, count, ...])`: Create from buffer
- [ ] `ma.fromfunction(function, shape, **kwargs)`: Create from function

### 11.2 Inspecting the Array
- [ ] `ma.all(a[, axis, out, keepdims])`: Check if all elements are True
- [ ] `ma.any(a[, axis, out, keepdims])`: Check if any element is True
- [ ] `ma.count(a[, axis, keepdims])`: Count non-masked elements
- [ ] `ma.count_masked(arr[, axis])`: Count masked elements
- [ ] `ma.getmask(a)`: Get mask
- [ ] `ma.getmaskarray(arr)`: Get mask or full array of False
- [ ] `ma.getdata(a[, subok])`: Get data
- [ ] `ma.nonzero(a)`: Return indices of non-zero unmasked elements
- [ ] `ma.shape(obj)`: Return shape
- [ ] `ma.size(obj[, axis])`: Return size

### 11.3 Manipulating Masked Arrays
- [ ] `ma.reshape(a, new_shape[, order])`: Reshape
- [ ] `ma.ravel(a[, order])`: Flatten
- [ ] `ma.concatenate(arrays[, axis])`: Concatenate
- [ ] `ma.stack(arrays[, axis, out])`: Stack along new axis
- [ ] `ma.vstack(tup)`: Stack vertically
- [ ] `ma.hstack(tup)`: Stack horizontally
- [ ] `ma.dstack(tup)`: Stack depth-wise

### 11.4 Operations on Masks
- [ ] `ma.make_mask(m[, copy, shrink, dtype])`: Create boolean mask
- [ ] `ma.make_mask_none(newshape[, dtype])`: Return mask of all False
- [ ] `ma.mask_or(m1, m2[, copy, shrink])`: Combine masks with OR
- [ ] `ma.mask_rowcols(a[, axis])`: Mask rows/columns with masked values

### 11.5 Conversion Operations
- [ ] `ma.filled(a[, fill_value])`: Return copy with masked values filled
- [ ] `ma.compressed(x)`: Return all non-masked data as 1-D array
- [ ] `ma.MaskedArray.tolist([fill_value])`: Convert to nested Python list
- [ ] `ma.MaskedArray.torecords()`: Convert to recarray

---

## 12. Mathematical Functions with Automatic Domain (numpy.emath)

- [ ] `emath.sqrt(x)`: Square root
- [ ] `emath.log(x)`: Natural logarithm
- [ ] `emath.log2(x)`: Base-2 logarithm
- [ ] `emath.log10(x)`: Base-10 logarithm
- [ ] `emath.logn(n, x)`: Base-n logarithm
- [ ] `emath.power(x, p)`: Power function
- [ ] `emath.arccos(x)`: Inverse cosine
- [ ] `emath.arcsin(x)`: Inverse sine
- [ ] `emath.arctanh(x)`: Inverse hyperbolic tangent

---

## 13. Floating Point Error Handling

- [ ] `seterr([all, divide, over, under, invalid])`: Set how floating-point errors are handled
- [ ] `geterr()`: Get current error handling
- [ ] `seterrcall(func)`: Set floating-point error callback
- [ ] `geterrcall()`: Get current error callback
- [ ] `errstate(**kwargs)`: Context manager for error handling

---

## 14. Discrete Fourier Transform (numpy.fft)

### 14.1 Standard FFTs
- [ ] `fft.fft(a[, n, axis, norm])`: 1-D discrete Fourier Transform
- [ ] `fft.ifft(a[, n, axis, norm])`: 1-D inverse discrete Fourier Transform
- [ ] `fft.fft2(a[, s, axes, norm])`: 2-D discrete Fourier Transform
- [ ] `fft.ifft2(a[, s, axes, norm])`: 2-D inverse discrete Fourier Transform
- [ ] `fft.fftn(a[, s, axes, norm])`: N-D discrete Fourier Transform
- [ ] `fft.ifftn(a[, s, axes, norm])`: N-D inverse discrete Fourier Transform

### 14.2 Real FFTs
- [ ] `fft.rfft(a[, n, axis, norm])`: 1-D FFT of real input
- [ ] `fft.irfft(a[, n, axis, norm])`: Inverse of rfft
- [ ] `fft.rfft2(a[, s, axes, norm])`: 2-D FFT of real input
- [ ] `fft.irfft2(a[, s, axes, norm])`: Inverse of rfft2
- [ ] `fft.rfftn(a[, s, axes, norm])`: N-D FFT of real input
- [ ] `fft.irfftn(a[, s, axes, norm])`: Inverse of rfftn

### 14.3 Hermitian FFTs
- [ ] `fft.hfft(a[, n, axis, norm])`: FFT of Hermitian symmetric sequence
- [ ] `fft.ihfft(a[, n, axis, norm])`: Inverse of hfft

### 14.4 Helper Routines
- [ ] `fft.fftfreq(n[, d])`: Return FFT sample frequencies
- [ ] `fft.rfftfreq(n[, d])`: Return rfft sample frequencies
- [ ] `fft.fftshift(x[, axes])`: Shift zero-frequency to center
- [ ] `fft.ifftshift(x[, axes])`: Inverse of fftshift

---

## 15. Functional Programming

- [ ] `apply_along_axis(func1d, axis, arr, *args, ...)`: Apply function along axis
- [ ] `apply_over_axes(func, a, axes)`: Apply function over multiple axes
- [ ] `vectorize(pyfunc[, otypes, doc, excluded, ...])`: Vectorize function
- [ ] `frompyfunc(func, nin, nout)`: Create ufunc from Python function
- [ ] `piecewise(x, condlist, funclist[, args, kw])`: Evaluate piecewise function

---

## 16. Input and Output

### 16.1 NumPy Binary Files (.npy, .npz)
- [ ] `load(file[, mmap_mode, allow_pickle, ...])`: Load arrays from .npy, .npz, or pickled files
- [ ] `save(file, arr[, allow_pickle, fix_imports])`: Save array to binary .npy file
- [ ] `savez(file, *args, **kwds)`: Save multiple arrays to uncompressed .npz file
- [ ] `savez_compressed(file, *args, **kwds)`: Save to compressed .npz file
- [ ] `lib.npyio.NpzFile(fid[, own_fid, ...])`: Dictionary-like object for .npz files

### 16.2 Text Files
- [ ] `loadtxt(fname[, dtype, comments, ...])`: Load data from text file
- [ ] `savetxt(fname, X[, fmt, delimiter, ...])`: Save array to text file
- [ ] `genfromtxt(fname[, dtype, comments, ...])`: Load data with missing value handling

### 16.3 Raw Binary Files
- [ ] `fromfile(file[, dtype, count, sep, offset])`: Construct array from file
- [ ] `ndarray.tofile(fid[, sep, format])`: Write to file

### 16.4 String Formatting
- [ ] `array2string(a[, max_line_width, ...])`: Return string representation
- [ ] `array_repr(arr[, max_line_width, precision, ...])`: Return string representation for __repr__
- [ ] `array_str(a[, max_line_width, precision, ...])`: Return string representation for __str__
- [ ] `format_float_positional(x[, precision, ...])`: Format float in positional notation
- [ ] `format_float_scientific(x[, precision, ...])`: Format float in scientific notation

### 16.5 Memory Mapping Files
- [ ] `memmap(filename[, dtype, mode, offset, ...])`: Create memory-map to array in file
- [ ] `lib.format.open_memmap(filename[, mode, ...])`: Open .npy file as memory-mapped array

### 16.6 Text Formatting Options
- [ ] `set_printoptions([precision, threshold, ...])`: Set printing options
- [ ] `get_printoptions()`: Get current print options
- [ ] `printoptions(*args, **kwargs)`: Context manager for print options
- [ ] `set_string_function(f[, repr])`: Set Python function for pretty-printing

### 16.7 Base-N Representations
- [ ] `binary_repr(num[, width])`: Binary representation
- [ ] `base_repr(number[, base, padding])`: Base-N representation

### 16.8 Data Sources
- [ ] `DataSource([destpath])`: Generic data source file

---

## 17. Indexing Routines

### 17.1 Generating Index Arrays
- [ ] `c_`: Translate slice objects to concatenation along second axis
- [ ] `r_`: Translate slice objects to concatenation along first axis
- [ ] `s_`: Build index tuples for arrays
- [ ] `nonzero(a)`: Return indices of non-zero elements
- [ ] `where(condition[, x, y])`: Return elements from x or y depending on condition
- [ ] `indices(dimensions[, dtype, sparse])`: Return grid of indices
- [ ] `ix_(*args)`: Construct open mesh from multiple sequences
- [ ] `ogrid`: Return open multi-dimensional meshgrid
- [ ] `ravel_multi_index(multi_index, dims[, ...])`: Convert multi-index to flat index
- [ ] `unravel_index(indices, shape[, order])`: Convert flat index to multi-index
- [ ] `diag_indices(n[, ndim])`: Return indices to access diagonal
- [ ] `diag_indices_from(arr)`: Return indices to access diagonal of n-D array
- [ ] `mask_indices(n, mask_func[, k])`: Return indices for masked region
- [ ] `tril_indices(n[, k, m])`: Return indices for lower-triangle
- [ ] `tril_indices_from(arr[, k])`: Return lower-triangle indices from array
- [ ] `triu_indices(n[, k, m])`: Return indices for upper-triangle
- [ ] `triu_indices_from(arr[, k])`: Return upper-triangle indices from array

### 17.2 Indexing-like Operations
- [ ] `take(a, indices[, axis, out, mode])`: Take elements from array
- [ ] `take_along_axis(arr, indices, axis)`: Take values along axis using indices
- [ ] `choose(a, choices[, out, mode])`: Construct array from index array
- [ ] `compress(condition, a[, axis, out])`: Return selected slices
- [ ] `diag(v[, k])`: Extract diagonal or construct diagonal array
- [ ] `diagonal(a[, offset, axis1, axis2])`: Return specified diagonals
- [ ] `select(condlist, choicelist[, default])`: Return array from conditions and choices

### 17.3 Inserting Data
- [ ] `place(arr, mask, vals)`: Change elements based on conditional and values
- [ ] `put(a, ind, v[, mode])`: Replace specified elements
- [ ] `put_along_axis(arr, indices, values, axis)`: Put values along axis
- [ ] `putmask(a, mask, values)`: Change elements based on mask
- [ ] `fill_diagonal(a, val[, wrap])`: Fill main diagonal

### 17.4 Iterating Over Arrays
- [ ] `nditer(op[, flags, op_flags, op_dtypes, ...])`: Efficient multi-dimensional iterator
- [ ] `ndenumerate(arr)`: Multi-dimensional index iterator
- [ ] `ndindex(*shape)`: N-dimensional index iterator
- [ ] `nested_iters(op, axes[, flags, op_flags, ...])`: Create nditers for nested loops
- [ ] `flatiter`: Flat iterator object

---

## 18. Polynomials

NumPy has two polynomial modules:

### 18.1 Polynomial Package (numpy.polynomial)

#### 18.1.1 Power Series (polynomial)
- [ ] `polynomial.Polynomial(coef[, domain, window, ...])`: Power series class
- [ ] `polynomial.polyval(x, c)`: Evaluate polynomial
- [ ] `polynomial.polyval2d(x, y, c)`: Evaluate 2-D polynomial
- [ ] `polynomial.polyval3d(x, y, z, c)`: Evaluate 3-D polynomial
- [ ] `polynomial.polyvander(x, deg)`: Vandermonde matrix
- [ ] `polynomial.polyvander2d(x, y, deg)`: 2-D Vandermonde matrix
- [ ] `polynomial.polyvander3d(x, y, z, deg)`: 3-D Vandermonde matrix
- [ ] `polynomial.polyfit(x, y, deg[, rcond, full, w])`: Least-squares fit
- [ ] `polynomial.polycompanion(c)`: Companion matrix
- [ ] `polynomial.polyroots(c)`: Roots
- [ ] `polynomial.polyfromroots(roots)`: Generate from roots
- [ ] Additional: polyadd, polysub, polymul, polymulx, polydiv, polypow, polyder, polyint

#### 18.1.2 Chebyshev Series (chebyshev)
- [ ] `chebyshev.Chebyshev(coef[, domain, window])`: Chebyshev series class
- [ ] `chebyshev.chebval(x, c)`: Evaluate Chebyshev series
- [ ] Similar functions as polynomial package (chebval2d, chebval3d, chebvander, chebfit, etc.)

#### 18.1.3 Legendre Series (legendre)
- [ ] `legendre.Legendre(coef[, domain, window])`: Legendre series class
- [ ] `legendre.legval(x, c)`: Evaluate Legendre series
- [ ] Similar functions as polynomial package

#### 18.1.4 Laguerre Series (laguerre)
- [ ] `laguerre.Laguerre(coef[, domain, window])`: Laguerre series class
- [ ] `laguerre.lagval(x, c)`: Evaluate Laguerre series
- [ ] Similar functions as polynomial package

#### 18.1.5 Hermite Series (hermite)
- [ ] `hermite.Hermite(coef[, domain, window])`: Hermite series class
- [ ] `hermite.hermval(x, c)`: Evaluate Hermite series
- [ ] Similar functions as polynomial package

#### 18.1.6 HermiteE Series (hermite_e)
- [ ] `hermite_e.HermiteE(coef[, domain, window])`: HermiteE series class
- [ ] `hermite_e.hermeval(x, c)`: Evaluate HermiteE series
- [ ] Similar functions as polynomial package

### 18.2 Legacy Polynomials (numpy.poly1d): Lower Priority

---

## 19. Random Sampling (numpy.random)

### 19.1 Generator and Initialization
- [ ] `default_rng([seed])`: Create Generator with default BitGenerator
- [ ] `Generator(bit_generator)`: Random value generator

### 19.2 Generator Methods: Simple Random Data
- [ ] `Generator.random([size, dtype, out])`: Random floats in [0.0, 1.0)
- [ ] `Generator.integers(low[, high, size, dtype, ...])`: Random integers
- [ ] `Generator.choice(a[, size, replace, p, axis, ...])`: Random sample from array
- [ ] `Generator.bytes(length)`: Random bytes

### 19.3 Generator Methods: Permutations
- [ ] `Generator.shuffle(x[, axis])`: Shuffle in-place
- [ ] `Generator.permutation(x[, axis])`: Permuted sequence or range
- [ ] `Generator.permuted(x[, axis, out])`: Permute array elements

### 19.4 Generator Methods: Distributions

#### Uniform
- [ ] `Generator.uniform([low, high, size])`: Uniform distribution

#### Normal
- [ ] `Generator.normal([loc, scale, size])`: Normal (Gaussian) distribution
- [ ] `Generator.standard_normal([size, dtype, out])`: Standard normal distribution

#### Exponential
- [ ] `Generator.exponential([scale, size])`: Exponential distribution

#### Gamma
- [ ] `Generator.gamma(shape[, scale, size])`: Gamma distribution
- [ ] `Generator.standard_gamma(shape[, size, dtype, out])`: Standard gamma

#### Beta
- [ ] `Generator.beta(a, b[, size])`: Beta distribution

#### Binomial
- [ ] `Generator.binomial(n, p[, size])`: Binomial distribution
- [ ] `Generator.negative_binomial(n, p[, size])`: Negative binomial

#### Poisson
- [ ] `Generator.poisson([lam, size])`: Poisson distribution

#### Chi-square
- [ ] `Generator.chisquare(df[, size])`: Chi-square distribution
- [ ] `Generator.noncentral_chisquare(df, nonc[, size])`: Non-central chi-square

#### F-distribution
- [ ] `Generator.f(dfnum, dfden[, size])`: F distribution
- [ ] `Generator.noncentral_f(dfnum, dfden, nonc[, size])`: Non-central F

#### Student's t
- [ ] `Generator.standard_t(df[, size])`: Student's t distribution

#### Distributions with multiple parameters
- [ ] `Generator.dirichlet(alpha[, size])`: Dirichlet distribution
- [ ] `Generator.multinomial(n, pvals[, size])`: Multinomial distribution
- [ ] `Generator.multivariate_normal(mean, cov[, ...])`: Multivariate normal
- [ ] `Generator.multivariate_hypergeometric(colors, ...)`: Multivariate hypergeometric

#### Other distributions
- [ ] `Generator.cauchy([loc, scale, size])`: Cauchy distribution
- [ ] `Generator.geometric(p[, size])`: Geometric distribution
- [ ] `Generator.gumbel([loc, scale, size])`: Gumbel distribution
- [ ] `Generator.hypergeometric(ngood, nbad, nsample[, ...])`: Hypergeometric
- [ ] `Generator.laplace([loc, scale, size])`: Laplace distribution
- [ ] `Generator.logistic([loc, scale, size])`: Logistic distribution
- [ ] `Generator.lognormal([mean, sigma, size])`: Log-normal distribution
- [ ] `Generator.logseries(p[, size])`: Logarithmic series distribution
- [ ] `Generator.pareto(a[, size])`: Pareto distribution
- [ ] `Generator.power(a[, size])`: Power distribution
- [ ] `Generator.rayleigh([scale, size])`: Rayleigh distribution
- [ ] `Generator.standard_cauchy([size])`: Standard Cauchy distribution
- [ ] `Generator.standard_exponential([size, dtype, ...])`: Standard exponential
- [ ] `Generator.triangular(left, mode, right[, size])`: Triangular distribution
- [ ] `Generator.vonmises(mu, kappa[, size])`: von Mises distribution
- [ ] `Generator.wald(mean, scale[, size])`: Wald (inverse Gaussian) distribution
- [ ] `Generator.weibull(a[, size])`: Weibull distribution
- [ ] `Generator.zipf(a[, size])`: Zipf distribution

### 19.5 BitGenerators
- [ ] `MT19937([seed])`: Mersenne Twister
- [ ] `PCG64([seed])`: PCG-64 (default)
- [ ] `PCG64DXSM([seed])`: PCG-64 DXSM
- [ ] `Philox([seed, counter, key])`: Philox
- [ ] `SFC64([seed])`: SFC-64

### 19.6 Legacy Random Generation (Lower Priority, for compatibility)
- [ ] `RandomState([seed])`: Legacy random state
- [ ] All legacy methods (rand, randn, randint, etc.)

---

## 20. Set Routines

- [ ] `unique(ar[, return_index, return_inverse, ...])`: Find unique elements
- [ ] `in1d(ar1, ar2[, assume_unique, invert])`: Test whether elements of 1-D array in another
- [ ] `intersect1d(ar1, ar2[, assume_unique, ...])`: Find intersection
- [ ] `isin(element, test_elements[, ...])`: Test element-wise if in test_elements
- [ ] `setdiff1d(ar1, ar2[, assume_unique])`: Set difference
- [ ] `setxor1d(ar1, ar2[, assume_unique])`: Set exclusive-or
- [ ] `union1d(ar1, ar2)`: Set union

---

## 21. Sorting, Searching, and Counting

### 21.1 Sorting
- [ ] `sort(a[, axis, kind, order])`: Return sorted copy
- [ ] `lexsort(keys[, axis])`: Indirect stable sort using sequence of keys
- [ ] `argsort(a[, axis, kind, order])`: Return indices that would sort
- [ ] `ndarray.sort([axis, kind, order])`: Sort in-place
- [ ] `sort_complex(a)`: Sort complex array using real part first, then imaginary

### 21.2 Searching
- [ ] `argmax(a[, axis, out, keepdims])`: Indices of maximum values
- [ ] `nanargmax(a[, axis, keepdims])`: Indices of max ignoring NaNs
- [ ] `argmin(a[, axis, out, keepdims])`: Indices of minimum values
- [ ] `nanargmin(a[, axis, keepdims])`: Indices of min ignoring NaNs
- [ ] `argwhere(a)`: Find indices where condition is non-zero
- [ ] `nonzero(a)`: Return indices of non-zero elements
- [ ] `flatnonzero(a)`: Return flat indices of non-zero elements
- [ ] `where(condition[, x, y])`: Return elements from x or y depending on condition
- [ ] `searchsorted(a, v[, side, sorter])`: Find indices to insert elements
- [ ] `extract(condition, arr)`: Return elements satisfying condition

### 21.3 Counting
- [ ] `count_nonzero(a[, axis, keepdims])`: Count non-zero elements

---

## 22. Statistics

### 22.1 Order Statistics
- [ ] `ptp(a[, axis, out, keepdims])`: Range (max: min)
- [ ] `percentile(a, q[, axis, out, ...])`: Compute percentile
- [ ] `nanpercentile(a, q[, axis, out, ...])`: Percentile ignoring NaNs
- [ ] `quantile(a, q[, axis, out, ...])`: Compute quantile
- [ ] `nanquantile(a, q[, axis, out, ...])`: Quantile ignoring NaNs

### 22.2 Averages and Variances
- [ ] `median(a[, axis, out, overwrite_input, ...])`: Compute median
- [ ] `average(a[, axis, weights, returned, ...])`: Compute weighted average
- [ ] `mean(a[, axis, dtype, out, keepdims, ...])`: Compute arithmetic mean
- [ ] `std(a[, axis, dtype, out, ddof, ...])`: Compute standard deviation
- [ ] `var(a[, axis, dtype, out, ddof, ...])`: Compute variance
- [ ] `nanmedian(a[, axis, out, overwrite_input, ...])`: Median ignoring NaNs
- [ ] `nanmean(a[, axis, dtype, out, keepdims, ...])`: Mean ignoring NaNs
- [ ] `nanstd(a[, axis, dtype, out, ddof, ...])`: Standard deviation ignoring NaNs
- [ ] `nanvar(a[, axis, dtype, out, ddof, ...])`: Variance ignoring NaNs

### 22.3 Correlating
- [ ] `corrcoef(x[, y, rowvar, bias, ddof])`: Pearson correlation coefficients
- [ ] `correlate(a, v[, mode])`: Cross-correlation
- [ ] `cov(m[, y, rowvar, bias, ddof, ...])`: Covariance matrix

### 22.4 Histograms
- [ ] `histogram(a[, bins, range, density, ...])`: Compute histogram
- [ ] `histogram2d(x, y[, bins, range, density, ...])`: Compute 2-D histogram
- [ ] `histogramdd(sample[, bins, range, ...])`: Compute multidimensional histogram
- [ ] `bincount(x[, weights, minlength])`: Count occurrences of each value
- [ ] `histogram_bin_edges(a[, bins, range, ...])`: Calculate histogram bin edges
- [ ] `digitize(x, bins[, right])`: Return indices of bins to which each value belongs

---

## 23. Test Support (numpy.testing)

- [ ] `testing.assert_allclose(actual, desired[, ...])`: Assert arrays are equal within tolerance
- [ ] `testing.assert_array_almost_equal(x, y[, ...])`: Assert arrays are almost equal
- [ ] `testing.assert_array_equal(x, y[, err_msg, ...])`: Assert arrays are equal
- [ ] `testing.assert_array_less(x, y[, err_msg, ...])`: Assert x < y element-wise
- [ ] `testing.assert_equal(actual, desired[, ...])`: Assert objects are equal
- [ ] `testing.assert_raises(exception_class[, ...])`: Assert callable raises exception
- [ ] `testing.assert_warns(warning_class[, ...])`: Assert callable produces warning
- [ ] `testing.assert_string_equal(actual, desired)`: Assert strings are equal
- [ ] `testing.assert_approx_equal(actual, desired[, ...])`: Assert numbers are approximately equal
- [ ] Additional testing utilities...

---

## 24. Window Functions

- [ ] `bartlett(M)`: Bartlett window
- [ ] `blackman(M)`: Blackman window
- [ ] `hamming(M)`: Hamming window
- [ ] `hanning(M)`: Hanning window
- [ ] `kaiser(M, beta)`: Kaiser window

---

## 25. Miscellaneous Routines

### 25.1 Buffer Protocol
- [ ] `frombuffer(buffer[, dtype, count, offset])`: Interpret buffer as 1-D array

### 25.2 Performance Tuning
- [ ] `shares_memory(a, b[, max_work])`: Check if arrays share memory
- [ ] `may_share_memory(a, b[, max_work])`: Check if arrays might share memory
- [ ] `byte_bounds(a)`: Return pointers to end-points of array

### 25.3 Array Mixins
- [ ] `lib.mixins.NDArrayOperatorsMixin`: Mixin for operator support

---

## 26. NumPy-specific Configuration

### 26.1 Configuration
- [ ] `show_config()`: Show NumPy build configuration
- [ ] `get_include()`: Return directory containing NumPy C header files

### 26.2 Compilation
- [ ] `distutils` module (deprecated in favor of other build systems)
