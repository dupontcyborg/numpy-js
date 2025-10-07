# NumPy.js Documentation

## Overview

This directory contains comprehensive planning and specification documents for NumPy.js - a complete TypeScript/JavaScript implementation of NumPy 2.0+.

---

## Documents

### [00-PROJECT-OVERVIEW.md](./00-PROJECT-OVERVIEW.md)
High-level project overview including mission, goals, architecture philosophy, and development phases.

**Key Topics:**
- Project mission and goals
- Success criteria
- Technology stack
- Repository structure
- Open questions

---

### [01-API-INVENTORY.md](./01-API-INVENTORY.md)
**Complete** inventory of all 800+ NumPy 2.0+ Python API functions organized by category.

**Key Topics:**
- Core array object (NDArray)
- Array creation routines (~40 functions)
- Array manipulation (~50 functions)
- Mathematical functions (~120 functions)
- Linear algebra (~40 functions)
- FFT, Random, Statistics, etc.
- Implementation priority by phase

**Status:** ✅ Comprehensive checklist with ~800-900 functions catalogued

---

### [02-ARCHITECTURE.md](./02-ARCHITECTURE.md)
Detailed technical architecture design with "correctness first" philosophy.

**Key Topics:**
- High-level architecture layers
- Core classes (NDArray, DType, Broadcasting, Ufuncs)
- Memory management strategy
- Implementation strategy (pure TS → optimized)
- Special considerations (int64, complex numbers, etc.)
- Extensibility points

**Status:** ✅ Complete architecture specification

---

### [03-TESTING-STRATEGY.md](./03-TESTING-STRATEGY.md)
Comprehensive testing approach using Python NumPy as oracle.

**Key Topics:**
- Testing pyramid (unit → validation → integration → benchmarks)
- Cross-validation with Python NumPy
- Test harness design (subprocess vs HTTP)
- Numerical comparison strategies
- Property-based testing
- CI/CD pipeline

**Status:** ✅ Complete testing strategy

---

### [04-NPY-FORMAT-SPEC.md](./04-NPY-FORMAT-SPEC.md)
Detailed specification for .npy and .npz file format support.

**Key Topics:**
- .npy format structure (versions 1.0, 2.0, 3.0)
- Header format and parsing
- .npz format (ZIP-based)
- Read/write algorithms
- Memory-mapped files
- Edge cases and compatibility

**Status:** ✅ Complete file format specification

---

### [05-IMPLEMENTATION-ROADMAP.md](./05-IMPLEMENTATION-ROADMAP.md)
Phased implementation plan with 56 milestones over 12-18 months.

**Key Topics:**
- Phase 0: Project Setup (Weeks 1-2)
- Phase 1: Core Foundation (Months 1-3)
- Phase 2: Core Functionality (Months 4-6)
- Phase 3: Advanced Features (Months 7-9)
- Phase 4: I/O and Specialized (Months 10-12)
- Phase 5: Polish and Release (Month 12+)
- Post-v1.0: Optimization phases

**Status:** ✅ Complete 12-month roadmap to v1.0

---

### [06-DTYPE-SPECIFICATION.md](./06-DTYPE-SPECIFICATION.md)
Complete specification of all supported data types.

**Key Topics:**
- Boolean, integers, floats
- Complex numbers (interleaved storage)
- 64-bit integers (BigInt strategy)
- Datetime/timedelta
- Strings (byte and Unicode)
- Structured dtypes
- Object arrays
- Type promotion rules

**Status:** ✅ Complete dtype specification

---

### [07-DESIGN-DECISIONS.md](./07-DESIGN-DECISIONS.md)
Record of all major design decisions with rationale.

**Key Decisions:**
1. **64-bit integers**: Use BigInt for correctness
2. **Complex numbers**: Nested array format `[[real, imag]]`
3. **Linear algebra**: Pure TS first, WASM later
4. **FFT**: Use fft.js library
5. **Browser/Node**: Separate I/O APIs
6. **Performance**: Correctness first, never compromise
7. **Object arrays**: JS objects only (no pickle)
8. **Random**: PCG64 with WebCrypto seeding
9. **Memory**: Auto GC + optional `.dispose()`
10. **JS extensions**: Support both NumPy API and JS idioms

**Status:** ✅ Core decisions finalized

---

## Quick Start

### For Contributors

1. **Start here**: [00-PROJECT-OVERVIEW.md](./00-PROJECT-OVERVIEW.md)
2. **Understand architecture**: [02-ARCHITECTURE.md](./02-ARCHITECTURE.md)
3. **Pick a task**: [05-IMPLEMENTATION-ROADMAP.md](./05-IMPLEMENTATION-ROADMAP.md)
4. **Check API**: [01-API-INVENTORY.md](./01-API-INVENTORY.md)
5. **Follow decisions**: [07-DESIGN-DECISIONS.md](./07-DESIGN-DECISIONS.md)

### For Users (Future)

When the project is implemented:
1. Installation: `npm install numpy-js`
2. Quick start guide: (TBD)
3. API documentation: (TBD)
4. Examples: (TBD)

---

## Project Status

### Planning Phase: ✅ COMPLETE

All planning documents are complete and comprehensive:

- ✅ Project scope and goals defined
- ✅ Complete API inventory (800+ functions)
- ✅ Architecture designed (correctness-first)
- ✅ Testing strategy established
- ✅ File format specifications documented
- ✅ Implementation roadmap created
- ✅ Data types specified
- ✅ Design decisions finalized

### Next Steps

**Phase 0: Project Setup** (Weeks 1-2)
- [ ] Initialize npm project
- [ ] Configure TypeScript
- [ ] Set up test framework
- [ ] Create Python test oracle
- [ ] Set up CI/CD

See [05-IMPLEMENTATION-ROADMAP.md](./05-IMPLEMENTATION-ROADMAP.md) for detailed next steps.

---

## Key Metrics

### Scope
- **Functions to implement**: ~800-900
- **DTypes supported**: 20+ (bool, int8-64, uint8-64, float16-64, complex64-128, datetime64, timedelta64, strings, structured, object)
- **Documentation pages**: 7 comprehensive docs
- **Estimated timeline**: 12-18 months to v1.0

### Goals
- ✅ 100% NumPy 2.0+ API coverage
- ✅ Full .npy/.npz compatibility
- ✅ Cross-validation with Python NumPy
- ✅ Comprehensive documentation
- ✅ Browser and Node.js support

---

## Design Philosophy

### Core Principles

1. **Correctness First**
   - Never compromise correctness for performance
   - Match NumPy behavior exactly
   - Extensive validation against Python

2. **Clean Architecture**
   - Simple, understandable implementations
   - Maintainable codebase
   - Clear abstractions

3. **Testability**
   - Every function validated against Python
   - Comprehensive test coverage
   - Property-based testing

4. **Extensibility**
   - Plugin architecture for backends
   - WASM/GPU optimization paths
   - JavaScript-friendly extensions

5. **Transparency**
   - Document all decisions
   - Clear about limitations
   - Explicit trade-offs

---

## Contributing

### Development Workflow (Future)

1. Pick a function from [01-API-INVENTORY.md](./01-API-INVENTORY.md)
2. Implement following [02-ARCHITECTURE.md](./02-ARCHITECTURE.md)
3. Test using strategy from [03-TESTING-STRATEGY.md](./03-TESTING-STRATEGY.md)
4. Validate against Python NumPy
5. Submit PR

### Guidelines

- **Read docs first**: Understand architecture and decisions
- **Write tests**: Cross-validate with Python
- **Follow conventions**: TypeScript strict mode, clear naming
- **Document**: JSDoc for all public APIs
- **Ask questions**: Open issues for clarification

---

## Resources

### NumPy Documentation
- [NumPy 2.0 Reference](https://numpy.org/doc/2.0/reference/)
- [NumPy Enhancement Proposals](https://numpy.org/neps/)
- [Array API Standard](https://data-apis.org/array-api/latest/)

### Related Projects
- [numjs](https://github.com/nicolaspanel/numjs) - Partial NumPy implementation
- [ndarray](https://github.com/scijs/ndarray) - Multidimensional arrays
- [tensorflow.js](https://www.tensorflow.org/js) - ML in JavaScript
- [fft.js](https://github.com/indutny/fft.js) - FFT implementation

### Discussions
- GitHub Issues: (TBD)
- Discord: (TBD)

---

## License

(TBD - likely MIT or Apache 2.0)

---

## Acknowledgments

This project aims to bring the power of NumPy to the JavaScript ecosystem. We're grateful to the NumPy team for creating such an amazing library and for their excellent documentation.

---

**Last Updated**: 2025-10-07
**Planning Status**: ✅ Complete
**Implementation Status**: Not started (ready to begin Phase 0)
