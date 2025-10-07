# NumPy.js - Project Overview

## Mission

Create a complete, functionally-equivalent implementation of NumPy 2.0+ for TypeScript/JavaScript with the following principles:

1. **Correctness First**: Prioritize functional equivalency and correctness over performance
2. **Comprehensive Coverage**: Implement the complete NumPy 2.0+ Python API (no legacy/deprecated functionality)
3. **Full Compatibility**: Support .npy and .npz file format compatibility
4. **Rigorous Testing**: Build a comprehensive test harness comparing our implementation against Python NumPy

## Project Goals

### Primary Goals
- ✅ Full NumPy 2.0+ API implementation in TypeScript
- ✅ Functional equivalency with Python NumPy
- ✅ .npy/.npz file format support
- ✅ Cross-validation testing framework (TypeScript ↔ Python)
- ✅ Type-safe TypeScript interfaces

### Secondary Goals (Post-v1.0)
- Performance optimization (consider Rust/WASM for hot paths)
- SIMD optimization where applicable
- GPU acceleration for specific operations
- Streaming operations for large datasets

## Non-Goals (Initial Release)
- C API compatibility (Python-only API focus)
- Legacy NumPy 1.x deprecated functions
- Cryptographic security for random number generation
- Matching Python NumPy's exact performance characteristics

## Success Criteria

A successful v1.0 release will:
1. Pass 100% of functionality tests against Python NumPy
2. Support all public NumPy 2.0+ API functions
3. Successfully read/write .npy and .npz files compatible with Python
4. Provide comprehensive TypeScript type definitions
5. Include documentation with examples for all major API surfaces

## Architecture Philosophy

### Correctness-First Development
1. Implement functionality correctly first
2. Validate against Python NumPy
3. Optimize only after validation

### Modular Design
- Separate core array operations from higher-level functions
- Plugin architecture for performance backends (pure TS → WASM later)
- Clean separation between:
  - Data structures (NDArray, DType, etc.)
  - Core operations (element-wise, reduction, etc.)
  - High-level APIs (linalg, fft, random, etc.)
  - I/O operations (npy/npz support)

## Technology Stack

### Core Implementation
- **Language**: TypeScript (strict mode)
- **Runtime**: Node.js + Browser compatible
- **Build**: To be determined (likely esbuild or rollup)

### Testing Infrastructure
- **Unit Tests**: Vitest or Jest
- **Cross-validation**: Python subprocess calls from Node.js
- **Performance Benchmarks**: Benchmark.js or similar

### Development Tools
- **Type Checking**: TypeScript compiler
- **Linting**: ESLint
- **Formatting**: Prettier
- **Documentation**: TypeDoc

## Development Phases

### Phase 1: Foundation
- Core data structures (NDArray, DType system)
- Basic array creation and manipulation
- Element-wise operations
- Testing infrastructure

### Phase 2: Core Functionality
- Mathematical functions
- Linear algebra basics
- Array manipulation routines
- Broadcasting implementation

### Phase 3: Advanced Features
- Random number generation
- FFT operations
- Statistics
- Polynomial operations

### Phase 4: I/O and Polish
- .npy/.npz file format support
- Comprehensive documentation
- Performance baseline establishment
- v1.0 release preparation

### Phase 5: Optimization
- Performance profiling
- WASM backend for critical paths
- Advanced optimizations

## Repository Structure

```
numpy-js/
├── docs/                    # Comprehensive documentation
│   ├── 00-PROJECT-OVERVIEW.md
│   ├── 01-API-INVENTORY.md
│   ├── 02-ARCHITECTURE.md
│   ├── 03-TESTING-STRATEGY.md
│   ├── 04-NPY-FORMAT-SPEC.md
│   ├── 05-IMPLEMENTATION-ROADMAP.md
│   └── 06-API-COVERAGE-MATRIX.md
├── src/
│   ├── core/               # Core array and dtype implementation
│   ├── lib/                # High-level API implementations
│   ├── io/                 # File I/O (npy/npz)
│   ├── types/              # TypeScript type definitions
│   └── index.ts            # Main entry point
├── tests/
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── validation/         # Python cross-validation tests
│   └── benchmarks/         # Performance benchmarks
├── scripts/
│   └── test-harness/       # Python-Node.js test orchestration
└── package.json
```

## Key Design Decisions

### TypeScript Over JavaScript
- Type safety critical for numerical computing
- Better IDE support and developer experience
- Easier refactoring as codebase grows

### Correctness Over Performance (Initially)
- NumPy API is complex; correctness is paramount
- Easier to optimize correct code than fix optimized buggy code
- Performance improvements can be incremental

### Python Test Oracle
- Python NumPy is the source of truth
- Cross-validation ensures functional equivalency
- Automated comparison reduces manual testing burden

### Modular Backend Architecture
- Start with pure TypeScript implementation
- Design allows swapping in WASM/native backends later
- Each operation can be independently optimized

## Open Questions

1. **Broadcasting**: Implement using views or materialized copies?
2. **Memory Layout**: Row-major only or support column-major (F-order)?
3. **Integer Types**: How to handle int64/uint64 with JavaScript's number limitations?
4. **Complex Numbers**: Native support or separate real/imag arrays?
5. **Error Handling**: Match NumPy's warnings or use TypeScript exceptions?

## Resources

- [NumPy 2.0 Documentation](https://numpy.org/doc/2.0/)
- [NumPy Enhancement Proposals (NEPs)](https://numpy.org/neps/)
- [.npy Format Specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
- [Array API Standard](https://data-apis.org/array-api/latest/)

---

**Last Updated**: 2025-10-07
**Status**: Planning Phase
