# Contributing to numpy-ts

Thank you for your interest in contributing to numpy-ts! We welcome contributions of all kinds.

## Getting Started

```bash
git clone https://github.com/dupontcyborg/numpy-ts.git
cd numpy-ts
npm install
npm test
```

## Development Workflow

### 1. Pick a Function

Choose an unimplemented function from [docs/API-REFERENCE.md](docs/API-REFERENCE.md). Functions are organized by category:
- Array Creation
- Array Manipulation
- Mathematical Operations
- Linear Algebra
- Reductions
- And more...

**Tip:** Start with "Medium priority" functions for quick wins.

### 2. Implement the Function

Add your implementation in the appropriate file:
- Array creation → `src/lib/creation.ts`
- Arithmetic operations → `src/ops/arithmetic.ts`
- Linear algebra → `src/ops/linalg.ts`
- Reductions → `src/ops/reductions.ts`

**Don't forget to export** your function in `src/index.ts`!

### 3. Add Tests

Every new function requires three types of tests:

#### Unit Tests (`tests/unit/`)

Fast, isolated tests of your implementation:

```typescript
// tests/unit/math.test.ts
import { describe, it, expect } from 'vitest';
import * as np from '../../src/index.js';

describe('sqrt', () => {
  it('computes square root element-wise', () => {
    const arr = np.array([4, 9, 16]);
    const result = np.sqrt(arr);
    expect(result.toArray()).toEqual([2, 3, 4]);
  });

  it('preserves shape', () => {
    const arr = np.ones([2, 3]);
    const result = np.sqrt(arr);
    expect(result.shape).toEqual([2, 3]);
  });
});
```

Run unit tests:
```bash
npm run test:unit
```

#### Validation Tests (`tests/validation/`)

Cross-validate against Python NumPy:

```typescript
// tests/validation/math.numpy.test.ts
import { describe, it, expect } from 'vitest';
import * as np from '../../src/index.js';
import { runNumPy } from '../helpers/numpy-runner.js';

describe('NumPy Validation: sqrt', () => {
  it('matches NumPy sqrt output', () => {
    const input = [1, 4, 9, 16, 25];
    const result = np.sqrt(np.array(input));

    const npResult = runNumPy(`
      import numpy as np
      arr = np.array([1, 4, 9, 16, 25])
      result = np.sqrt(arr)
    `);

    expect(result.toArray()).toEqual(npResult);
  });
});
```

**Important:** Validation tests require Python + NumPy:
```bash
source ~/.zshrc && conda activate py313
npm run test:validation
```

#### Benchmarks (`benchmarks/`)

Compare performance against NumPy. Add to three files:

**1. benchmarks/src/specs.ts**
```typescript
{
  name: 'sqrt',
  category: 'math',
  sizes: [1000, [100, 100], [500, 500]],
  setup: (size) => ({
    arr: np.random.randn(size)
  })
}
```

**2. benchmarks/src/runner.ts**
```typescript
case 'sqrt':
  return () => np.sqrt(inputs.arr);
```

**3. benchmarks/scripts/numpy_benchmark.py**
```python
elif spec_name == 'sqrt':
    return lambda: np.sqrt(inputs['arr'])
```

Run benchmarks:
```bash
source ~/.zshrc && conda activate py313
npm run bench:quick    # Fast (~2-3 min)
npm run bench:view     # View results
```

### 4. Update Documentation

After implementing and testing, update [docs/API-REFERENCE.md](docs/API-REFERENCE.md):

```markdown
- [x] `sqrt(x)` - Square root
```

Update progress counts from test output.

### 5. Run All Checks

```bash
# Type check
npm run typecheck

# Lint and format
npm run lint
npm run format

# Run all tests (requires conda env)
source ~/.zshrc && conda activate py313
npm test

# Or quick tests (unit only, no Python needed)
npm run test:quick
```

### 6. Submit Pull Request

1. Create a branch: `git checkout -b feature/add-sqrt`
2. Commit your changes: `git commit -m "Add sqrt function with tests"`
3. Push: `git push origin feature/add-sqrt`
4. Open a PR on GitHub

## Testing Strategy

### Two-Tier Approach

1. **Unit Tests** (1365+ tests)
   - Fast, no external dependencies
   - Test implementation correctness
   - Run frequently during development

2. **NumPy Validation**
   - Cross-validate against Python NumPy
   - Ensure NumPy compatibility
   - Catch edge cases and overflow behavior

### Test Commands

```bash
# Unit tests only (fast, no Python needed)
npm run test:unit

# All tests (requires Python + NumPy)
source ~/.zshrc && conda activate py313
npm test

# Quick tests (skip slow validation)
npm run test:quick

# Validation only
source ~/.zshrc && conda activate py313
npm run test:validation

# Watch mode
npm run test:watch
```

## Python/Conda Setup

Validation tests and benchmarks require Python with NumPy:

```bash
# Activate conda environment
source ~/.zshrc && conda activate py313

# Verify activation
echo $CONDA_DEFAULT_ENV  # Should print: py313
```

If you see "Python NumPy not available" errors, make sure conda is activated.

## Code Style

- **TypeScript** — Use strict types, avoid `any`
- **Formatting** — Run `npm run format` before committing
- **Linting** — Run `npm run lint` and fix issues
- **Comments** — Add JSDoc for public APIs
- **Tests** — Every function needs unit + validation tests

## API Coverage Tracking

Check current API coverage and identify gaps:

```bash
# Update README with current coverage
python scripts/compare-api-coverage.py

# Show detailed list of missing functions
python scripts/compare-api-coverage.py --verbose

# Short form
python scripts/compare-api-coverage.py -v
```

This script:
- Audits NumPy's API (top-level functions and ndarray methods)
- Audits numpy-ts's implementation
- Compares them properly (member-vs-member, global-vs-global)
- Updates README.md with accurate coverage statistics
- With `--verbose`: Shows complete list of missing/extra functions

## What to Contribute

### High Priority

- [ ] Trigonometric functions (`sin`, `cos`, `tan`, etc.)
- [ ] Exponential/logarithmic functions (`exp`, `log`, etc.)
- [ ] Random number generation (`random.randn`, `random.randint`, etc.)
- [ ] Sorting and searching (`sort`, `argsort`, `searchsorted`)
- [ ] Linear algebra decompositions (`qr`, `svd`, `eig`)

### Medium Priority

- [ ] Rounding functions (`round`, `floor`, `ceil`)
- [ ] Logic functions (`logical_and`, `logical_or`, etc.)
- [ ] Set operations (`unique`, `intersect1d`, etc.)
- [ ] Polynomial functions

### Low Priority

- [ ] FFT functions
- [ ] Advanced indexing
- [ ] Masked arrays

See [docs/API-REFERENCE.md](docs/API-REFERENCE.md) for the complete list.

## Tips for Contributors

1. **Batch related functions** — e.g., implement all `*_like` functions together
2. **Read existing code** — See how similar functions are implemented
3. **Check NumPy docs** — Ensure your implementation matches NumPy behavior
4. **Test edge cases** — Overflow, broadcasting, empty arrays, etc.
5. **Ask questions** — Open an issue if you're unsure about anything

## Documentation

Before contributing, read:
- [docs/TESTING-GUIDE.md](docs/TESTING-GUIDE.md) — Comprehensive testing guide
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System design
- [docs/API-REFERENCE.md](docs/API-REFERENCE.md) — Function checklist

## Questions?

- **Issues:** https://github.com/dupontcyborg/numpy-ts/issues
- **Discussions:** https://github.com/dupontcyborg/numpy-ts/discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
