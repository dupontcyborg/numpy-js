#!/usr/bin/env node
import { build, Plugin } from 'esbuild';
import path from 'node:path';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';

// Read version from package.json
const packageJson = JSON.parse(readFileSync('./package.json', 'utf-8'));
const VERSION = packageJson.version;

// Selective alias: Only redirect @stdlib packages that try to load native addons
// This bypasses the native addon loading in lib/index.js and uses pure JS implementations
// But preserves the normal lib/index.js for packages that don't have native addons
function stdlibAutoMainAlias(): Plugin {
  return {
    name: 'stdlib-auto-main-alias',
    setup(b) {
      // Only redirect specific @stdlib BLAS packages that load native addons
      // Other packages like layouts, orders, etc. need their lib/index.js setup code
      const NATIVE_ADDON_PACKAGES = [
        '@stdlib/blas/base/dgemm',
        '@stdlib/blas/base/ddot',
        '@stdlib/blas/base/daxpy',
        '@stdlib/blas/base/dscal',
        '@stdlib/blas/base/dswap',
        '@stdlib/blas/base/dcopy',
        '@stdlib/blas/base/dasum',
        '@stdlib/blas/base/dnrm2',
        '@stdlib/blas/base/idamax',
        '@stdlib/blas/base/sgemm',
        '@stdlib/blas/base/sdot',
        '@stdlib/blas/ddot',
        '@stdlib/blas/daxpy',
        '@stdlib/blas/dscal',
        '@stdlib/blas/dswap',
        '@stdlib/blas/dcopy',
        '@stdlib/blas/dasum',
        '@stdlib/blas/dnrm2',
        '@stdlib/blas/idamax',
      ];

      const filter = /.*/;
      b.onResolve({ filter }, (args) => {
        const req = createRequire(args.resolveDir + '/package.json');

        // Only redirect if this is one of the packages that loads native addons
        if (NATIVE_ADDON_PACKAGES.some(pkg => args.path === pkg || args.path.startsWith(pkg + '/'))) {
          try {
            // Extract the package name (before any sub-path)
            let pkgName = args.path;
            const match = args.path.match(/^(@stdlib\/[^/]+(?:\/[^/]+)*)/);
            if (match) {
              pkgName = match[1];
            }

            // Try to resolve to lib/main.js for this package
            const target = `${pkgName}/lib/main.js`;
            const resolved = req.resolve(target);
            return { path: path.normalize(resolved) };
          } catch {
            // Fallback to normal resolution
            return null;
          }
        }

        // For all other imports, let esbuild handle normally
        return null;
      });
    },
  };
}

async function buildAll() {
  console.log('Building Node.js bundle...');
  // 1) Node CJS — no alias needed
  // Using .cjs extension so it's treated as CommonJS even with "type": "module" in package.json
  await build({
    entryPoints: ['src/index.ts'],
    bundle: true,
    platform: 'node',
    format: 'cjs',
    outfile: 'dist/numpy.node.cjs',
    sourcemap: true,
    minify: true,
    define: {
      '__VERSION_PLACEHOLDER__': JSON.stringify(VERSION)
    }
  });
  console.log('✓ Node.js build complete');

  console.log('Building browser bundle...');
  // 2) Browser IIFE — alias applied
  await build({
    entryPoints: ['src/index.ts'],
    bundle: true,
    platform: 'browser',
    format: 'iife',
    globalName: 'np',
    outfile: 'dist/numpy.browser.js',
    sourcemap: true,
    minify: true,
    plugins: [stdlibAutoMainAlias()],
    define: {
      '__VERSION_PLACEHOLDER__': JSON.stringify(VERSION)
    }
  });
  console.log('✓ Browser build complete');

  console.log('Building ESM bundle...');
  // 3) ESM for browsers and bundlers
  // Using platform=browser (same as IIFE) with selective stdlib aliasing
  // Only BLAS packages with native addons are redirected to lib/main.js
  // Other packages use normal lib/index.js to preserve setup code
  await build({
    entryPoints: ['src/index.ts'],
    bundle: true,
    platform: 'browser',
    format: 'esm',
    outfile: 'dist/numpy.esm.js',
    sourcemap: true,
    minify: true,
    plugins: [stdlibAutoMainAlias()],
    define: {
      '__VERSION_PLACEHOLDER__': JSON.stringify(VERSION)
    }
  });
  console.log('✓ ESM build complete');

  console.log('\n✓ All builds completed successfully!');
}

buildAll().catch((err) => {
  console.error('Build failed:', err);
  process.exit(1);
});
