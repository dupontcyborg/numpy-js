#!/usr/bin/env python3
"""
Compare numpy-ts vs NumPy API and update documentation.

Compares:
- Top-level functions vs top-level functions (ignoring if also methods)
- ndarray methods vs NDArray methods (ignoring if also functions)

Usage:
    python scripts/compare-api-coverage.py              # Update README
    python scripts/compare-api-coverage.py --verbose    # Show detailed gaps
    python scripts/compare-api-coverage.py -v           # Short form
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run_audits():
    """Run both audit scripts to generate fresh JSON files."""
    scripts_dir = Path(__file__).parent

    print("Running audits to generate fresh data...\n")

    # Run NumPy audit
    print("1. Auditing NumPy API...")
    result = subprocess.run(
        [sys.executable, scripts_dir / 'audit-numpy-api.py'],
        cwd=scripts_dir.parent,
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode != 0:
        print(f"Error running NumPy audit:\n{result.stderr}")
        sys.exit(1)

    # Run numpy-ts audit
    print("2. Auditing numpy-ts API...")
    result = subprocess.run(
        ['npx', 'tsx', scripts_dir / 'audit-numpyts-api.ts'],
        cwd=scripts_dir.parent,
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode != 0:
        print(f"Error running numpy-ts audit:\n{result.stderr}")
        sys.exit(1)

    print("✅ Audits complete\n")


def load_json(filename):
    """Load JSON file."""
    filepath = Path(__file__).parent / filename
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_verbose_gaps(numpy_audit, numpyts_audit):
    """Print detailed list of missing and extra functions."""
    numpy_toplevel = set()
    for funcs in numpy_audit['categorized'].values():
        numpy_toplevel.update(funcs)

    numpyts_toplevel = set(numpyts_audit['all_functions'].keys())
    numpy_methods = set(numpy_audit['ndarray_methods'].keys())
    numpyts_methods = set(numpyts_audit['ndarray_methods'].keys())

    print("\n" + "=" * 70)
    print("VERBOSE GAP ANALYSIS")
    print("=" * 70)

    # Top-level function gaps
    missing_toplevel = numpy_toplevel - numpyts_toplevel
    extra_toplevel = numpyts_toplevel - numpy_toplevel

    print("\nTOP-LEVEL FUNCTIONS:")
    print(f"  Missing from numpy-ts ({len(missing_toplevel)}):")
    for i, func in enumerate(sorted(missing_toplevel), 1):
        print(f"    {i:3d}. {func}")

    if extra_toplevel:
        print(f"\n  Extra in numpy-ts (not in NumPy, {len(extra_toplevel)}):")
        for i, func in enumerate(sorted(extra_toplevel), 1):
            print(f"    {i:3d}. {func}")

    # Method gaps
    missing_methods = numpy_methods - numpyts_methods
    extra_methods = numpyts_methods - numpy_methods

    print("\nNDARRAY METHODS:")
    print(f"  Missing from NDArray ({len(missing_methods)}):")
    for i, method in enumerate(sorted(missing_methods), 1):
        print(f"    {i:3d}. {method}")

    if extra_methods:
        print(f"\n  Extra in NDArray (not in ndarray, {len(extra_methods)}):")
        for i, method in enumerate(sorted(extra_methods), 1):
            print(f"    {i:3d}. {method}")

    print("\n" + "=" * 70)


def analyze_coverage(verbose=False):
    """Analyze API coverage between NumPy and numpy-ts."""
    numpy_audit = load_json('numpy-api-audit.json')
    numpyts_audit = load_json('numpyts-api-audit.json')

    # Get ALL NumPy top-level functions (from categorized)
    numpy_toplevel = set()
    for funcs in numpy_audit['categorized'].values():
        numpy_toplevel.update(funcs)

    # Get ALL numpy-ts top-level functions
    numpyts_toplevel = set(numpyts_audit['all_functions'].keys())

    # Get methods
    numpy_methods = set(numpy_audit['ndarray_methods'].keys())
    numpyts_methods = set(numpyts_audit['ndarray_methods'].keys())

    print("=" * 70)
    print("ACCURATE API COVERAGE ANALYSIS")
    print("=" * 70)

    # Top-level comparison (ignoring if also methods)
    toplevel_implemented = numpy_toplevel & numpyts_toplevel
    toplevel_coverage = (
        100 * len(toplevel_implemented) / len(numpy_toplevel)
        if numpy_toplevel else 0
    )

    print("\nTOP-LEVEL FUNCTIONS (ignoring if also methods):")
    print(f"  NumPy functions:         {len(numpy_toplevel)}")
    print(f"  numpy-ts functions:      {len(numpyts_toplevel)}")
    print(f"  Implemented:             {len(toplevel_implemented)}")
    print(f"  Coverage:                "
          f"{len(toplevel_implemented)}/{len(numpy_toplevel)} "
          f"({toplevel_coverage:.1f}%)")

    # Methods comparison (ignoring if also functions)
    methods_implemented = numpy_methods & numpyts_methods
    methods_coverage = (
        100 * len(methods_implemented) / len(numpy_methods)
        if numpy_methods else 0
    )

    print("\nNDARRAY METHODS (ignoring if also functions):")
    print(f"  NumPy methods:           {len(numpy_methods)}")
    print(f"  numpy-ts methods:        {len(numpyts_methods)}")
    print(f"  Implemented:             {len(methods_implemented)}")
    print(f"  Coverage:                "
          f"{len(methods_implemented)}/{len(numpy_methods)} "
          f"({methods_coverage:.1f}%)")

    # Combined (for reference)
    # Total unique = union of all
    numpy_total = len(numpy_toplevel | numpy_methods)
    numpyts_total = len(numpyts_toplevel | numpyts_methods)
    overall_coverage = 100 * numpyts_total / numpy_total if numpy_total else 0

    print("\nCOMBINED UNIQUE (functions ∪ methods):")
    print(f"  NumPy total:             {numpy_total}")
    print(f"  numpy-ts total:          {numpyts_total}")
    print(f"  Overall coverage:        "
          f"{numpyts_total}/{numpy_total} ({overall_coverage:.1f}%)")

    # Category breakdown
    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN (by top-level functions)")
    print("=" * 70)

    category_stats = {}
    # Use union for implementation check (function OR method)
    numpyts_all = numpyts_toplevel | numpyts_methods

    for category, funcs in sorted(numpy_audit['categorized'].items()):
        numpy_cat = set(funcs)
        implemented = numpy_cat & numpyts_all
        pct = 100 * len(implemented) / len(numpy_cat) if numpy_cat else 0

        status = "✅" if pct == 100 else ("🟡" if pct >= 50 else "🔴")

        category_stats[category] = {
            'total': len(numpy_cat),
            'implemented': len(implemented),
            'missing': numpy_cat - numpyts_all,
            'percentage': pct,
            'status': status
        }

        impl_str = f"{len(implemented):3d}/{len(numpy_cat):3d}"
        pct_str = f"{pct:5.1f}%"
        print(f"{category:35s} {impl_str} ({pct_str}) {status}")

    # Print verbose gaps if requested
    if verbose:
        print_verbose_gaps(numpy_audit, numpyts_audit)

    return {
        'numpy_toplevel': len(numpy_toplevel),
        'numpyts_toplevel': len(numpyts_toplevel),
        'toplevel_coverage': toplevel_coverage,
        'numpy_methods': len(numpy_methods),
        'numpyts_methods': len(numpyts_methods),
        'methods_coverage': methods_coverage,
        'numpy_total': numpy_total,
        'numpyts_total': numpyts_total,
        'overall_coverage': overall_coverage,
        'category_stats': category_stats,
        'numpy_audit': numpy_audit,
        'numpyts_audit': numpyts_audit
    }


def generate_readme_table(category_stats):
    """Generate markdown table for README."""
    lines = []
    lines.append("| Category | Complete | Total | Status |")
    lines.append("|----------|----------|-------|--------|")

    # Sort by percentage (complete first)
    sorted_cats = sorted(
        category_stats.items(),
        key=lambda x: (-x[1]['percentage'], x[0])
    )

    for category, stats in sorted_cats:
        complete = f"{stats['implemented']}/{stats['total']}"
        pct = f"{stats['percentage']:.0f}%"
        status = stats['status']
        category_md = f"**{category}**"
        lines.append(f"| {category_md} | {complete} | {pct} | {status} |")

    return "\n".join(lines)


def update_readme(analysis):
    """Update README.md with accurate statistics."""
    readme_path = Path(__file__).parent.parent / 'README.md'

    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    total_impl = analysis['numpyts_total']
    total_numpy = analysis['numpy_total']
    coverage = analysis['overall_coverage']

    # Update the intro paragraph
    old_pattern = r'A faithful NumPy.*?more\.'
    new_intro = (
        f"A faithful NumPy 2.0+ implementation for TypeScript/JavaScript, "
        f"validated against Python NumPy. **{total_impl} of {total_numpy} "
        f"NumPy functions ({coverage:.1f}% complete)** covering array "
        f"creation, manipulation, linear algebra, reductions, and more."
    )
    content = re.sub(old_pattern, new_intro, content, flags=re.DOTALL)

    # Update the "Why numpy-ts?" section
    old_pattern = r'- \*\*📊 Growing API\*\* — .*'
    new_bullet = (
        f"- **📊 Growing API** — {total_impl}/{total_numpy} NumPy "
        f"functions ({coverage:.1f}% complete) with 1365+ tests "
        f"validated against Python NumPy"
    )
    content = re.sub(old_pattern, new_bullet, content)

    # Update the API Coverage table
    table = generate_readme_table(analysis['category_stats'])
    overall_line = (
        f"\n**Overall: {total_impl}/{total_numpy} "
        f"functions ({coverage:.1f}% complete)**"
    )

    # Find and replace the table
    # Use a more specific pattern to avoid exponential backtracking
    table_pattern = (
        r'### API Coverage\n\n'
        r'Progress toward complete NumPy API compatibility:\n\n'
        r'(?:\|[^\n]*\n)+'  # Match table rows (non-capturing, possessive)
        r'\n?'  # Optional blank line before Overall
        r'\*\*Overall:[^\n]*\n'
    )
    new_table_section = (
        f"### API Coverage\n\n"
        f"Progress toward complete NumPy API compatibility:\n\n"
        f"{table}\n{overall_line}\n"
    )

    content = re.sub(table_pattern, new_table_section, content,
                     flags=re.DOTALL)

    # Update architecture diagram
    old_pattern = r'│  NumPy-Compatible API \(.*?\)   │'
    new_arch = f"│  NumPy-Compatible API ({total_impl}/{total_numpy})   │"
    content = re.sub(old_pattern, new_arch, content)

    # Update comparison table
    old_pattern = r'\| NumPy API Coverage \| .*? \|'
    new_comp = (
        f"| NumPy API Coverage | "
        f"{total_impl}/{total_numpy} ({coverage:.0f}%) |"
    )
    content = re.sub(old_pattern, new_comp, content)

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✅ Updated {readme_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare numpy-ts vs NumPy API and update documentation'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed list of missing and extra functions'
    )
    args = parser.parse_args()

    print("Comparing numpy-ts vs NumPy and updating documentation...\n")

    # Run audits first to get fresh data
    run_audits()

    # Analyze coverage
    analysis = analyze_coverage(verbose=args.verbose)

    # Update README
    update_readme(analysis)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal NumPy functions: {analysis['numpy_total']}")
    print(f"Total numpy-ts functions: {analysis['numpyts_total']}")
    print(f"Overall coverage: {analysis['overall_coverage']:.1f}%")

    print(f"\nTop-level functions: "
          f"{analysis['toplevel_coverage']:.1f}% coverage")
    print(f"NDArray methods: {analysis['methods_coverage']:.1f}% coverage")

    print("\nTop completing categories:")
    sorted_cats = sorted(
        analysis['category_stats'].items(),
        key=lambda x: -x[1]['percentage']
    )
    for cat, stats in sorted_cats[:5]:
        print(f"  {cat:30s} {stats['percentage']:5.1f}%")

    print("\nNext priorities (50-90% complete):")
    for cat, stats in sorted_cats:
        if 50 <= stats['percentage'] < 90:
            missing_list = ', '.join(sorted(stats['missing'])[:3])
            pct_str = f"{stats['percentage']:.0f}%"
            print(f"  {cat:30s} ({pct_str}) - Need: {missing_list}")

    print("\n" + "=" * 70)
    if args.verbose:
        print("Done! (use -v flag to see detailed gaps)")
    else:
        print("Done! (use -v flag for detailed missing/extra functions list)")


if __name__ == '__main__':
    main()
