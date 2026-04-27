#!/usr/bin/env python3
"""
Fixes 1-space indentation to 4-space indentation in Python files.

The broken AI-generated files use 1 space for all indentation.
This script multiplies all indentation by 4.
"""

import ast
import sys
from pathlib import Path


def fix_indentation(content: str) -> str:
    """Multiply all indentation by 4."""
    lines = content.splitlines()
    fixed_lines = []
    
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            # Empty line - keep as-is
            fixed_lines.append(line)
        else:
            # Count leading spaces
            leading_spaces = len(line) - len(stripped)
            # Multiply by 4
            new_indent = leading_spaces * 4
            fixed_lines.append(' ' * new_indent + stripped)
    
    return '\n'.join(fixed_lines)


def fix_file(filepath: Path) -> bool:
    """Fix indentation in a single Python file."""
    print(f"Processing: {filepath}")
    
    # Read original content
    with open(filepath, 'r') as f:
        original_content = f.read()
    
    # Try to parse original (should fail for broken files)
    try:
        ast.parse(original_content)
        print(f"  ✓ Already parses correctly, skipping")
        return True
    except SyntaxError as e:
        print(f"  ✗ Parse error: line {e.lineno} - {e.msg}")
    
    # Attempt fix
    fixed_content = fix_indentation(original_content)
    
    # Validate fix
    try:
        ast.parse(fixed_content)
        print(f"  ✓ Fixed successfully")
    except SyntaxError as e:
        print(f"  ✗ Fix produced invalid Python: line {e.lineno} - {e.msg}")
        print(f"  Writing to .fixed for manual inspection")
        with open(filepath.with_suffix('.py.fixed'), 'w') as f:
            f.write(fixed_content)
        return False
    
    # Write fixed content
    with open(filepath, 'w') as f:
        f.write(fixed_content)
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: fix_1space_indentation.py <file_or_dir> [file_or_dir ...]")
        sys.exit(1)
    
    paths = [Path(p) for p in sys.argv[1:]]
    
    for path in paths:
        if path.is_file():
            fix_file(path)
        elif path.is_dir():
            # Find all .py files
            for py_file in path.rglob('*.py'):
                # Skip __pycache__
                if '__pycache__' in py_file.parts:
                    continue
                fix_file(py_file)
        else:
            print(f"Not found: {path}")


if __name__ == '__main__':
    main()
