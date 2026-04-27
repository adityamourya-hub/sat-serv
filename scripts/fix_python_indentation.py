#!/usr/bin/env python3
"""
Fixes flat indentation in AI-generated Python files.

Root cause: Method/class bodies are at the same indentation as their definitions.
This script uses a line-based heuristic to detect definitions and indent their bodies.
"""

import re
import sys
from pathlib import Path


def fix_indentation(content: str) -> tuple[str, bool]:
    """
    Fix flat indentation by detecting definition lines and indenting following lines.
    
    Strategy:
    1. Detect lines starting with 'def ' or 'class ' followed by a colon
    2. Indent all subsequent lines until we hit:
       - Another 'def ' or 'class ' at the same indentation level (or less)
       - A line that's already properly indented (starts with 4+ spaces)
    
    Returns (fixed_content, success)
    """
    lines = content.splitlines()
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        leading_spaces = len(line) - len(stripped)
        
        # Detect definition lines at column 0 (no leading spaces)
        if leading_spaces == 0 and re.match(r'^(def\s+\w+\s*\(|class\s+\w+\s*[\(:])', stripped):
            # This is a definition line - add it as-is
            fixed_lines.append(line)
            
            # Now indent all following lines until we hit the next definition
            i += 1
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.lstrip()
                next_leading = len(next_line) - len(next_stripped)
                
                # Stop conditions:
                # 1. Next definition at same or lower indentation
                # 2. Line is already indented (4+ spaces) - might be proper nested structure
                if next_leading == 0 and re.match(r'^(def\s+\w+\s*\(|class\s+\w+\s*[\(:])', next_stripped):
                    # Next definition at column 0 - stop indenting
                    break
                
                # If line has no leading spaces, indent it
                if next_leading == 0 and next_line.strip():
                    fixed_lines.append('    ' + next_line)
                else:
                    # Already indented or empty line - keep as-is
                    fixed_lines.append(next_line)
                
                i += 1
        else:
            # Not a definition line - keep as-is
            fixed_lines.append(line)
            i += 1
    
    return '\n'.join(fixed_lines), True


def fix_file(filepath: Path) -> bool:
    """Fix indentation in a single Python file."""
    print(f"Processing: {filepath}")
    
    # Read original content
    with open(filepath, 'r') as f:
        original_content = f.read()
    
    # Try to parse original (should fail for broken files)
    try:
        import ast
        ast.parse(original_content)
        print(f"  ✓ Already parses correctly, skipping")
        return True
    except SyntaxError as e:
        print(f"  ✗ Parse error: line {e.lineno} - {e.msg}")
    
    # Attempt fix
    fixed_content, success = fix_indentation(original_content)
    
    if not success:
        print(f"  ✗ Fix heuristic failed")
        return False
    
    # Validate fix
    try:
        import ast
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
        print("Usage: fix_python_indentation.py <file_or_dir> [file_or_dir ...]")
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
