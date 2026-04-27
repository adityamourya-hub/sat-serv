#!/usr/bin/env python3
"""
Fixes flat indentation in Python files by reconstructing nesting depth.

The broken files have all indented lines at exactly 1 space — the entire
indentation hierarchy is collapsed. This script reconstructs proper 4-space
indentation by tracking Python block structure.
"""

import ast
import re
import sys
from pathlib import Path


def fix_indentation(content: str) -> str:
    """
    Fix flat indentation by reconstructing nesting depth from code structure.
    
    Since all lines are at 1 space, we need to infer the correct nesting level
    based on Python keywords and code patterns.
    """
    lines = content.splitlines()
    fixed_lines = []
    
    # Track nesting: 0 = module level, 1 = class body, 2 = method body, 3+ = nested blocks
    nesting_level = 0
    in_class = False
    in_method = False
    paren_depth = 0  # Track open parens for continuation lines
    
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            fixed_lines.append(line)
            continue
        
        leading_spaces = len(line) - len(stripped)
        
        # Track paren/bracket depth
        paren_depth += stripped.count('(') + stripped.count('[') + stripped.count('{')
        paren_depth -= stripped.count(')') + stripped.count(']') + stripped.count('}')
        
        # Continuation line - preserve as-is
        if paren_depth > 0:
            fixed_lines.append('    ' * nesting_level + stripped)
            continue
        
        # Top-level definitions (0 spaces)
        if leading_spaces == 0:
            if stripped.startswith('class '):
                in_class = True
                in_method = False
                nesting_level = 1
            elif stripped.startswith('def ') and not in_class:
                # Top-level function
                in_class = False
                in_method = True
                nesting_level = 1
            elif stripped.startswith('def ') and in_class:
                # Method inside class
                in_method = True
                nesting_level = 2
            elif stripped.startswith('@'):
                # Decorator - same level as next def/class
                pass
            else:
                # Module-level code
                nesting_level = 0
            fixed_lines.append(line)
            continue
        
        # Everything else is at 1 space - infer correct nesting
        if stripped.startswith('def '):
            # Method definition
            in_method = True
            nesting_level = 2 if in_class else 1
        elif stripped.startswith(('if ', 'elif ', 'else:', 'for ', 'while ', 'with ', 'try:', 'except ', 'finally:')):
            # Nested block keyword
            if in_method:
                nesting_level = 3
            elif in_class:
                nesting_level = 2
            else:
                nesting_level = 1
        elif stripped.startswith('class '):
            # Nested class (rare but possible)
            nesting_level = nesting_level + 1
        elif stripped.startswith('@'):
            # Decorator
            pass
        else:
            # Regular code body
            if in_method:
                nesting_level = 2
            elif in_class:
                nesting_level = 1
            else:
                nesting_level = 0
        
        fixed_lines.append('    ' * nesting_level + stripped)
    
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
        print("Usage: fix_flat_indentation.py <file_or_dir> [file_or_dir ...]")
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
