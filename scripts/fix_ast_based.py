"""
Fix indentation by using AST to detect where blocks should be indented.
Strategy: Parse with tolerant parsing to find valid blocks, then fix indentation.
"""

import ast
import sys

def fix_with_ast(path):
    """Try to parse with ast to identify block structure, then fix indentation."""
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Count indentation distribution
    from collections import Counter
    counts = Counter(len(l) - len(l.lstrip()) for l in lines if l.strip())
    
    # If most lines are at 1-space, it's a flat file
    if counts.get(1, 0) > len(lines) * 0.5:
        return fix_flat_file(lines)
    else:
        return lines  # Already has some structure, skip

def fix_flat_file(lines):
    """Fix a flat file where everything is at 1-space."""
    result = []
    indent_stack = [0]  # Current indent level in 4-space units
    
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            result.append(raw_line)
            continue
        
        orig_spaces = len(raw_line) - len(raw_line.lstrip())
        
        # Check if this is a block opener
        is_opener = (stripped.startswith('def ') or stripped.startswith('class ') or
                     stripped.startswith('if ') or stripped.startswith('for ') or
                     stripped.startswith('while ') or stripped.startswith('with ') or
                     stripped.startswith('try:') or stripped.startswith('else:') or
                     stripped.startswith('elif ') or stripped.startswith('except ') or
                     stripped.startswith('finally:'))
        
        # Check if this is a block continuer
        is_continuer = (stripped.startswith('else:') or stripped.startswith('elif ') or
                        stripped.startswith('except ') or stripped.startswith('finally:'))
        
        # Check if source dedented (signal to pop stack)
        if orig_spaces == 0:
            # Top level - reset
            indent_stack = [0]
        elif orig_spaces == 1:
            # At method body level - check if we should dedent
            # If previous line was at 0-space, this is a new top-level statement
            pass
        
        # Determine target indentation
        if is_continuer and len(indent_stack) > 1:
            # Dedent to the matching opener level
            indent_stack.pop()
            target = indent_stack[-1]
            # Push back for the new block
            indent_stack.append(target)
        elif is_opener and stripped.endswith(':'):
            # Block opener - body will be indented
            target = indent_stack[-1]
            # Don't push yet - push when we see the first body line
            pending_push = True
        else:
            target = indent_stack[-1]
            pending_push = False
        
        # Actually, let's use a simpler heuristic for flat files:
        # - If line ends with ':' and is a block opener, keep current level
        # - Next non-empty line should be indented
        pass
    
    # Simpler approach: just multiply 1-space by 4, then fix def/class bodies
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        orig_spaces = len(line) - len(line.lstrip())
        stripped = line.strip()
        
        if not stripped:
            result.append(line)
            i += 1
            continue
        
        if orig_spaces == 0:
            result.append(line)
            i += 1
        elif orig_spaces == 1:
            # Multiply by 4
            result.append('    ' * 1 + stripped)
            i += 1
        else:
            result.append(line)
            i += 1
    
    # Now fix def/class bodies that are at wrong level
    return fix_def_bodies_after_multiply(result)

def fix_def_bodies_after_multiply(lines):
    """After ×4 multiplication, fix def/class bodies that are still at wrong level."""
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        spaces = len(line) - len(line.lstrip())
        stripped = line.strip()
        
        if not stripped:
            result.append(line)
            i += 1
            continue
        
        # Check for def/class at 4-space (level 1)
        if spaces == 4 and (stripped.startswith('def ') or stripped.startswith('class ')) and stripped.endswith(':'):
            result.append(line)
            i += 1
            # Process body - should be at 8-space (level 2)
            while i < len(lines):
                body_line = lines[i]
                body_spaces = len(body_line) - len(body_line.lstrip())
                body_stripped = body_line.strip()
                
                if not body_stripped:
                    result.append(body_line)
                    i += 1
                    continue
                
                if body_spaces == 0:
                    break  # Top-level
                
                if body_spaces == 4:
                    # Body at same level as def - bump to 8-space
                    result.append('    ' + body_line)
                    i += 1
                elif body_spaces == 8:
                    # Already at correct level, check for nested blocks
                    # If it's a nested def/class/if/for, its body should be at 12-space
                    nested_is_opener = (body_stripped.startswith('def ') or body_stripped.startswith('class ') or
                                       body_stripped.startswith('if ') or body_stripped.startswith('for ') or
                                       body_stripped.startswith('while ') or body_stripped.startswith('with ') or
                                       body_stripped.startswith('try:') or body_stripped.startswith('else:') or
                                       body_stripped.startswith('elif ') or body_stripped.startswith('except ') or
                                       body_stripped.startswith('finally:'))
                    
                    if nested_is_opener and body_stripped.endswith(':'):
                        result.append(body_line)
                        i += 1
                        # Process nested body - should be at 12-space
                        while i < len(lines):
                            nested_line = lines[i]
                            nested_spaces = len(nested_line) - len(nested_line.lstrip())
                            nested_stripped = nested_line.strip()
                            
                            if not nested_stripped:
                                result.append(nested_line)
                                i += 1
                                continue
                            
                            if nested_spaces < 8:
                                break
                            
                            if nested_spaces == 8:
                                result.append('    ' + nested_line)  # 8 -> 12
                                i += 1
                            else:
                                result.append(nested_line)
                                i += 1
                    else:
                        result.append(body_line)
                        i += 1
                else:
                    result.append(body_line)
                    i += 1
        else:
            result.append(line)
            i += 1
    
    return result

if __name__ == '__main__':
    files = ['ai/v3/check_manifold.py', 'ai/v3/koopman_mpc.py']
    fixed = 0
    for f in files:
        with open(f, 'r') as fh:
            content = fh.read()
        lines = content.splitlines()
        new_lines = fix_with_ast(f)
        new_content = '\n'.join(new_lines)
        if content.endswith('\n'):
            new_content += '\n'
        
        try:
            ast.parse(new_content)
            with open(f, 'w') as fh:
                fh.write(new_content)
            fixed += 1
            print(f"FIXED: {f}")
        except SyntaxError as e:
            print(f"BROKEN: {f} line {e.lineno}: {e.msg}")
    
    print(f"\nFixed: {fixed}/2")
