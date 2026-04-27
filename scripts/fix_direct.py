"""
Direct fix: Find all def/class at 1-space and bump their bodies to 2-space.
Handle nested blocks recursively.
"""

def fix_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
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
            # Check if this is a block opener
            is_opener = (stripped.startswith('def ') or stripped.startswith('class ') or
                         stripped.startswith('if ') or stripped.startswith('for ') or
                         stripped.startswith('while ') or stripped.startswith('with ') or
                         stripped.startswith('try:') or stripped.startswith('else:') or
                         stripped.startswith('elif ') or stripped.startswith('except ') or
                         stripped.startswith('finally:'))
            
            if is_opener and stripped.endswith(':'):
                result.append(line)
                i += 1
                # Process the body
                result.extend(process_block_body(lines, i, base_level=1))
                i = advance_to_next_block(lines, i, base_level=1)
            else:
                result.append(line)
                i += 1
        else:
            result.append(line)
            i += 1
    
    return ''.join(result)

def process_block_body(lines, start_idx, base_level):
    """Process block body, bumping lines at base_level to base_level+1."""
    result = []
    i = start_idx
    
    while i < len(lines):
        line = lines[i]
        spaces = len(line) - len(line.lstrip())
        stripped = line.strip()
        
        if not stripped:
            result.append(line)
            i += 1
            continue
        
        if spaces == 0:
            break  # Top-level
        
        if spaces < base_level:
            break  # Dedented past our block
        
        if spaces == base_level:
            # Body line at base level - bump to base_level+1
            result.append(' ' + line)  # Add one space
            i += 1
        elif spaces == base_level + 1:
            # Already at correct level - check for nested opener
            is_opener = (stripped.startswith('def ') or stripped.startswith('class ') or
                         stripped.startswith('if ') or stripped.startswith('for ') or
                         stripped.startswith('while ') or stripped.startswith('with ') or
                         stripped.startswith('try:') or stripped.startswith('else:') or
                         stripped.startswith('elif ') or stripped.startswith('except ') or
                         stripped.startswith('finally:'))
            
            if is_opener and stripped.endswith(':'):
                result.append(line)
                i += 1
                # Process nested body
                result.extend(process_block_body(lines, i, base_level=base_level+1))
                i = advance_to_next_block(lines, i, base_level=base_level+1)
            else:
                result.append(line)
                i += 1
        else:
            result.append(line)
            i += 1
    
    return result

def advance_to_next_block(lines, start_idx, base_level):
    """Advance index past the current block."""
    i = start_idx
    while i < len(lines):
        line = lines[i]
        spaces = len(line) - len(line.lstrip())
        stripped = line.strip()
        
        if not stripped:
            i += 1
            continue
        
        if spaces < base_level:
            return i  # Dedented
        
        i += 1
    
    return i

if __name__ == '__main__':
    import ast
    files = ['ai/v3/check_manifold.py', 'ai/v3/koopman_mpc.py']
    fixed = 0
    for f in files:
        new_content = fix_file(f)
        try:
            ast.parse(new_content)
            with open(f, 'w') as fh:
                fh.write(new_content)
            fixed += 1
            print(f"FIXED: {f}")
        except SyntaxError as e:
            print(f"BROKEN: {f} line {e.lineno}: {e.msg}")
    
    print(f"\nFixed: {fixed}/2")
