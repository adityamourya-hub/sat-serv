"""
Manual fix: For each def/class at 1-space, add 1 space to its body lines.
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
                result.append(line)  # Keep opener at 1-space
                i += 1
                # Process body - add 1 space to body lines
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
                    
                    if body_spaces == 1:
                        # Body at 1-space -> add 1 space (make it 2-space)
                        result.append(' ' + body_line)
                        i += 1
                    elif body_spaces == 2:
                        # Already at 2-space - check for nested opener
                        nested_is_opener = (body_stripped.startswith('def ') or body_stripped.startswith('class ') or
                                           body_stripped.startswith('if ') or body_stripped.startswith('for ') or
                                           body_stripped.startswith('while ') or body_stripped.startswith('with ') or
                                           body_stripped.startswith('try:') or body_stripped.startswith('else:') or
                                           body_stripped.startswith('elif ') or body_stripped.startswith('except ') or
                                           body_stripped.startswith('finally:'))
                        
                        if nested_is_opener and body_stripped.endswith(':'):
                            result.append(body_line)
                            i += 1
                            # Process nested body - add 1 space (2 -> 3)
                            while i < len(lines):
                                nested_line = lines[i]
                                nested_spaces = len(nested_line) - len(nested_line.lstrip())
                                nested_stripped = nested_line.strip()
                                
                                if not nested_stripped:
                                    result.append(nested_line)
                                    i += 1
                                    continue
                                
                                if nested_spaces < 2:
                                    break
                                
                                if nested_spaces == 2:
                                    result.append(' ' + nested_line)  # 2 -> 3
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
        else:
            result.append(line)
            i += 1
    
    return ''.join(result)

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
