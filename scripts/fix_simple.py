"""
Simple fix: multiply 1-space by 4, then fix def/class bodies that are at 4-space (should be at 8-space).
"""

import ast

def fix_file(path):
    with open(path, 'r') as f:
        content = f.read()
    
    lines = content.splitlines()
    
    # Step 1: multiply 1-space by 4
    step1 = []
    for line in lines:
        if line == '' or line[0] != ' ':
            step1.append(line)
        else:
            stripped = line.lstrip(' ')
            n_spaces = len(line) - len(stripped)
            step1.append('    ' * n_spaces + stripped)
    
    # Step 2: fix def/class bodies that are at 4-space (should be at 8-space)
    step2 = []
    i = 0
    while i < len(step1):
        line = step1[i]
        spaces = len(line) - len(line.lstrip())
        stripped = line.strip()
        
        if not stripped:
            step2.append(line)
            i += 1
            continue
        
        # Check for def/class at 4-space (level 1)
        if spaces == 4 and (stripped.startswith('def ') or stripped.startswith('class ')) and stripped.endswith(':'):
            step2.append(line)  # Keep def at 4-space
            i += 1
            # Process body - bump 4-space to 8-space
            while i < len(step1):
                body_line = step1[i]
                body_spaces = len(body_line) - len(body_line.lstrip())
                body_stripped = body_line.strip()
                
                if not body_stripped:
                    step2.append(body_line)
                    i += 1
                    continue
                
                if body_spaces == 0:
                    break  # Top-level
                
                if body_spaces == 4:
                    # Body at 4-space -> bump to 8-space
                    step2.append('    ' + body_line)
                    i += 1
                elif body_spaces == 8:
                    # Already at 8-space - check for nested blocks
                    nested_is_opener = (body_stripped.startswith('def ') or body_stripped.startswith('class ') or
                                       body_stripped.startswith('if ') or body_stripped.startswith('for ') or
                                       body_stripped.startswith('while ') or body_stripped.startswith('with ') or
                                       body_stripped.startswith('try:') or body_stripped.startswith('else:') or
                                       body_stripped.startswith('elif ') or body_stripped.startswith('except ') or
                                       body_stripped.startswith('finally:'))
                    
                    if nested_is_opener and body_stripped.endswith(':'):
                        step2.append(body_line)
                        i += 1
                        # Process nested body - bump 8-space to 12-space
                        while i < len(step1):
                            nested_line = step1[i]
                            nested_spaces = len(nested_line) - len(nested_line.lstrip())
                            nested_stripped = nested_line.strip()
                            
                            if not nested_stripped:
                                step2.append(nested_line)
                                i += 1
                                continue
                            
                            if nested_spaces < 8:
                                break
                            
                            if nested_spaces == 8:
                                step2.append('    ' + nested_line)  # 8 -> 12
                                i += 1
                            else:
                                step2.append(nested_line)
                                i += 1
                    else:
                        step2.append(body_line)
                        i += 1
                else:
                    step2.append(body_line)
                    i += 1
        else:
            step2.append(line)
            i += 1
    
    output = '\n'.join(step2)
    if content.endswith('\n'):
        output += '\n'
    return output

if __name__ == '__main__':
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
