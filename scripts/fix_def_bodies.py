import ast

# Fix the specific pattern: def/class at 1-space, body at 1-space -> body at 2-space
# Also handle nested blocks inside those bodies

def fix_file(path):
    with open(path, 'r') as f:
        content = f.read()

    lines = content.splitlines()
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

        # Track whether we're in a block that needs indentation adjustment
        if orig_spaces == 0:
            result.append(line)
            i += 1
            continue

        # Check if this is a def/class at 1-space
        if orig_spaces == 1 and (stripped.startswith('def ') or stripped.startswith('class ')) and stripped.endswith(':'):
            result.append(line)
            i += 1
            # Process the body: bump 1-space lines to 2-space until we hit 0-space or dedent
            while i < len(lines):
                body_line = lines[i]
                body_spaces = len(body_line) - len(body_line.lstrip())
                body_stripped = body_line.strip()

                if not body_stripped:
                    result.append(body_line)
                    i += 1
                    continue

                if body_spaces == 0:
                    break  # top-level, end of this block

                if body_spaces == 1:
                    # Body line at 1-space -> bump to 2-space
                    result.append(' ' + body_line)
                    i += 1
                elif body_spaces == 2:
                    # Already at 2-space (nested block), keep as-is
                    result.append(body_line)
                    i += 1
                else:
                    result.append(body_line)
                    i += 1
        else:
            result.append(line)
            i += 1

    output = '\n'.join(result)
    if content.endswith('\n'):
        output += '\n'
    return output

if __name__ == '__main__':
    files = ['ai/v3/check_manifold.py', 'ai/v3/koopman_mpc.py']
    fixed = 0
    for f in files:
        try:
            new_content = fix_file(f)
            ast.parse(new_content)
            with open(f, 'w') as fh:
                fh.write(new_content)
            fixed += 1
            print(f"FIXED: {f}")
        except SyntaxError as e:
            print(f"BROKEN: {f} line {e.lineno}: {e.msg}")

    print(f"\nFixed: {fixed}/2")
