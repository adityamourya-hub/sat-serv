"""
fix_flat_v3.py - Uses source indentation DELTA as the primary signal.

Since all 1-space-per-level files have uniform 1-space increments,
we can detect dedent by checking if orig_spaces DECREASES between lines.
"""
import ast
import re

BLOCK_OPENERS = re.compile(
    r'^(?:def |class |if |for |while |with |try:|async def |async for |async with )'
)
BLOCK_CONTINUERS = re.compile(r'^(?:else:|elif |except[\s:]|finally:)')


def _code(s):
    return s.split('#')[0].rstrip()


def _opens_block(s):
    c = _code(s)
    return bool(c.endswith(':') and (BLOCK_OPENERS.match(s) or BLOCK_CONTINUERS.match(s)))


def _is_continuer(s):
    return bool(BLOCK_CONTINUERS.match(s))


def fix_file(path):
    content = open(path).read()
    lines = content.splitlines()
    result = []

    indent_stack = []
    current_level = 0
    pending_indent = False
    prev_src_spaces = 0

    paren_depth = 0
    cont_is_block_opener = False

    in_multiline = False
    multiline_char = None
    multiline_level = 0

    for raw_line in lines:
        stripped = raw_line.strip()

        if not stripped:
            result.append('')
            continue

        if in_multiline:
            result.append('    ' * multiline_level + stripped)
            if (multiline_char == '"""' and stripped.count('"""') % 2 == 1) or \
               (multiline_char == "'''" and stripped.count("'''") % 2 == 1):
                in_multiline = False
            continue

        for mchar in ['"""', "'''"]:
            if stripped.count(mchar) == 1:
                in_multiline = True
                multiline_char = mchar
                multiline_level = current_level + (1 if pending_indent else 0)
                break

        orig_spaces = len(raw_line) - len(raw_line.lstrip())
        is_indented = orig_spaces > 0
        was_cont = paren_depth > 0

        if was_cont:
            target = current_level
        else:
            if not is_indented:
                current_level = 0
                indent_stack.clear()
                pending_indent = False
                prev_src_spaces = 0
            else:
                if _is_continuer(stripped) and not pending_indent:
                    # First apply any source-level dedent (handles nested else/except)
                    if orig_spaces < prev_src_spaces:
                        delta = prev_src_spaces - orig_spaces
                        for _ in range(delta):
                            if indent_stack:
                                current_level = indent_stack.pop()
                    # Then pop back to the matching block opener level
                    if indent_stack:
                        current_level = indent_stack[-1]
                    prev_src_spaces = orig_spaces

                elif pending_indent:
                    # entering block body
                    indent_stack.append(current_level)
                    current_level += 1
                    pending_indent = False
                    prev_src_spaces = orig_spaces

                elif orig_spaces < prev_src_spaces:
                    # source DEDENTED: pop one output level per source-space decrease
                    delta = prev_src_spaces - orig_spaces
                    for _ in range(delta):
                        if indent_stack:
                            current_level = indent_stack.pop()
                    prev_src_spaces = orig_spaces

                else:
                    prev_src_spaces = orig_spaces

            target = current_level

        result.append('    ' * target + stripped)

        prev_p = paren_depth
        for ch in stripped:
            if ch in '([{':
                paren_depth += 1
            elif ch in ')]}':
                paren_depth = max(0, paren_depth - 1)

        if was_cont:
            if paren_depth == 0 and cont_is_block_opener:
                if _code(stripped).endswith(':'):
                    pending_indent = True
                cont_is_block_opener = False
        else:
            if paren_depth == 0:
                if _opens_block(stripped):
                    pending_indent = True
            elif prev_p == 0 and paren_depth > 0:
                cont_is_block_opener = bool(
                    BLOCK_OPENERS.match(stripped) or BLOCK_CONTINUERS.match(stripped)
                )

    out = '\n'.join(result)
    if content.endswith('\n'):
        out += '\n'
    return out


if __name__ == '__main__':
    files = [
        'ai/v3/adaptive_sindy.py', 'ai/v3/check_manifold.py',
        'ai/v3/koopman_mpc.py', 'ai/v3/wandb_setup.py',
    ]
    fixed = 0
    for f in files:
        r = fix_file(f)
        try:
            ast.parse(r)
            open(f, 'w').write(r)
            print(f'FIXED: {f}')
            fixed += 1
        except SyntaxError as e:
            ls = r.splitlines()
            print(f'FAIL {f}:{e.lineno}: {e.msg}')
            for i, l in enumerate(ls[max(0, e.lineno - 4):e.lineno + 2], max(1, e.lineno - 3)):
                print(f'  {i}: {repr(l[:90])}')
    print(f'\nFixed: {fixed}/4')
