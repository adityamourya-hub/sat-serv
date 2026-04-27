"""
fix_flat_v2.py - Reconstruct indentation from flat (1-space) Python files.

Strategy:
- All code is at 1-space indent (class bodies, method bodies, nested blocks).
- Track a nesting stack: when we see a colon-ending line (def/class/if/for/while/with/try/except/else/elif/finally),
  the NEXT non-empty line should be indented one level deeper.
- Pop the stack when we see a line that "closes" the block contextually.

Since we can't distinguish levels from spaces alone, we use keyword heuristics
to determine when to push/pop the indentation stack.
"""

import ast
import re
import sys

BLOCK_OPENERS = re.compile(
    r'^(?:def |class |if |for |while |with |try:|async def |async for |async with )'
)
# These dedent to the level of their matching opener, then open a new block
BLOCK_CONTINUERS = re.compile(r'^(?:else:|elif |except[\s:]|finally:)')


def _code(stripped: str) -> str:
    """Strip inline comment."""
    return stripped.split('#')[0].rstrip()


def _opens_block(stripped: str) -> bool:
    code = _code(stripped)
    return bool(code.endswith(':') and (BLOCK_OPENERS.match(stripped) or BLOCK_CONTINUERS.match(stripped)))


def _is_continuer(stripped: str) -> bool:
    return bool(BLOCK_CONTINUERS.match(stripped))


def fix_file(path: str) -> str:
    with open(path, 'r') as f:
        content = f.read()

    lines = content.splitlines()
    result = []

    # indent_stack: list of indent levels for each open block
    # Each entry = the indent level AT which the block opener was placed.
    # When we enter a block, we push current level; body is at level+1.
    # When we see else/elif/except/finally, we pop back to the opener level.
    # Each stack entry: (output_level, src_spaces_of_opener)
    indent_stack = []
    current_level = 0
    pending_indent = False  # next non-continuation non-empty line enters a new block
    pending_src_spaces = 0  # src_spaces of the line that set pending_indent

    paren_depth = 0
    cont_is_block_opener = False
    prev_src_spaces = 0

    in_multiline = False
    multiline_char = None
    multiline_level = 0

    for raw_line in lines:
        stripped = raw_line.strip()

        if not stripped:
            result.append('')
            continue

        # ── multiline string passthrough ──────────────────────────────────
        if in_multiline:
            result.append('    ' * multiline_level + stripped)
            if (multiline_char == '"""' and stripped.count('"""') % 2 == 1) or \
               (multiline_char == "'''" and stripped.count("'''") % 2 == 1):
                in_multiline = False
            continue

        # detect start of multiline string (odd number of triple-quote tokens)
        for mchar in ['"""', "'''"]:
            if stripped.count(mchar) == 1:
                in_multiline = True
                multiline_char = mchar
                multiline_level = current_level + (1 if pending_indent else 0)
                break

        # ── determine this line's source-level indentation ────────────────
        orig_spaces = len(raw_line) - len(raw_line.lstrip())
        is_indented = orig_spaces > 0
        was_continuation = paren_depth > 0

        if was_continuation:
            # continuation line: keep at same level as the opening statement
            target = current_level
        else:
            if not is_indented:
                target = current_level
            else:
                if _is_continuer(stripped) and not pending_indent:
                    # Find the stack entry pushed by the opener at this src_spaces.
                    # The opener (if/try/for) that matches this else/except/elif
                    # was at the same src_spaces as the continuer.
                    # Stack entries are (output_level, opener_src_spaces).
                    # Pop entries until we find one whose opener_src_spaces == orig_spaces,
                    # then pop that one too (it was the body entry pushed by the opener).
                    popped = False
                    while indent_stack:
                        entry_level, entry_src = indent_stack[-1]
                        if entry_src == orig_spaces:
                            # This entry was pushed by our matching opener
                            indent_stack.pop()
                            popped = True
                            break
                        elif entry_src > orig_spaces:
                            # Deeper nested entry — pop it and keep looking
                            indent_stack.pop()
                        else:
                            # entry_src < orig_spaces — went too far, stop
                            break
                    current_level = indent_stack[-1][0] if indent_stack else 0
                    prev_src_spaces = orig_spaces
                elif pending_indent:
                    # entering a new block body — store BODY's src_spaces so
                    # that a continuer (else/except) at the OPENER's src_spaces
                    # can pop this entry: continuer is at opener level,
                    # opener_src_spaces = pending_src_spaces, body is here at orig_spaces.
                    # We store pending_src_spaces (opener's depth) so the continuer
                    # at that same depth can find and pop this entry.
                    indent_stack.append((current_level, pending_src_spaces))
                    current_level += 1
                    pending_indent = False
                    prev_src_spaces = orig_spaces
            target = current_level

        result.append('    ' * target + stripped)

        # ── update paren depth ────────────────────────────────────────────
        prev_paren = paren_depth
        for ch in stripped:
            if ch in '([{':
                paren_depth += 1
            elif ch in ')]}':
                paren_depth = max(0, paren_depth - 1)

        # ── reset stack if a continuation at col 0 just closed its parens ──
        # This handles multi-line function signatures whose closing ): is at
        # column 0 — the NEXT line starts a new top-level block body.
        if was_continuation and not is_indented and paren_depth == 0:
            current_level = 0
            indent_stack.clear()
            pending_indent = False
            prev_src_spaces = 0

        # ── decide pending_indent for NEXT line ───────────────────────────
        if was_continuation:
            # closing a multi-line statement
            if paren_depth == 0 and cont_is_block_opener:
                code = _code(stripped)
                if code.endswith(':'):
                    pending_indent = True
                    pending_src_spaces = orig_spaces
                cont_is_block_opener = False
        else:
            if paren_depth == 0:
                if _opens_block(stripped):
                    pending_indent = True
                    pending_src_spaces = orig_spaces
            elif prev_paren == 0 and paren_depth > 0:
                # just opened a multi-line statement
                cont_is_block_opener = bool(BLOCK_OPENERS.match(stripped) or BLOCK_CONTINUERS.match(stripped))
                if cont_is_block_opener:
                    pending_src_spaces = orig_spaces

    output = '\n'.join(result)
    if content.endswith('\n'):
        output += '\n'
    return output


def process_files(files):
    fixed = 0
    still_broken = []

    for f in files:
        # Skip files that already parse correctly
        try:
            ast.parse(open(f).read())
            fixed += 1
            print(f"ALREADY OK: {f}")
            continue
        except SyntaxError:
            pass
        try:
            new_content = fix_file(f)
            ast.parse(new_content)
            with open(f, 'w') as fh:
                fh.write(new_content)
            fixed += 1
            print(f"FIXED: {f}")
        except SyntaxError as e:
            still_broken.append((f, e.lineno, str(e.msg)))

    print(f"\nFixed: {fixed}, Still broken: {len(still_broken)}")
    for f, ln, msg in still_broken:
        print(f"  BROKEN: {f}  line {ln}: {msg}")

    return still_broken


if __name__ == '__main__':
    broken = [
        'ai/v3/adaptive_sindy.py', 'ai/v3/check_manifold.py',
        'ai/v3/koopman_mpc.py', 'ai/v3/wandb_setup.py',
    ]
    process_files(broken)
