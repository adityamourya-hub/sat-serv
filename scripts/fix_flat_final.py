"""
fix_flat_final.py - Two-pass fixer for the 4 remaining flat-indentation files.

Pass 1: Run fix_flat_v2.fix_file() to get best-effort output.
Pass 2: Fix over-indented else/elif/except/finally by scanning backward to
        find the matching if/for/try/with at the correct level.
"""
import ast
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from fix_flat_v2 import fix_file as fix_pass1

CONTINUERS = re.compile(r'^\s*(else:|elif |except[\s:]|finally:)')
OPENERS = re.compile(r'^\s*(if |elif |for |while |with |try:)')


def _indent_level(line):
    return (len(line) - len(line.lstrip())) // 4


def fix_continuer_levels(code: str) -> str:
    """
    For every else/elif/except/finally that is OVER-indented relative to its
    matching opener, reduce its indentation to match.

    In valid Python:
        if ...:          # level N
            body         # level N+1
        else:            # level N  (same as if)
            body         # level N+1

    Strategy: scan backward from the continuer to find the nearest non-empty
    line at a LOWER level that is a valid opener (if/elif/for/try/with).
    The continuer should be placed at that opener's level.
    Only modify if the continuer is MORE indented than the opener.
    """
    for _ in range(20):  # iterate until stable
        lines = code.splitlines()
        changed = False
        new_lines = list(lines)

        for i, line in enumerate(lines):
            if not CONTINUERS.match(line):
                continue
            lvl = _indent_level(line)
            if lvl == 0:
                continue

            # The body above this continuer is at level `lvl` (or deeper).
            # The matching opener (if/try/for) should be at level `lvl - 1`.
            # Scan backward: skip lines at level >= lvl, stop at first line
            # at level == lvl - 1 that is an opener/continuer.
            # If the first shallower line is at level < lvl - 1, the continuer
            # is over-indented and should be placed at that line's level.
            target_lvl = None
            for j in range(i - 1, -1, -1):
                prev = lines[j]
                if not prev.strip():
                    continue
                prev_lvl = _indent_level(prev)
                if prev_lvl >= lvl:
                    continue  # still inside a deeper/same block, keep scanning

                # First line shallower than the continuer
                if prev_lvl == lvl - 1 and (OPENERS.match(prev) or CONTINUERS.match(prev)):
                    # Already correct — opener is exactly one level up
                    target_lvl = prev_lvl
                elif prev_lvl < lvl - 1:
                    # Over-indented — should match this shallower opener
                    target_lvl = prev_lvl
                else:
                    # prev_lvl == lvl - 1 but not an opener (e.g. assignment)
                    # The continuer is already correct relative to this block
                    target_lvl = prev_lvl
                break

            if target_lvl is not None and target_lvl < lvl:
                target_spaces = '    ' * target_lvl
                new_line = target_spaces + line.lstrip()
                if new_line != line:
                    new_lines[i] = new_line
                    changed = True

        code = '\n'.join(new_lines)
        if not changed:
            break

    return code


def fix_file_final(path: str) -> str:
    # Pass 1: structural fixer
    code = fix_pass1(path)
    # Pass 2: fix over-indented continuers
    code = fix_continuer_levels(code)
    return code


if __name__ == '__main__':
    files = [
        'ai/v3/adaptive_sindy.py',
        'ai/v3/check_manifold.py',
        'ai/v3/koopman_mpc.py',
        'ai/v3/wandb_setup.py',
    ]
    fixed = 0
    for f in files:
        result = fix_file_final(f)
        try:
            ast.parse(result)
            open(f, 'w').write(result)
            print(f'FIXED: {f}')
            fixed += 1
        except SyntaxError as e:
            ls = result.splitlines()
            print(f'FAIL {f}:{e.lineno}: {e.msg}')
            for i, l in enumerate(ls[max(0, e.lineno - 4):e.lineno + 3], max(1, e.lineno - 3)):
                print(f'  {i}: {repr(l[:90])}')
    print(f'\nFixed: {fixed}/4')
