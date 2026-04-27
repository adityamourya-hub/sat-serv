import sys, ast
sys.path.insert(0, 'scripts')
exec(open('scripts/fix_flat_v2.py').read().replace('process_files(broken)', ''))

content = open('ai/v3/koopman_mpc.py').read()
lines_src = content.splitlines()

indent_stack = []; current_level = 0; pending_indent = False; pending_src_spaces = 0
paren_depth = 0; cont_is_block_opener = False; prev_src_spaces = 0
in_multiline = False; multiline_char = None; multiline_level = 0; result = []

for lineno, raw_line in enumerate(lines_src, 1):
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
            in_multiline = True; multiline_char = mchar
            multiline_level = current_level + (1 if pending_indent else 0)
            break
    orig_spaces = len(raw_line) - len(raw_line.lstrip())
    is_indented = orig_spaces > 0
    was_cont = paren_depth > 0
    if was_cont:
        target = current_level
    else:
        if not is_indented and paren_depth == 0:
            current_level = 0; indent_stack.clear(); pending_indent = False; prev_src_spaces = 0
        else:
            if _is_continuer(stripped) and not pending_indent:
                while indent_stack:
                    el, es = indent_stack[-1]
                    if es == orig_spaces:
                        indent_stack.pop(); break
                    elif es > orig_spaces:
                        indent_stack.pop()
                    else:
                        break
                current_level = indent_stack[-1][0] if indent_stack else 0
                prev_src_spaces = orig_spaces
            elif pending_indent:
                indent_stack.append((current_level, pending_src_spaces))
                current_level += 1; pending_indent = False; prev_src_spaces = orig_spaces
        target = current_level
    result.append('    ' * target + stripped)
    if 295 <= lineno <= 375:
        top3 = indent_stack[-3:] if len(indent_stack) >= 3 else indent_stack
        print(f'{lineno:3d} depth={len(indent_stack):2d} lv={current_level:2d} pd={int(pending_indent)} ps={pending_src_spaces} os={orig_spaces} top={top3}: {stripped[:40]}')
    prev_p = paren_depth
    for ch in stripped:
        if ch in '([{': paren_depth += 1
        elif ch in ')]}': paren_depth = max(0, paren_depth - 1)
    if was_cont:
        if paren_depth == 0 and cont_is_block_opener:
            if _code(stripped).endswith(':'):
                pending_indent = True; pending_src_spaces = orig_spaces
            cont_is_block_opener = False
    else:
        if paren_depth == 0:
            if _opens_block(stripped):
                pending_indent = True; pending_src_spaces = orig_spaces
        elif prev_p == 0 and paren_depth > 0:
            cont_is_block_opener = bool(BLOCK_OPENERS.match(stripped) or BLOCK_CONTINUERS.match(stripped))
            if cont_is_block_opener:
                pending_src_spaces = orig_spaces
