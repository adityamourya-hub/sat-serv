import ast

broken = [
    'ai/v3/adaptive_sindy.py', 'ai/v3/check_manifold.py', 'ai/v3/check_sindy_bases.py',
    'ai/v3/conformal_uq.py', 'ai/v3/e3nn_equivariant_hnn.py', 'ai/v3/e3nn_hamiltonian.py',
    'ai/v3/eso_neural_sindy.py', 'ai/v3/hamiltonian_pretraining.py', 'ai/v3/hgv_phno.py',
    'ai/v3/hno_physics_integration.py', 'ai/v3/hno_physics_integration_torch.py',
    'ai/v3/icnn_h_net.py', 'ai/v3/integrity_gate.py', 'ai/v3/koopman_mpc.py',
    'ai/v3/koopman_operator.py', 'ai/v3/lc_phs_trainer.py', 'ai/v3/lie_integrator.py',
    'ai/v3/lie_phs.py', 'ai/v3/lie_utils.py', 'ai/v3/mbpo_hardened.py',
    'ai/v3/mbpo_hardened_test.py', 'ai/v3/onnx_export.py', 'ai/v3/port_hamiltonian.py',
    'ai/v3/rodrigues.py', 'ai/v3/streak_detection.py', 'ai/v3/train_phs_jax.py',
    'ai/v3/train_with_monitoring.py', 'ai/v3/trainer.py', 'ai/v3/unresolved_spectral_convnext.py',
    'ai/v3/wandb_setup.py', 'q/core/auditor.py', 'q/core/fractional_spectral_conv.py',
    'q/core/hno.py', 'q/core/hno_pytorch.py',
]

fixed = 0
still_broken = []

for f in broken:
    with open(f, 'r') as fh:
        content = fh.read()

    lines = content.splitlines()
    new_lines = []
    for line in lines:
        if line == '' or line[0] != ' ':
            new_lines.append(line)
        else:
            stripped = line.lstrip(' ')
            n_spaces = len(line) - len(stripped)
            new_lines.append('    ' * n_spaces + stripped)

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
        still_broken.append((f, e.lineno, str(e)))

print(f"\nFixed: {fixed}, Still broken: {len(still_broken)}")
for f, ln, e in still_broken:
    print(f"  STILL BROKEN: {f}  line {ln}: {e}")
