import jax
import jax.numpy as jnp
import numpy as np
import time
from scipy.linalg import schur
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.auditor_layer import RealTimePfaffianChecker

def benchmark_pfaffian(n=32, trials=1000):
    print(f"Benchmarking Pfaffian for {n}x{n} matrix over {trials} trials...")
    
    # Create random antisymmetric matrix
    np.random.seed(42)
    A_base = np.random.randn(n, n)
    A = (A_base - A_base.T)
    
    checker = RealTimePfaffianChecker(use_schur=True)
    
    # Warm up
    _ = checker._pfaffian_schur(A, 1e-10)
    
    start = time.perf_counter_ns()
    for _ in range(trials):
        _ = checker._pfaffian_schur(A, 1e-10)
    end = time.perf_counter_ns()
    
    avg_ns = (end - start) / trials
    print(f"NumPy/SciPy Average Latency: {avg_ns:.2f} ns")

    # JAX Implementation (Optimized Householder/Gaussian)
    from core.auditor_layer import jax_pfaffian_kernel
    
    # Warm up JAX
    A_jax = jnp.array(A)
    _ = jax_pfaffian_kernel(A_jax)
    
    # Benchmark function with static inner loop count
    @jax.jit
    def internal_benchmark(A_mat):
        def loop_fn(carry, _):
            return carry, jax_pfaffian_kernel(A_mat)
        return jax.lax.scan(loop_fn, None, None, length=1000)

    # Warm up internal
    _ = internal_benchmark(A_jax)
    
    # Correctness Check
    pf_jax = jax_pfaffian_kernel(A_jax)
    # The CPU checker expects Nambu basis and extracts Majorana via np.imag
    # So we pass 1j * A
    pf_cpu = checker.compute_pfaffian(1j * A)
    print(f"Correctness Check: JAX Sign={pf_jax}, CPU Sign={pf_cpu.sign}")
    if int(pf_jax) == int(pf_cpu.sign):
        print(">> PASS: JAX kernel matches CPU implementation.")
    else:
        print(">> FAIL: JAX kernel mismatch!")
    
    inner_trials = 10000
    @jax.jit
    def internal_benchmark_deep(A_mat):
        def loop_fn(carry, _):
            return carry, jax_pfaffian_kernel(A_mat)
        return jax.lax.scan(loop_fn, None, None, length=inner_trials)

    start_inner = time.perf_counter_ns()
    _ = internal_benchmark_deep(A_jax)[1].block_until_ready()
    end_inner = time.perf_counter_ns()
    
    avg_ns_jax_internal = (end_inner - start_inner) / inner_trials
    print(f"JAX Kernel Internal Latency: {avg_ns_jax_internal:.2f} ns")
    
    if avg_ns_jax_internal < 100:
        print(">> SUCCESS: Sub-100ns internal latency achieved (Hardware Target Ready)")

if __name__ == "__main__":
    benchmark_pfaffian(n=32)
