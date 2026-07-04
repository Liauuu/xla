"""Investigation logs for jax-ml/jax#38609 / openxla/xla#44738.

Run:
  python experiments/jax38609_investigation.py > experiments/jax38609_logs.txt 2>&1
"""
import os
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def summarize(name: str, arr) -> None:
    a = np.asarray(arr)
    finite = np.isfinite(a)
    n_finite = int(finite.sum())
    n_total = a.size
    print(f"  {name}:")
    print(f"    shape={a.shape} dtype={a.dtype}")
    print(
        f"    finite={n_finite}/{n_total}  "
        f"nan={int(np.isnan(a).sum())}  inf={int(np.isinf(a).sum())}"
    )
    if n_finite > 0:
        fa = a[finite]
        print(f"    min={fa.min():.6g}  max={fa.max():.6g}  mean={fa.mean():.6g}")
    if a.size <= 8:
        print(f"    values={a.ravel()}")
    else:
        print(f"    first_row={a.reshape(-1, a.shape[-1])[0]}")


def compare(name: str, eager, jit) -> bool:
    e = np.asarray(eager)
    j = np.asarray(jit)
    if e.shape != j.shape:
        print(f"  {name}: SHAPE MISMATCH {e.shape} vs {j.shape}")
        return False
    both_finite = np.isfinite(e) & np.isfinite(j)
    if both_finite.any():
        max_abs_diff = float(np.max(np.abs(e[both_finite] - j[both_finite])))
    else:
        max_abs_diff = float("nan")
    match = bool(np.allclose(e, j, rtol=1e-5, atol=1e-5, equal_nan=True))
    print(f"  {name}: match={match}  max_abs_diff={max_abs_diff:.6g}")
    if not match and both_finite.any():
        diff = np.abs(e - j)
        idx = np.unravel_index(
            int(np.argmax(np.where(both_finite, diff, -1))), e.shape
        )
        print(
            f"    largest finite diff at {idx}: "
            f"eager={e[idx]:.6g}  jit={j[idx]:.6g}  diff={diff[idx]:.6g}"
        )
    return match


def make_reproducer_1_inputs():
    _key = jax.random.PRNGKey(344146)
    _key, _k1 = jax.random.split(_key)
    t = jax.random.normal(_k1, (5,))
    _key, _k2 = jax.random.split(_key)
    K = jax.random.normal(_k2, (5, 5))
    _key, _k3 = jax.random.split(_key)
    V = jax.random.normal(_k3, (5, 5))
    return t, K, V


def reproducer_1_fn_full(t, K, V):
    y = jnp.expm1(jnp.clip(t, -4, 4))
    z = jnp.arctan2(t, t)
    w = jnp.outer(y, z)
    m = jnp.einsum("ij,jk->ik", w, w)
    Q = jax.scipy.linalg.expm(m * 0.1)
    scale = Q.shape[-1] ** -0.5
    logits = jnp.matmul(Q, K.T) * scale
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - logits_max
    exp_shifted = jnp.exp(shifted)
    denom = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    weights = exp_shifted / denom
    out = jnp.matmul(weights, V)
    return {
        "Q": Q,
        "logits": logits,
        "logits_max": logits_max,
        "shifted": shifted,
        "exp_shifted": exp_shifted,
        "denom": denom,
        "weights": weights,
        "out": out,
    }


def reproducer_1_fn(t, K, V):
    return reproducer_1_fn_full(t, K, V)["out"]


def run_reproducer_1():
    banner("Reproducer 1 (jax-ml/jax#38609)")
    t, K, V = make_reproducer_1_inputs()

    eager_dict = reproducer_1_fn_full(t, K, V)
    jit_dict = jax.jit(reproducer_1_fn_full)(t, K, V)

    banner("Eager intermediate values")
    stages = [
        "Q",
        "logits",
        "logits_max",
        "shifted",
        "exp_shifted",
        "denom",
        "weights",
        "out",
    ]
    for k in stages:
        summarize(k, eager_dict[k])

    banner("JIT intermediate values")
    for k in stages:
        summarize(k, jit_dict[k])

    banner("Eager vs JIT step-by-step comparison")
    first_mismatch = None
    for k in stages:
        m = compare(k, eager_dict[k], jit_dict[k])
        if not m and first_mismatch is None:
            first_mismatch = k

    banner("Interpretation")
    print("  Through matrix expm + scaled logits (Q, logits, logits_max):")
    pre_match = all(
        compare(k, eager_dict[k], jit_dict[k]) for k in stages[:3]
    )
    print(f"    all match: {pre_match}")
    print()
    print("  Softmax numerics (shifted, exp, normalize, output):")
    for k in stages[3:]:
        compare(k, eager_dict[k], jit_dict[k])
    print()
    print(f"  first divergence: {first_mismatch}")
    print(
        "  logits_max matches as a standalone tensor, but shifted = logits - max"
    )
    print(
        "  diverges in JIT. NaNs then appear in exp_shifted / weights / output."
    )
    print()
    print(f"  eager out[0] = {np.asarray(eager_dict['out'])[0]}")
    print(f"  jit   out[0] = {np.asarray(jit_dict['out'])[0]}")


def run_reproducer_2():
    banner("Reproducer 2 (jax-ml/jax#38609)")
    _key = jax.random.PRNGKey(919749)
    _key, _k1 = jax.random.split(_key)
    t1 = jax.random.normal(_k1, (8, 3))
    _key, _k2 = jax.random.split(_key)
    t2 = jax.random.normal(_k2, (3, 3))
    _key, _k3 = jax.random.split(_key)
    t3 = jax.random.normal(_k3, (3, 3))

    def f(t1, t2, t3):
        s1 = jax.nn.sparse_plus(t1)
        s2 = jax.scipy.special.digamma(jnp.abs(s1) + 0.1)
        s3 = jnp.rot90(s2)
        s4 = jnp.atleast_1d(jnp.inner(s3, s3))
        Q = jax.scipy.linalg.expm(s4 * 0.1)
        s6 = jnp.nansum(s4, axis=1)
        Q = jnp.maximum(Q, s6)
        scale = Q.shape[-1] ** -0.5
        w = jax.nn.softmax(jnp.matmul(Q, jnp.swapaxes(t2, -2, -1)) * scale, axis=-1)
        return jnp.matmul(w, t3)

    eager = f(t1, t2, t3)
    jit_out = jax.jit(f)(t1, t2, t3)
    print(f"  eager[0] = {np.asarray(eager)[0]}")
    print(f"  jit[0]   = {np.asarray(jit_out)[0]}")
    print(f"  match: {bool(np.allclose(eager, jit_out, equal_nan=True))}")


def run_fusion_disabled_subprocess():
    banner("With CPU fusion pass disabled: --xla_disable_hlo_passes=fusion")
    script = r"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=fusion"

import jax
import jax.numpy as jnp
import numpy as np

_key = jax.random.PRNGKey(344146)
_key, _k1 = jax.random.split(_key)
t = jax.random.normal(_k1, (5,))
_key, _k2 = jax.random.split(_key)
K = jax.random.normal(_k2, (5, 5))
_key, _k3 = jax.random.split(_key)
V = jax.random.normal(_k3, (5, 5))

def f(t, K, V):
    y = jnp.expm1(jnp.clip(t, -4, 4))
    z = jnp.arctan2(t, t)
    w = jnp.outer(y, z)
    m = jnp.einsum("ij,jk->ik", w, w)
    Q = jax.scipy.linalg.expm(m * 0.1)
    scale = Q.shape[-1] ** -0.5
    weights = jax.nn.softmax(jnp.matmul(Q, K.T) * scale, axis=-1)
    return jnp.matmul(weights, V)

eager = f(t, K, V)
jit_out = jax.jit(f)(t, K, V)
print("  XLA_FLAGS:", os.environ["XLA_FLAGS"])
print("  eager[0]:", np.asarray(eager)[0])
print("  jit[0]:  ", np.asarray(jit_out)[0])
print("  eager/jit match:", bool(np.allclose(eager, jit_out, equal_nan=True)))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print("  exit code:", proc.returncode)
        print(proc.stderr[:2000])


def run_hlo_snippet():
    banner("JIT HLO: scaled logits split across separate CPU fusions")
    t, K, V = make_reproducer_1_inputs()
    text = jax.jit(reproducer_1_fn).lower(t, K, V).compile().as_text()
    print("  Entry computation (softmax-related fusions):")
    print()
    for line in text.splitlines():
        if "fusion(" in line and any(
            x in line
            for x in [
                "multiply_reduce_fusion",
                "subtract_exponential_fusion",
                "dot_general.24",
            ]
        ):
            print("  " + line)
    print()
    print("  fused_computation.19 (row-max path recomputes dot*scale inside fusion):")
    start = text.find("%fused_computation.19")
    if start >= 0:
        for line in text[start:].splitlines()[:7]:
            print("  " + line)
    print()
    print("  fused_computation.18 (subtract+exp path recomputes dot*scale again):")
    start = text.find("%fused_computation.18")
    if start >= 0:
        for line in text[start:].splitlines()[:8]:
            print("  " + line)


if __name__ == "__main__":
    print(f"jax version: {jax.__version__}")
    print(f"devices: {jax.devices()}")
    run_reproducer_1()
    run_reproducer_2()
    run_fusion_disabled_subprocess()
    run_hlo_snippet()
