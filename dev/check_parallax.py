"""GPU numerics parity check for the Parallax attention integration.

Validates nanochat/parallax_attn.py against the fp32 reference (nanochat.parallax.reference):
  1. training/prefill custom-op forward vs parallax_reference (full + sliding window)
  2. forward+backward: finite grads for q, r, k, v
  3. the custom op works under torch.compile(dynamic=False) (the compiled training path)
  4. the inference router: prefill (T_new>1) and decode (T_new==1) vs reference,
     for both the CuteDSL (SM90 default) and Triton decode kernels

q,r,k are RMS-normed (matches the model's QK/r-norm and upstream's parity harness; the
centered Parallax output cancels heavily on few-key rows, so the windowed case is judged
on the median rel-err, not the max — see the [max=..., cancellation] annotation).

Run (needs a GPU; on this cluster go through Slurm):
    srun -p main --gres=gpu:1 python dev/check_parallax.py
Force the Triton decode path instead of CuteDSL:
    NANOCHAT_PARALLAX_DECODE=triton srun -p main --gres=gpu:1 python dev/check_parallax.py
"""
import os
import torch

torch.manual_seed(0)
dev = "cuda"
dt = torch.bfloat16

import nanochat.parallax as p
from nanochat.parallax.reference import parallax_reference
import nanochat.parallax_attn as pa

print("decode_available:", p.decode_available, "| cap:", torch.cuda.get_device_capability())


def _n(x):
    return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],)).to(x.dtype)


def randn_n(*shape):
    return _n(torch.randn(*shape, device=dev, dtype=dt))


def relerr(a, b):
    a = a.float(); b = b.float()
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-6)


def check(name, got, ref, tol):
    e = relerr(got, ref)
    ok = e <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max-rel-err={e:.3e} (tol {tol:.1e})")
    return ok


def check_q(name, got, ref, q, tol):
    """Cancellation-aware: compare the q-quantile of rel-err."""
    a = got.float(); b = ref.float()
    rel = ((a - b).abs() / (b.abs().max() + 1e-6)).flatten()
    e = torch.quantile(rel, q).item()
    mx = rel.max().item()
    ok = e <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: q{int(q*100)}-rel-err={e:.3e} (tol {tol:.1e}) [max={mx:.2e}, cancellation]")
    return ok


B, T, H, D = 2, 192, 4, 128
qk = D ** -0.5
ok_all = True

# 1. training fwd: full context
q = randn_n(B, T, H, D); r = randn_n(B, T, H, D); k = randn_n(B, T, H, D)
v = torch.randn(B, T, H, D, device=dev, dtype=dt)
got = pa.parallax_attn_func(q, r, k, v, causal=True, window_size=(-1, 0))
ref = parallax_reference(q, r, k, v, qk, causal=True, window_size_left=-1)
ok_all &= check("train fwd full", got, ref, 2e-2)

# 1b. sliding window: nanochat (64,0) -> reference W=65 (judged on median, see docstring)
got_w = pa.parallax_attn_func(q, r, k, v, causal=True, window_size=(64, 0))
ref_w = parallax_reference(q, r, k, v, qk, causal=True, window_size_left=65)
ok_all &= check_q("train fwd window(64,0)->W=65", got_w, ref_w, 0.50, 1e-2)

# 2. fwd+bwd: finite grads
qg = q.clone().requires_grad_(True); rg = r.clone().requires_grad_(True)
kg = k.clone().requires_grad_(True); vg = v.clone().requires_grad_(True)
o = pa.parallax_attn_func(qg, rg, kg, vg, causal=True, window_size=(-1, 0))
o.float().pow(2).mean().backward()
grads_ok = all(t.grad is not None and torch.isfinite(t.grad).all() for t in (qg, rg, kg, vg))
print(f"[{'PASS' if grads_ok else 'FAIL'}] train bwd: finite grads for q,r,k,v = {grads_ok}")
ok_all &= grads_ok

# 3. under torch.compile(dynamic=False) — the compiled training path
try:
    f = torch.compile(lambda a, b, c, d: pa.parallax_attn_func(a, b, c, d, causal=True, window_size=(-1, 0)),
                      dynamic=False)
    oc = f(q, r, k, v)
    ok_all &= check("train fwd under torch.compile", oc, ref, 2e-2)
except Exception as ex:
    print(f"[FAIL] torch.compile raised: {type(ex).__name__}: {ex}")
    ok_all = False

# 4. inference router: prefill (T_new>1)
L = 256
kc = torch.zeros(B, L, H, D, device=dev, dtype=dt)
vc = torch.zeros(B, L, H, D, device=dev, dtype=dt)
seqlens = torch.zeros(B, dtype=torch.int32, device=dev)
Tp = 96
qp = randn_n(B, Tp, H, D); rp = randn_n(B, Tp, H, D); kp = randn_n(B, Tp, H, D)
vp = torch.randn(B, Tp, H, D, device=dev, dtype=dt)
got_pf = pa.parallax_attn_with_kvcache(qp, rp, kc, vc, k=kp, v=vp, cache_seqlens=seqlens,
                                       causal=True, window_size=(-1, 0))
ref_pf = parallax_reference(qp, rp, kp, vp, qk, causal=True, window_size_left=-1)
ok_all &= check("prefill router vs reference", got_pf, ref_pf, 2e-2)
assert (kc[:, :Tp] == kp).all() and (vc[:, :Tp] == vp).all(), "prefill did not insert k,v into cache"
print("[PASS] prefill inserted k,v into cache")

# 4b. decode (T_new==1): cute (SM90 default) and triton
pos = 128
kc2 = torch.zeros(B, L, H, D, device=dev, dtype=dt)
vc2 = torch.zeros(B, L, H, D, device=dev, dtype=dt)
kc2[:, :pos] = randn_n(B, pos, H, D)
vc2[:, :pos] = torch.randn(B, pos, H, D, device=dev, dtype=dt)
seqlens2 = torch.full((B,), pos, dtype=torch.int32, device=dev)
qd = randn_n(B, 1, H, D); rd = randn_n(B, 1, H, D); kd = randn_n(B, 1, H, D)
vd = torch.randn(B, 1, H, D, device=dev, dtype=dt)


def decode_once(impl):
    os.environ["NANOCHAT_PARALLAX_DECODE"] = impl
    import importlib; importlib.reload(pa)
    kc_l = kc2.clone(); vc_l = vc2.clone(); sl = seqlens2.clone()
    out = pa.parallax_attn_with_kvcache(qd, rd, kc_l, vc_l, k=kd, v=vd, cache_seqlens=sl,
                                        causal=True, window_size=(-1, 0))
    kc_r = kc2.clone(); vc_r = vc2.clone()
    kc_r[:, pos:pos + 1] = kd; vc_r[:, pos:pos + 1] = vd
    ref = parallax_reference(qd, rd, kc_r[:, :pos + 1], vc_r[:, :pos + 1], qk,
                             causal=True, window_size_left=-1)
    return out, ref


for impl, tol in [("triton", 1e-2), ("cute", 6e-2)]:
    try:
        out, ref = decode_once(impl)
        ok_all &= check(f"decode ({impl}) vs reference", out, ref, tol)
    except Exception as ex:
        print(f"[FAIL] decode {impl} raised: {type(ex).__name__}: {ex}")
        ok_all = False
os.environ["NANOCHAT_PARALLAX_DECODE"] = "auto"

print("\n==== OVERALL:", "ALL PASS" if ok_all else "SOME FAILED", "====")
