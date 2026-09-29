# -*- coding: utf-8 -*-
import numpy as np, json, pickle
from gridworld import N_STATES, random_policy, up_policy
from ipe import two_array, in_place

results = {}

def run(name, policy, gamma, theta, v0, max_sweeps=2000, seed=0):
    r = {}
    for method_name, fn in [("two_array", two_array), ("in_place", in_place)]:
        V,d,h,k,conv = fn(policy, gamma=gamma, theta=theta, v0=v0, max_sweeps=max_sweeps, seed=seed)
        r[method_name] = dict(V=V.tolist(), deltas=d, sweeps=k, converged=conv)
    results[name] = r
    return r

# ---- A. Baseline: random policy, gamma=1, V0=0, theta=1e-4 ----
run("A_baseline_random", random_policy, gamma=1.0, theta=1e-4, v0=0.0)

# ---- B. Effect of initial value (random policy, gamma=1, theta=1e-4) ----
for tag, v0 in [("zero", 0.0), ("optimistic+100", 100.0), ("pessimistic-100", -100.0), ("random", "random")]:
    run(f"B_initval_{tag}", random_policy, gamma=1.0, theta=1e-4, v0=v0)

# ---- C. Effect of threshold theta (random policy, gamma=1, V0=0) ----
for theta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
    run(f"C_theta_{theta}", random_policy, gamma=1.0, theta=theta, v0=0.0)

# ---- D. Always-up policy, gamma=1 (uji non-konvergensi) ----
run("D_up_gamma1", up_policy, gamma=1.0, theta=1e-4, v0=0.0, max_sweeps=300)

# ---- E. Always-up policy, gamma=0.9 (versi diskon, well-posed) ----
run("E_up_gamma0.9", up_policy, gamma=0.9, theta=1e-4, v0=0.0)

# ---- E2. Effect of initial value, always-up gamma=0.9 ----
for tag, v0 in [("zero", 0.0), ("optimistic+100", 100.0), ("pessimistic-100", -100.0)]:
    run(f"E2_up09_initval_{tag}", up_policy, gamma=0.9, theta=1e-4, v0=v0)

# ---- E3. Effect of threshold, always-up gamma=0.9 ----
for theta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
    run(f"E3_up09_theta_{theta}", up_policy, gamma=0.9, theta=theta, v0=0.0)

# ---- F. Sweep order effect, in-place only (random policy, gamma=1) ----
from gridworld import is_terminal
forward = [s for s in range(N_STATES) if not is_terminal(s)]
backward = list(reversed(forward))
for tag, order in [("forward_1to14", forward), ("backward_14to1", backward)]:
    V,d,h,k,conv = in_place(random_policy, gamma=1.0, theta=1e-4, v0=0.0, order=order)
    results.setdefault("F_sweeporder", {})[tag] = dict(V=V.tolist(), deltas=d, sweeps=k, converged=conv)

with open("results.pkl", "wb") as f:
    pickle.dump(results, f)

# Ringkasan cepat di terminal
print(f"{'Eksperimen':35s} {'two_array':>12s} {'in_place':>12s}")
for name in results:
    if name == "F_sweeporder": continue
    r = results[name]
    print(f"{name:35s} {r['two_array']['sweeps']:>7d} ({'Y' if r['two_array']['converged'] else 'N'})  "
          f"{r['in_place']['sweeps']:>7d} ({'Y' if r['in_place']['converged'] else 'N'})")
print()
print("F_sweeporder (in_place, random policy, gamma=1):")
for tag,r in results["F_sweeporder"].items():
    print(f"  {tag:20s} sweeps={r['sweeps']}")
