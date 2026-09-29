# -*- coding: utf-8 -*-
"""
Iterative Policy Evaluation -- dua implementasi:
  (1) two_array   : Jacobi-style, sinkron. Update iterasi ke-k HANYA memakai
                     nilai V dari iterasi ke-(k-1). Array baru dan lama terpisah.
  (2) in_place    : Gauss-Seidel-style, asinkron dalam satu sweep. Satu array
                     saja; state yang diproses belakangan dalam sweep yang sama
                     bisa memakai nilai state lain yang SUDAH diperbarui pada
                     sweep itu juga (tercampur antar "generasi").
Satu iterasi = satu sweep penuh: seluruh 14 state non-terminal diperbarui
tepat satu kali (urutan default: 1, 2, ..., 14).
"""
import numpy as np
from gridworld import N_STATES, TERMINALS, is_terminal, init_values, expected_update

def two_array(policy, gamma=1.0, theta=1e-4, v0=0.0, max_sweeps=100_000, seed=0):
    V = init_values(v0, seed=seed)
    deltas, history = [], [V.copy()]
    order = [s for s in range(N_STATES) if not is_terminal(s)]
    for k in range(1, max_sweeps + 1):
        V_new = V.copy()
        delta = 0.0
        for s in order:
            v_new = expected_update(s, V, policy, gamma)   # <-- selalu baca dari V (array LAMA)
            delta = max(delta, abs(v_new - V[s]))
            V_new[s] = v_new
        V = V_new
        deltas.append(delta)
        history.append(V.copy())
        if delta < theta:
            return V, deltas, history, k, True
    return V, deltas, history, max_sweeps, False


def in_place(policy, gamma=1.0, theta=1e-4, v0=0.0, max_sweeps=100_000, seed=0, order=None):
    V = init_values(v0, seed=seed)
    deltas, history = [], [V.copy()]
    order = order or [s for s in range(N_STATES) if not is_terminal(s)]
    for k in range(1, max_sweeps + 1):
        delta = 0.0
        for s in order:
            v_old = V[s]
            v_new = expected_update(s, V, policy, gamma)   # <-- baca dari V yang SAMA (sudah sebagian diperbarui)
            V[s] = v_new                                    # <-- langsung ditimpa di tempat
            delta = max(delta, abs(v_new - v_old))
        deltas.append(delta)
        history.append(V.copy())
        if delta < theta:
            return V, deltas, history, k, True
    return V, deltas, history, max_sweeps, False
