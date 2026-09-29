# -*- coding: utf-8 -*-
"""
Gridworld 4x4 (Example 4.1, Sutton & Barto).
Dua sudut (kiri-atas dan kanan-bawah) adalah terminal state (formally satu
state yang sama, ditampilkan di dua tempat). 14 state non-terminal (1..14).
Aksi: up, down, left, right -- deterministik, tetap di tempat jika keluar grid.
Reward -1 di setiap transisi sampai mencapai terminal. Episodic.
"""
import numpy as np

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE          # 16 sel, index 0..15 (row-major)
TERMINALS = {0, GRID_SIZE * GRID_SIZE - 1}  # sel (0,0) dan (3,3)
ACTIONS = ["up", "down", "left", "right"]

def is_terminal(s):
    return s in TERMINALS

def step(s, a):
    """Return (s_next, reward) untuk aksi deterministik a dari state s."""
    row, col = divmod(s, GRID_SIZE)
    if a == "up":
        row = max(row - 1, 0)
    elif a == "down":
        row = min(row + 1, GRID_SIZE - 1)
    elif a == "left":
        col = max(col - 1, 0)
    elif a == "right":
        col = min(col + 1, GRID_SIZE - 1)
    return row * GRID_SIZE + col, -1.0

def random_policy(s):
    """Kebijakan acak berimbang: 4 aksi, masing-masing peluang 0.25."""
    return {a: 0.25 for a in ACTIONS}

def up_policy(s):
    """Kebijakan deterministik: selalu bergerak ke atas."""
    return {"up": 1.0}

def init_values(v0, n=N_STATES, seed=0):
    """v0: angka (nilai awal seragam) atau array/list panjang n, atau 'random'."""
    if isinstance(v0, str) and v0 == "random":
        rng = np.random.default_rng(seed)
        V = rng.uniform(-20, 20, size=n)
    elif np.isscalar(v0):
        V = np.full(n, float(v0))
    else:
        V = np.array(v0, dtype=float).copy()
    for t in TERMINALS:
        V[t] = 0.0
    return V

def expected_update(s, V, policy, gamma):
    """Hitung sisi kanan persamaan Bellman untuk v_pi(s) memakai array V yang diberikan."""
    total = 0.0
    for a, pa in policy(s).items():
        s2, r = step(s, a)
        total += pa * (r + gamma * V[s2])
    return total
