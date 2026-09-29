# -*- coding: utf-8 -*-
import numpy as np, pickle, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gridworld import N_STATES, TERMINALS, is_terminal

plt.rcParams['font.family'] = ['Carlito', 'DejaVu Sans']
plt.rcParams['font.size'] = 13
C_TWO, C_IN = "#2E3FA3", "#0B5E6A"
res = pickle.load(open("results.pkl", "rb"))

import os
os.makedirs("figures", exist_ok=True)

# ---------- Fig 1: layout gridworld ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=200)
titles = ["Penomoran state", "Kebijakan acak\n(random policy)", "Kebijakan selalu-atas\n(always-up policy)"]
for ax, title in zip(axes, titles):
    ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_aspect('equal')
    for i in range(5): ax.plot([0,4],[i,i], color="#C4CCD8", lw=1); ax.plot([i,i],[0,4], color="#C4CCD8", lw=1)
    for s in range(16):
        row, col = divmod(s, 4)
        x, y = col + 0.5, 3 - row + 0.5
        if s in TERMINALS:
            ax.add_patch(mpatches.Rectangle((col, 3-row), 1, 1, color="#FFE8A3"))
        ax.text(x, y, str(s), ha='center', va='center', fontsize=12,
                 color="#5B6776" if s in TERMINALS else "#14202B", fontweight='bold' if s in TERMINALS else 'normal')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=12.5)
    for spine in ax.spines.values(): spine.set_visible(False)

def draw_arrows(ax, states, dirs):
    for s in states:
        row, col = divmod(s, 4)
        x, y = col + 0.5, 3 - row + 0.5
        for d in dirs:
            dx,dy = {"up":(0,0.28),"down":(0,-0.28),"left":(-0.28,0),"right":(0.28,0)}[d]
            ax.annotate("", xy=(x+dx,y+dy), xytext=(x,y), arrowprops=dict(arrowstyle="->", color=C_TWO, lw=1.3))

nonterm = [s for s in range(16) if s not in TERMINALS]
draw_arrows(axes[1], nonterm, ["up","down","left","right"])
draw_arrows(axes[2], nonterm, ["up"])
fig.tight_layout()
fig.savefig("figures/fig1_layout.png", facecolor="white"); plt.close(fig)

# ---------- Fig 2: baseline convergence curve (random policy, gamma=1) ----------
fig, ax = plt.subplots(figsize=(7,4.6), dpi=200)
r = res["A_baseline_random"]
ax.plot(range(1, r["two_array"]["sweeps"]+1), r["two_array"]["deltas"], color=C_TWO, lw=2, label=f"Two-array ({r['two_array']['sweeps']} sweep)")
ax.plot(range(1, r["in_place"]["sweeps"]+1), r["in_place"]["deltas"], color=C_IN, lw=2, label=f"In-place ({r['in_place']['sweeps']} sweep)")
ax.axhline(1e-4, color="#8A94A3", ls="--", lw=1, label="θ = 1e-4")
ax.set_yscale("log"); ax.set_xlabel("Sweep ke-k"); ax.set_ylabel("Δ = maxₛ |Vₖ(s) − Vₖ₋₁(s)|  (skala log)")
ax.set_title("Konvergensi: random policy, γ=1, V₀=0")
ax.legend(); ax.grid(True, which="both", color="#E3E7ED", lw=0.6)
fig.tight_layout(); fig.savefig("figures/fig2_baseline.png", facecolor="white"); plt.close(fig)

# ---------- Fig 3: effect of initial value ----------
fig, axs = plt.subplots(1, 2, figsize=(13,4.6), dpi=200, sharey=True)
labels = {"B_initval_zero":"V₀=0","B_initval_optimistic+100":"V₀=+100","B_initval_pessimistic-100":"V₀=−100","B_initval_random":"V₀~U(−20,20)"}
colors = ["#0B5E6A","#2E3FA3","#C0562A","#8A4B00"]
for ax, method, title in zip(axs, ["two_array","in_place"], ["Two-array","In-place"]):
    for (name,label), c in zip(labels.items(), colors):
        d = res[name][method]["deltas"]
        ax.plot(range(1,len(d)+1), d, color=c, lw=1.8, label=f"{label} ({res[name][method]['sweeps']} sweep)")
    ax.axhline(1e-4, color="#8A94A3", ls="--", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("Sweep ke-k"); ax.set_title(f"{title}, random policy, γ=1")
    ax.grid(True, which="both", color="#E3E7ED", lw=0.6)
axs[0].set_ylabel("Δ (skala log)"); axs[0].legend(fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig3_initval.png", facecolor="white"); plt.close(fig)

# ---------- Fig 4: effect of theta ----------
thetas = [1e-1,1e-2,1e-3,1e-4,1e-6,1e-8]
fig, ax = plt.subplots(figsize=(7,4.8), dpi=200)
for method,color,marker in [("two_array",C_TWO,'o'),("in_place",C_IN,'s')]:
    sweeps = [res[f"C_theta_{t}"][method]["sweeps"] for t in thetas]
    ax.plot([-np.log10(t) for t in thetas], sweeps, marker=marker, color=color, lw=1.8, ms=6, label=method.replace("_"," "))
ax.set_xlabel("−log₁₀(θ)  (banyak digit presisi)"); ax.set_ylabel("Jumlah sweep sampai konvergen")
ax.set_title("Sweep dibutuhkan vs threshold θ\n(random policy, γ=1, V₀=0)")
ax.grid(True, color="#E3E7ED", lw=0.6); ax.legend()
fig.tight_layout(); fig.savefig("figures/fig4_theta.png", facecolor="white"); plt.close(fig)

# ---------- Fig 5: always-up, gamma=1, non-convergence ----------
# Panel kanan pakai snapshot di sweep ke-15 (bukan 300) supaya angkanya masih
# mudah dibaca (-15 dan seterusnya, bukan -300); ini yang dipakai di laporan.
from ipe import two_array
from gridworld import up_policy

fig, axs = plt.subplots(1,2, figsize=(13,4.6), dpi=200)
r = res["D_up_gamma1"]
axs[0].plot(range(1,len(r["two_array"]["deltas"])+1), r["two_array"]["deltas"], color=C_TWO, lw=2.4, label="Two-array")
axs[0].plot(range(1,len(r["in_place"]["deltas"])+1), r["in_place"]["deltas"], color=C_IN, lw=1.6, ls="--", label="In-place")
axs[0].axhline(1e-4, color="#8A94A3", ls=":", lw=1, label="θ = 1e-4 (tak pernah tercapai)")
axs[0].set_xlabel("Sweep ke-k"); axs[0].set_ylabel("Δ"); axs[0].set_ylim(0,1.15)
axs[0].set_title("Δ per sweep -- always-up, γ=1\n(konstan di 1.0 untuk kedua metode, selamanya)")
axs[0].legend(); axs[0].grid(True, color="#E3E7ED", lw=0.6)

_, _, h15, _, _ = two_array(up_policy, gamma=1.0, theta=1e-4, v0=0.0, max_sweeps=15)
V15 = np.array(h15[15]).reshape(4,4)
axs[1].imshow(np.clip(V15,-16,0), cmap="RdBu", vmin=-16, vmax=0)
for i in range(4):
    for j in range(4):
        s = i*4+j
        txt = "T" if s in TERMINALS else f"{V15[i,j]:.0f}"
        axs[1].text(j,i,txt, ha='center', va='center', fontsize=14,
                     color="white" if V15[i,j]<-8 else "#14202B", fontweight='bold')
axs[1].set_title("Nilai setelah 15 sweep (V\u2080=0)\n11 dari 14 state terus turun tanpa batas")
axs[1].set_xticks([]); axs[1].set_yticks([])
fig.tight_layout(); fig.savefig("figures/fig5_up_gamma1.png", facecolor="white"); plt.close(fig)

# ---------- Fig 6: always-up, gamma=0.9, converges ----------
fig, axs = plt.subplots(1,2, figsize=(13,4.6), dpi=200)
r = res["E_up_gamma0.9"]
axs[0].plot(range(1,r["two_array"]["sweeps"]+1), r["two_array"]["deltas"], color=C_TWO, lw=2, label=f"Two-array ({r['two_array']['sweeps']} sweep)")
axs[0].plot(range(1,r["in_place"]["sweeps"]+1), r["in_place"]["deltas"], color=C_IN, lw=2, ls="--", label=f"In-place ({r['in_place']['sweeps']} sweep)")
axs[0].axhline(1e-4, color="#8A94A3", ls=":", lw=1)
axs[0].set_yscale("log"); axs[0].set_xlabel("Sweep ke-k"); axs[0].set_ylabel("Δ (skala log)")
axs[0].set_title("Konvergensi: always-up, γ=0.9, V₀=0\n(kedua metode: 89 sweep -- lihat analisis)")
axs[0].legend(); axs[0].grid(True, which="both", color="#E3E7ED", lw=0.6)

V = np.array(r["two_array"]["V"]).reshape(4,4)
axs[1].imshow(V, cmap="RdBu", vmin=-10, vmax=0)
for i in range(4):
    for j in range(4):
        s = i*4+j
        txt = "T" if s in TERMINALS else f"{V[i,j]:.2f}"
        axs[1].text(j,i,txt, ha='center', va='center', fontsize=12.5,
                     color="white" if V[i,j]<-6 else "#14202B", fontweight='bold')
axs[1].set_title("v_π akhir, always-up, γ=0.9")
axs[1].set_xticks([]); axs[1].set_yticks([])
fig.tight_layout(); fig.savefig("figures/fig6_up_gamma09.png", facecolor="white"); plt.close(fig)

# ---------- Fig 7: final V grid, random policy (validasi vs Figure 4.1 buku) ----------
fig, ax = plt.subplots(figsize=(4.6,4.6), dpi=200)
V = np.array(res["A_baseline_random"]["two_array"]["V"]).reshape(4,4)
ax.imshow(V, cmap="RdBu", vmin=-22, vmax=0)
for i in range(4):
    for j in range(4):
        s = i*4+j
        txt = "T" if s in TERMINALS else f"{V[i,j]:.0f}"
        ax.text(j,i,txt, ha='center', va='center', fontsize=14,
                 color="white" if V[i,j]<-12 else "#14202B", fontweight='bold')
ax.set_title("v_π akhir -- random policy, γ=1\n(cocok dengan Figure 4.1 buku)")
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig("figures/fig7_random_final.png", facecolor="white"); plt.close(fig)

print("Semua figure tersimpan di figures/.")
for f in sorted(os.listdir('figures')):
    if f.endswith('.png'): print(' ', f, os.path.getsize(f'figures/{f}'), 'bytes')
