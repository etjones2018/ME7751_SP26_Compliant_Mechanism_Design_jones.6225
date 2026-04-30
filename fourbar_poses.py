"""
ME 7751: Compliant Mechanism Design
6-Panel Pose Comparison Visualizer

Generates static plot frames comparing the Su (2009) global PRB 3R model, 
the task-optimal PRB 3R model, and the exact continuous BVP beam at six 
specific crank angles (psi = 45, 90, 135, 180, 270, 315).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_bvp

# ── PRB 3R model parameters ──────────────────────────────────────────────────
MODELS = {
    "Su (2009)": {
        "gamma": [0.10, 0.35, 0.40, 0.15],
        "k":     [3.51, 2.99, 2.58],
        "color": "#4C9BE8",
        "lw":    3.0,
    },
    "Task-Optimal": {
        "gamma": [0.10339, 0.55447, 0.22403, 0.11811],
        "k":     [10.67649, 2.11254, 2.30961],
        "color": "#E84C9B",
        "lw":    2.0,
    },
}

# ── Linkage geometry (l = 1 normalized) ──────────────────────────────────────
SQ2 = np.sqrt(2)
B   = np.array([2 - SQ2, 1 / SQ2])
r   = 1 - SQ2 / 2
AQu = -1 / SQ2      # Q→A vector in local u
AQv =  1 / SQ2      # Q→A vector in local v

# ── Helpers ───────────────────────────────────────────────────────────────────
def crank_A(psi):
    """CW rotation by psi from neutral (AB pointing left).
    R_cw(psi) @ (-1,0) = (-cos psi, +sin psi)"""
    return B + r * np.array([-np.cos(psi), np.sin(psi)])

def Q_from(A, th0):
    c, s = np.cos(th0), np.sin(th0)
    return A - np.array([AQu*c - AQv*s, AQu*s + AQv*c])

def ik(gamma, Qx, Qy, th0):
    g0, g1, g2, g3 = gamma
    px  = Qx - g3*np.cos(th0) - g0
    py  = Qy - g3*np.sin(th0)
    arg = (px**2 + py**2 - g1**2 - g2**2) / (2*g1*g2)
    if abs(arg) > 1.0:
        return None
    T2 = np.arccos(np.clip(arg, -1, 1))          # elbow-down
    d  = g1**2 + g2**2 + 2*g1*g2*np.cos(T2)
    if d < 1e-12:
        return None
    T1 = np.arctan2(py*(g1 + g2*np.cos(T2)) - px*g2*np.sin(T2),
                    px*(g1 + g2*np.cos(T2)) + py*g2*np.sin(T2))
    return T1, T2, th0 - T1 - T2

def fk_points(gamma, T1, T2, T3):
    g0, g1, g2, g3 = gamma
    P0 = np.zeros(2)
    P1 = P0 + np.array([g0, 0.0])
    P2 = P1 + g1 * np.array([np.cos(T1),      np.sin(T1)])
    P3 = P2 + g2 * np.array([np.cos(T1+T2),   np.sin(T1+T2)])
    P4 = P3 + g3 * np.array([np.cos(T1+T2+T3), np.sin(T1+T2+T3)])
    return [P0, P1, P2, P3, P4]   # P4 = Q

def solve(phi, gamma, k):
    A = crank_A(phi)
    def energy(th0):
        Q = Q_from(A, th0)
        if np.linalg.norm(Q) > 1.05:
            return 1e10
        sol = ik(gamma, Q[0], Q[1], th0)
        if sol is None:
            return 1e10
        T1, T2, T3 = sol
        return 0.5*(k[0]*T1**2 + k[1]*T2**2 + k[2]*T3**2)

    th_grid = np.linspace(-np.pi*0.9, np.pi*0.9, 720)
    E_grid  = np.array([energy(th) for th in th_grid])
    if not (E_grid < 1e9).any():
        return None
    idx = np.argmin(E_grid)
    lo  = th_grid[max(0,   idx - 3)]
    hi  = th_grid[min(719, idx + 3)]
    res = minimize_scalar(energy, bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-9})
    th0 = res.x
    Q   = Q_from(A, th0)
    sol = ik(gamma, Q[0], Q[1], th0)
    if sol is None:
        return None
    return th0, *sol

# ── Phi values to plot ────────────────────────────────────────────────────────
psi_cases = [
    (np.pi/4,   r"$\psi = \pi/4$   (45°)"),
    (np.pi/2,   r"$\psi = \pi/2$   (90°)"),
    (3*np.pi/4, r"$\psi = 3\pi/4$  (135°)"),
    (np.pi,     r"$\psi = \pi$     (180°)"),
    (3*np.pi/2, r"$\psi = 3\pi/2$  (270°)"),
    (7*np.pi/4, r"$\psi = 7\pi/4$  (315°)"),
]

# ── Pre-compute solutions ─────────────────────────────────────────────────────
print("Solving PRB Models...")
solutions = {}   # (psi, name) -> (th0, T1, T2, T3) or None
for psi, _ in psi_cases:
    for name, mdl in MODELS.items():
        sol = solve(psi, mdl["gamma"], mdl["k"])
        solutions[(psi, name)] = sol
        status = f"th0={np.degrees(sol[0]):+.1f}deg" if sol else "NO SOL"
        print(f"  psi={np.degrees(psi):6.1f}deg  {name:12s}  {status}")

print("Solving Exact BVP Continuous Beams...")
def get_exact_Q(psi):
    def fun(s, Y, p):
        Fx, Fy, th0 = p
        x, y, th, M = Y
        return np.vstack((np.cos(th), np.sin(th), M, Fx*np.sin(th) - Fy*np.cos(th)))

    A = crank_A(psi)
    def bc(ya, yb, p):
        Fx, Fy, th0 = p
        Qx = A[0] - (AQu*np.cos(th0) - AQv*np.sin(th0))
        Qy = A[1] - (AQu*np.sin(th0) + AQv*np.cos(th0))
        return np.array([ya[0], ya[1], ya[2], yb[0]-Qx, yb[1]-Qy, yb[2]-th0, yb[3]-((A[0]-Qx)*Fy-(A[1]-Qy)*Fx)])

    s = np.linspace(0, 1, 30)
    Y = np.zeros((4, s.size)); Y[0] = s; Y[2] = 0.1*s; Y[3] = 0.1
    p = np.array([0.0, 0.5, 0.1])
    res = solve_bvp(fun, bc, s, Y, p=p, tol=1e-3, max_nodes=500)
    if res.success:
        return {
            'X': res.y[0],
            'Y': res.y[1],
            'th0': res.p[2],
            'E': 0.5 * np.trapezoid(res.y[3]**2, res.x)
        }
    return None

exact_solutions = {}
for psi, _ in psi_cases:
    exact_solutions[psi] = get_exact_Q(psi)

print("Pre-computing colored crank path...")
theta_c = np.linspace(0, 2*np.pi, 200)
su_wins_x, su_wins_y = [], []
opt_wins_x, opt_wins_y = [], []
for psi in theta_c:
    ex_sol = get_exact_Q(psi)
    if ex_sol is None: continue
    Q_ex = np.array([ex_sol['X'][-1], ex_sol['Y'][-1]])
    
    sol_su = solve(psi, MODELS["Su (2009)"]["gamma"], MODELS["Su (2009)"]["k"])
    sol_opt = solve(psi, MODELS["Task-Optimal"]["gamma"], MODELS["Task-Optimal"]["k"])
    if not sol_su or not sol_opt: continue
    
    Q_su = np.array(fk_points(MODELS["Su (2009)"]["gamma"], sol_su[1], sol_su[2], sol_su[3])[-1])
    Q_opt = np.array(fk_points(MODELS["Task-Optimal"]["gamma"], sol_opt[1], sol_opt[2], sol_opt[3])[-1])
    
    A_pt = crank_A(psi)
    if np.linalg.norm(Q_su - Q_ex) < np.linalg.norm(Q_opt - Q_ex):
        su_wins_x.append(A_pt[0])
        su_wins_y.append(A_pt[1])
    else:
        opt_wins_x.append(A_pt[0])
        opt_wins_y.append(A_pt[1])

print("Done.")

# ── Generate Individual Pose Plots ────────────────────────────────────────────
BG, PANEL = "#0d0d1a", "#111128"

for psi, title in psi_cases:
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    fig.suptitle(
        f"4-Bar Compliant Mechanism — {title}\n"
        "Su (2009) model vs. Task-Optimal model vs. Exact FEA",
        color="white", fontsize=12, fontweight="bold", y=0.96
    )
    
    ax.set_facecolor(PANEL)
    ax.set_xlim(-0.25, 2.0)
    ax.set_ylim(-0.675, 1.525)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=11, pad=6)
    ax.set_xlabel("x / l", color="#99aabb", fontsize=8)
    ax.set_ylabel("y / l", color="#99aabb", fontsize=8)
    for sp in ax.spines.values():
        sp.set_color("#334")
    ax.tick_params(colors="#99aabb", labelsize=7)
    ax.grid(True, color="#1e2240", linewidth=0.5)

    # crank locus
    if opt_wins_x:
        ax.plot(opt_wins_x, opt_wins_y, ".", color=MODELS["Task-Optimal"]["color"], ms=1.5, zorder=1)
    if su_wins_x:
        ax.plot(su_wins_x, su_wins_y, ".", color=MODELS["Su (2009)"]["color"], ms=1.5, zorder=1)

    # Fixed pivots
    ax.plot(0, 0, "^", color="white", ms=8, zorder=6)
    ax.plot(*B,  "s", color="#ffdd57", ms=8, zorder=6)
    ax.annotate("O", (0, 0), xytext=(-14,-10), textcoords="offset points",
                color="white", fontsize=8)
    ax.annotate("B", B, xytext=(4,-12), textcoords="offset points",
                color="#ffdd57", fontsize=8)

    A = crank_A(psi)

    # Draw exact beam
    ex_sol = exact_solutions[psi]
    Q_ex = None
    th0_ex = None
    if ex_sol is not None:
        ax.plot(ex_sol['X'], ex_sol['Y'], "-", color="#88E84C", lw=2.5, zorder=2, label="Exact FEA")
        Q_ex = np.array([ex_sol['X'][-1], ex_sol['Y'][-1]])
        th0_ex = ex_sol['th0']
        
        info_txt = (
            f"Exact Beam\n"
            f"Q =({Q_ex[0]:.3f},{Q_ex[1]:.3f})\n\n"
            f"θ₀={np.degrees(ex_sol['th0']):+6.1f}°\n"
        )
        ax.text(0.67, 0.98, info_txt, transform=ax.transAxes, color="#88E84C", fontsize=6, va="top", family="monospace")

    info_x_starts = {"Su (2009)": 0.02, "Task-Optimal": 0.35}

    # Draw each model
    for name, mdl in MODELS.items():
        c   = mdl["color"]
        lw  = mdl["lw"]
        sol = solutions[(psi, name)]
        if sol is None:
            continue
        th0, T1, T2, T3 = sol
        pts = fk_points(mdl["gamma"], T1, T2, T3)
        Q   = np.array(pts[-1])

        err_text = ""
        err_th0_text = ""
        if Q_ex is not None:
            err_x = abs(Q[0] - Q_ex[0]) * 100
            err_y = abs(Q[1] - Q_ex[1]) * 100
            err_text = f"er(X:{err_x:.1f}%,Y:{err_y:.1f}%)\n"
            
            if th0_ex != 0:
                err_th0 = abs((th0 - th0_ex)/th0_ex) * 100
                err_th0_text = f"er={err_th0:.1f}%"
            else:
                err_th0_text = "er=N/A"

        info_txt = (
            f"{name.split(chr(10))[0]}\n"
            f"Q =({Q[0]:.3f},{Q[1]:.3f})\n"
            f"{err_text}"
            f"θ₀={np.degrees(th0):+6.1f}°\n"
            f"{err_th0_text}"
        )
        ax.text(info_x_starts.get(name, 0.02), 0.98, info_txt, transform=ax.transAxes, color=c, fontsize=6, va="top", family="monospace")

        # PRB beam (rigid link chain)
        bx = [p[0] for p in pts]
        by = [p[1] for p in pts]
        ax.plot(bx, by, "-o", color=c, lw=lw, ms=4, zorder=4,
                label=name)

        # QA rigid link
        ax.plot([Q[0], A[0]], [Q[1], A[1]], "-", color=c,
                lw=lw*0.8, zorder=3, alpha=0.85)

        # AB crank
        ax.plot([A[0], B[0]], [A[1], B[1]], "-", color=c,
                lw=lw*0.9, zorder=3, alpha=0.85)

        # Tip Q marker
        ax.plot(*Q, "D", color=c, ms=6, zorder=5)
        # A marker
        ax.plot(*A, "o", color=c, ms=6, zorder=5)

    # Label A and Q for last model drawn (shared position of A)
    ax.annotate("A", A, xytext=(5, 4), textcoords="offset points",
                color="#cccccc", fontsize=7)

    # Angle arc on B to show psi (CW sweep from neutral=pi to pi-psi)
    arc_r = r * 0.45
    arc_t = np.linspace(np.pi, np.pi - psi, 60)
    ax.plot(B[0] + arc_r*np.cos(arc_t),
            B[1] + arc_r*np.sin(arc_t),
            color="#ffdd57", lw=1.2, zorder=2, alpha=0.7)

    # ── Shared legend ─────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=mdl["color"], lw=mdl["lw"],
               marker="o", ms=5, label=name)
        for name, mdl in MODELS.items()
    ] + [
        Line2D([0], [0], color="#88E84C", lw=2.5, label="Exact FEA"),
        Line2D([0], [0], color="white",   marker="^", ms=8,
               linestyle="none", label="Fixed pivot O"),
        Line2D([0], [0], color="#ffdd57", marker="s", ms=8,
               linestyle="none", label="Fixed pivot B"),
        Line2D([0], [0], color="#aaaaaa", lw=1.5,
               label="Rigid links (QA, AB)"),
    ]
    leg = fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               facecolor="#1a1a3a", edgecolor="#334", labelcolor="white",
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    for text, handle in zip(leg.get_texts(), legend_elements):
        if handle.get_label() in MODELS or "Exact" in handle.get_label():
            text.set_color(handle.get_color())
            text.set_fontweight("bold")

    plt.tight_layout(rect=[0, 0.14, 1, 0.92])
    fname = rf"C:\Users\ejones\Desktop\ME 7751 Compliant Mechanism Design\fourbar_pose_{int(np.degrees(psi))}.png"
    plt.savefig(fname, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close()
