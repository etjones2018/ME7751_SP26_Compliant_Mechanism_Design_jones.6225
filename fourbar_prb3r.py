"""
ME 7751: Compliant Mechanism Design
Interactive Forward Kinematics GUI

This is the core interactive application that simultaneously simulates the exact 
continuous BVP beam alongside the rigid link chains of both the Su (2009) global 
PRB model and the newly proposed task-optimal PRB model.

4-Bar Compliant Mechanism Simulation
Linkage geometry (normalized, l=1):
  O = (0, 0)                    fixed beam root
  B = (2-sqrt2, 1/sqrt2)        fixed crank pivot
  r = 1 - sqrt2/2               crank length |AB|
  AQ rigid in local frame:
      A_u/l = -1/sqrt2,  A_v/l = 1/sqrt2   =>  |AQ| = 1

Equilibrium method:
  For each phi, find beam tip slope theta0 that minimises total
  spring strain energy  E = 0.5 * sum(k_i * Theta_i^2)
  subject to the kinematic closure constraint.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_bvp
from matplotlib.widgets import Slider, Button, CheckButtons

# ── PRB 3R model parameters ──────────────────────────────────────────────────
MODELS = {
    "Su (2009)\nγ=[0.10,0.35,0.40,0.15]": {
        "gamma": [0.10, 0.35, 0.40, 0.15],
        "k":     [3.51, 2.99, 2.58],
        "color": "#4C9BE8",
        "exact": False,
    },
    "Task-Optimal\n(4-Bar Specific)": {
        "gamma": [0.10339, 0.55447, 0.22403, 0.11811],
        "k":     [10.67649, 2.11254, 2.30961],
        "color": "#E84C9B",
        "exact": False,
    },
    "Exact Beam\n(Continuous BVP)": {
        "color": "#88E84C",
        "exact": True,
    }
}

# ── Linkage constants ─────────────────────────────────────────────────────────
SQ2 = np.sqrt(2)
B   = np.array([2 - SQ2,  1 / SQ2])
r   = 1 - SQ2 / 2
AQu = -1 / SQ2          # rigid link Q→A in local u
AQv =  1 / SQ2          # rigid link Q→A in local v

# ── Kinematic helpers ─────────────────────────────────────────────────────────
def crank_A(psi):
    """CW rotation by psi from neutral (AB pointing left).
    R_cw(psi) @ (-1,0) = (-cos psi, +sin psi)"""
    return B + r * np.array([-np.cos(psi), np.sin(psi)])


def Q_from(A, th0):
    """
    Q = A - R(th0) @ [AQu, AQv]
    because  A = Q + R(th0) @ [AQu, AQv]
    """
    c, s = np.cos(th0), np.sin(th0)
    return A - np.array([AQu*c - AQv*s,
                          AQu*s + AQv*c])


def ik(gamma, Qx, Qy, th0, elbow='down'):
    """PRB 3R inverse kinematics.  Returns (T1,T2,T3) or None."""
    g0, g1, g2, g3 = gamma
    px = Qx - g3*np.cos(th0) - g0
    py = Qy - g3*np.sin(th0)
    arg = (px**2 + py**2 - g1**2 - g2**2) / (2*g1*g2)
    if abs(arg) > 1.0:
        return None
    T2 = np.arccos(np.clip(arg, -1, 1)) * (1 if elbow == 'down' else -1)
    d  = g1**2 + g2**2 + 2*g1*g2*np.cos(T2)
    if d < 1e-12:
        return None
    T1 = np.arctan2(py*(g1 + g2*np.cos(T2)) - px*g2*np.sin(T2),
                    px*(g1 + g2*np.cos(T2)) + py*g2*np.sin(T2))
    return T1, T2, th0 - T1 - T2


def fk_points(gamma, T1, T2, T3):
    """All joint positions: [O, P1, P2, P3, Q]."""
    g0, g1, g2, g3 = gamma
    P0 = np.zeros(2)
    P1 = P0 + np.array([g0, 0.0])
    P2 = P1 + g1 * np.array([np.cos(T1),       np.sin(T1)])
    P3 = P2 + g2 * np.array([np.cos(T1+T2),     np.sin(T1+T2)])
    P4 = P3 + g3 * np.array([np.cos(T1+T2+T3),  np.sin(T1+T2+T3)])
    return [P0, P1, P2, P3, P4]   # P4 == Q


# ── Equilibrium solver ────────────────────────────────────────────────────────
def solve(phi, gamma, k):
    """
    Find theta0 that minimises spring energy E = 0.5*sum(k_i*Ti^2)
    subject to the geometric constraint.
    Returns (th0, T1, T2, T3) or None.
    """
    A = crank_A(phi)

    def energy(th0):
        Q = Q_from(A, th0)
        if np.linalg.norm(Q) > 1.05:   # outside PRB workspace
            return 1e10
        sol = ik(gamma, Q[0], Q[1], th0)
        if sol is None:
            return 1e10
        T1, T2, T3 = sol
        return 0.5*(k[0]*T1**2 + k[1]*T2**2 + k[2]*T3**2)

    # Coarse sweep to find global minimum basin
    th_grid = np.linspace(-np.pi*0.9, np.pi*0.9, 720)
    E_grid  = np.array([energy(th) for th in th_grid])
    valid   = E_grid < 1e9
    if not valid.any():
        return None

    idx = np.argmin(E_grid)
    lo  = th_grid[max(0,   idx - 3)]
    hi  = th_grid[min(len(th_grid)-1, idx + 3)]
    res = minimize_scalar(energy, bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-9})
    th0 = res.x
    Q   = Q_from(A, th0)
    sol = ik(gamma, Q[0], Q[1], th0)
    if sol is None:
        return None
    return (th0, *sol)

# ── Exact Beam BVP Solver ─────────────────────────────────────────────────────
def exact_beam_bvp_solver(psis):
    def fun(s, Y, p):
        Fx, Fy, th0 = p
        x, y, th, M = Y
        return np.vstack((np.cos(th), np.sin(th), M, Fx*np.sin(th) - Fy*np.cos(th)))

    def build_bc(psi):
        A = crank_A(psi)
        def bc(ya, yb, p):
            Fx, Fy, th0 = p
            Qx = A[0] - (AQu*np.cos(th0) - AQv*np.sin(th0))
            Qy = A[1] - (AQu*np.sin(th0) + AQv*np.cos(th0))
            return np.array([
                ya[0], ya[1], ya[2],
                yb[0] - Qx, yb[1] - Qy, yb[2] - th0,
                yb[3] - ((A[0] - Qx)*Fy - (A[1] - Qy)*Fx)
            ])
        return bc

    exact_frames = []
    s = np.linspace(0, 1, 30)
    Y = np.zeros((4, s.size))
    Y[0] = s; Y[2] = 0.1*s; Y[3] = 0.1
    p = np.array([0.0, 0.5, 0.1])
    
    for psi in psis:
        bc = build_bc(psi)
        res = solve_bvp(fun, bc, s, Y, p=p, tol=1e-3, max_nodes=500)
        if res.success:
            Y = res.y
            p = res.p
            s = res.x
            exact_frames.append({'th0': p[2], 'X': Y[0], 'Y': Y[1], 'E': 0.5*np.trapezoid(Y[3]**2, s)})
        else:
            exact_frames.append(None)
    return exact_frames


# ── Pre-compute all frames ────────────────────────────────────────────────────
N      = 360
psis   = np.linspace(0, 2*np.pi, N, endpoint=False)
frames = {name: [] for name in MODELS}

print("Pre-computing frames ...")
for name, mdl in MODELS.items():
    print(f"  {name.split(chr(10))[0]} ...", end="", flush=True)
    if mdl.get("exact"):
        frames[name] = exact_beam_bvp_solver(psis)
    else:
        gamma, k = mdl["gamma"], mdl["k"]
        for psi in psis:
            frames[name].append(solve(psi, gamma, k))
    n_ok = sum(f is not None for f in frames[name])
    print(f"  {n_ok}/{N} valid")
print("Done.")

# ── Figure setup ──────────────────────────────────────────────────────────────
BG, PANEL = "#0d0d1a", "#111128"
fig, ax = plt.subplots(figsize=(11, 8.5), facecolor=BG)
fig.suptitle("4-Bar Compliant Mechanism  —  Overlapped Model Comparison",
             color="white", fontsize=15, fontweight="bold", y=0.96)

pad = 0.1
xlim = (-0.2, 2.1)
ylim = (-1.0, 1.4)

ax.set_facecolor(PANEL)
ax.set_xlim(*xlim);  ax.set_ylim(*ylim)
ax.set_aspect("equal")
ax.set_xlabel("x / l", color="#99aabb");  ax.set_ylabel("y / l", color="#99aabb")
for sp in ax.spines.values(): sp.set_color("#334")
ax.tick_params(colors="#99aabb")
ax.grid(True, color="#1e2240", linewidth=0.5)

# Fixed ground symbols
ax.plot(0,    0,    "^", color="white",   ms=9, zorder=6)
ax.plot(B[0], B[1], "s", color="#ffdd57", ms=9, zorder=6)
ax.annotate("O", (0, 0),    xytext=(-14,-12), textcoords="offset points", color="white", fontsize=9)
ax.annotate("B", (B[0], B[1]), xytext=(5,-13), textcoords="offset points", color="#ffdd57", fontsize=9)

# B crank circle (dashed)
θc = np.linspace(0, 2*np.pi, 200)
ax.plot(B[0] + r*np.cos(θc), B[1] + r*np.sin(θc), "--", color="#444466", lw=0.8, zorder=1)

# Highlight segments of the crank arc where Su (2009) tracks closer to the exact beam
exact_name = "Exact Beam\n(Continuous BVP)"
su_name = "Su (2009)\nγ=[0.10,0.35,0.40,0.15]"
opt_name = "Task-Optimal\n(4-Bar Specific)"

su_wins_x, su_wins_y = [], []
opt_wins_x, opt_wins_y = [], []
for i, psi in enumerate(psis):
    if frames[exact_name][i] is None or frames[su_name][i] is None or frames[opt_name][i] is None: continue
    Q_ex = np.array([frames[exact_name][i]['X'][-1], frames[exact_name][i]['Y'][-1]])
    
    fr_su = frames[su_name][i]
    Q_su = np.array(fk_points(MODELS[su_name]["gamma"], fr_su[1], fr_su[2], fr_su[3])[-1])
    
    fr_opt = frames[opt_name][i]
    Q_opt = np.array(fk_points(MODELS[opt_name]["gamma"], fr_opt[1], fr_opt[2], fr_opt[3])[-1])
    
    A_pt = crank_A(psi)
    if np.linalg.norm(Q_su - Q_ex) < np.linalg.norm(Q_opt - Q_ex):
        su_wins_x.append(A_pt[0])
        su_wins_y.append(A_pt[1])
    else:
        opt_wins_x.append(A_pt[0])
        opt_wins_y.append(A_pt[1])

if opt_wins_x:
    ax.plot(opt_wins_x, opt_wins_y, ".", color=MODELS[opt_name]["color"], ms=1.5, zorder=2)
if su_wins_x:
    ax.plot(su_wins_x, su_wins_y, ".", color=MODELS[su_name]["color"], ms=1.5, zorder=2)

# Shared crank elements
shared_art = {
    "AB": ax.plot([], [], "-", color="#ffdd57", lw=2.5, zorder=3)[0],
    "A":  ax.plot([], [], "o", color="#88ff88", ms=8, zorder=5)[0],
    "Albl": ax.annotate("A", (0,0), xytext=(5,5), textcoords="offset points", color="#88ff88", fontsize=8, zorder=7)
}

art = {}
info_y_starts = [0.98, 0.98, 0.98]
info_x_starts = [0.02, 0.35, 0.68]

for idx, (name, mdl) in enumerate(MODELS.items()):
    c = mdl["color"]
    line_fmt = "-" if mdl.get("exact") else "-o"
    
    # Use just the first line of the name for the legend
    short_name = name.split('\n')[0]
    
    art[name] = {
        "beam": ax.plot([], [], line_fmt, color=c, lw=2.5, ms=5, zorder=4, label=short_name)[0],
        "QA":   ax.plot([], [], "-",  color=c, alpha=0.6, lw=1.5, zorder=3)[0],
        "Q":    ax.plot([], [], "D",  color=c, ms=6, zorder=5)[0],
        "Qlbl": ax.annotate("Q", (0,0), xytext=(5,5), textcoords="offset points", color=c, fontsize=8, zorder=7),
        "info": ax.text(info_x_starts[idx], info_y_starts[idx], "", transform=ax.transAxes,
                        color=c, fontsize=9, va="top", family="monospace"),
    }

ax.legend(facecolor="#1a1a3a", edgecolor="#334", labelcolor="white", fontsize=10, loc="lower right")

# ── Widgets & Interactivity ───────────────────────────────────────────────────
fig.subplots_adjust(bottom=0.20)

ax_slider = fig.add_axes([0.35, 0.05, 0.50, 0.03], facecolor="#1e2240")
slider = Slider(ax_slider, 'Crank Angle ψ (deg)', 0.0, 360.0, valinit=0.0, color="#E87C4C")

ax_play = fig.add_axes([0.05, 0.045, 0.08, 0.04])
btn_play = Button(ax_play, 'Pause', color="#1e2240", hovercolor="#334")
btn_play.label.set_color("white")

ax_check = fig.add_axes([0.80, 0.55, 0.16, 0.18], facecolor="#1e2240")
model_names = [n.split('\n')[0] for n in MODELS.keys()]
visibility = {name: True for name in MODELS}
chk = CheckButtons(ax_check, model_names, [True, True, True])

for idx, text in enumerate(chk.labels):
    model_key = list(MODELS.keys())[idx]
    text.set_color(MODELS[model_key]["color"])
    text.set_fontweight("bold")

phi_txt = fig.text(0.5, 0.12, "", ha="center", color="white", fontsize=13, fontweight="bold")
slider.label.set_color("white")
slider.valtext.set_color("white")

is_playing = True

def draw_frame(val):
    idx = (np.abs(np.degrees(psis) - val)).argmin()
    i = idx
    psi = psis[i]
    A = crank_A(psi)
    phi_txt.set_text(f"ψ = {np.degrees(psi):6.1f}°   (CW from neutral)")

    shared_art["AB"].set_data([A[0], B[0]], [A[1], B[1]])
    shared_art["A"].set_data([A[0]], [A[1]])
    shared_art["Albl"].set_position(A + np.array([0.04, 0.04]))

    # Pre-fetch Exact Beam position
    Q_ex = None
    th0_ex = None
    exact_name = "Exact Beam\n(Continuous BVP)"
    if frames[exact_name][i] is not None:
        Q_ex = np.array([frames[exact_name][i]['X'][-1], frames[exact_name][i]['Y'][-1]])
        th0_ex = frames[exact_name][i]['th0']

    current_errors = {}
    for name in MODELS:
        a  = art[name]
        fr = frames[name][i]

        if not visibility[name] or fr is None:
            for key in ("beam","QA","Q"):
                a[key].set_data([], [])
            a["Qlbl"].set_text("")
            a["info"].set_text(name.split('\n')[0] + (": Hidden" if not visibility[name] else ": No solution"))
            continue

        a["Qlbl"].set_text("Q")
        
        if MODELS[name].get("exact"):
            th0 = fr['th0']
            Q = np.array([fr['X'][-1], fr['Y'][-1]])
            a["beam"].set_data(fr['X'], fr['Y'])
            a["QA"].set_data([Q[0], A[0]], [Q[1], A[1]])
            a["Q"].set_data([Q[0]], [Q[1]])
            a["Qlbl"].set_position(Q + np.array([-0.05, 0.04]))
            a["info"].set_text(
                f"{name.split(chr(10))[0]}\n"
                f"Q  = ({Q[0]:.3f}, {Q[1]:.3f})\n\n"
                f"θ₀ = {np.degrees(th0):+7.2f}°\n"
            )
        else:
            th0, T1, T2, T3 = fr
            gamma = MODELS[name]["gamma"]
            pts   = fk_points(gamma, T1, T2, T3)
            Q     = np.array(pts[-1])

            bx = [p[0] for p in pts];  by = [p[1] for p in pts]
            a["beam"].set_data(bx, by)
            a["QA"].set_data([Q[0], A[0]], [Q[1], A[1]])
            a["Q"].set_data([Q[0]], [Q[1]])
            a["Qlbl"].set_position(Q + np.array([-0.05, 0.04]))
            
            err_text = ""
            err_th0_text = ""
            if Q_ex is not None:
                err_x = abs(Q[0] - Q_ex[0]) * 100
                err_y = abs(Q[1] - Q_ex[1]) * 100
                err_norm = np.linalg.norm(Q - Q_ex)
                current_errors[name] = err_norm
                err_text = f"err= (X:{err_x:.2f}%, Y:{err_y:.2f}%)\n"
                
                if th0_ex != 0:
                    err_th0 = abs((th0 - th0_ex)/th0_ex) * 100
                    err_th0_text = f"err= {err_th0:.2f}%"
                else:
                    err_th0_text = "err= N/A"

            a["info"].set_text(
                f"{name.split(chr(10))[0]}\n"
                f"Q  = ({Q[0]:.3f}, {Q[1]:.3f})\n"
                f"{err_text}"
                f"θ₀ = {np.degrees(th0):+7.2f}°\n"
                f"{err_th0_text}"
            )

    fig.canvas.draw_idle()

def update_anim(frame):
    if is_playing:
        val = slider.val + (360.0 / N)
        if val >= 360.0: val -= 360.0
        slider.set_val(val)

def toggle_play(event):
    global is_playing
    is_playing = not is_playing
    btn_play.label.set_text('Pause' if is_playing else 'Play')
    btn_play.label.set_color("white")

def on_check(label):
    for name in MODELS:
        if name.startswith(label):
            visibility[name] = not visibility[name]
    draw_frame(slider.val)

slider.on_changed(draw_frame)
btn_play.on_clicked(toggle_play)
chk.on_clicked(on_check)

draw_frame(0.0)
ani = animation.FuncAnimation(fig, update_anim, interval=40, cache_frame_data=False)
plt.show()
