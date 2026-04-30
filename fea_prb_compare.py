"""
ME 7751: Compliant Mechanism Design
FEA vs Analytical PRB Comparison

Large-deflection FEA of the compliant 4-bar beam (OQ) using
CalculiX (bundled with FreeCAD), compared with PRB 3R models.

What it does:
1. Builds a CalculiX .inp for a cantilever beam (B32R elements,
   NLGEOM large deflection) for each load case.
2. Runs ccx (the FreeCAD-bundled CalculiX solver).
3. Parses tip displacement from the .dat output file.
4. Compares FEA tip position with both PRB 3R model predictions.
5. Saves a CSV table and PNG comparison plot.
"""

import os, math, subprocess, re, csv
import FreeCAD

# -- Locate CalculiX (bundled with FreeCAD) ------------------------------------
_fc_bin = os.path.join(FreeCAD.getHomePath(), "bin")
CCX = None
for _c in [os.path.join(_fc_bin, "ccx.exe"),
           os.path.join(_fc_bin, "ccx"),
           "ccx"]:
    if os.path.exists(_c):
        CCX = _c
        break
if CCX is None:
    CCX = "ccx"
print("CalculiX: " + CCX)

# -- Physical beam parameters --------------------------------------------------
L    = 100.0      # beam length [mm]
b    = 5.0        # cross-section width  [mm]
h    = 5.0        # cross-section height [mm]
E    = 210000.0   # Young's modulus [MPa]
nu   = 0.3        # Poisson's ratio
rho  = 7.85e-9    # density [t/mm^3] -- required by CalculiX

I_cs = b * h**3 / 12.0
EI   = E * I_cs
print("I = %.4f mm^4,  EI = %.1f N.mm^2,  EI/L = %.2f N.mm" % (I_cs, EI, EI/L))

N_el = 30        # number of B32R elements (gives 61 nodes)

# -- Output directory ----------------------------------------------------------
OUT = r"C:\Users\ejones\Desktop\ME 7751 Compliant Mechanism Design\FEA"
os.makedirs(OUT, exist_ok=True)

# -- PRB 3R models -------------------------------------------------------------
MODELS = {
    "Su2009":   {"gamma": [0.10, 0.35, 0.40, 0.15], "k": [3.51, 2.99, 2.58]},
    'OurModel': {'gamma': [0.09123, 0.39120, 0.27428, 0.24329],
                 'k': [4.02860, 4.89071, 1.82713]},
}

def prb_ik(gamma, Qx, Qy, th0):
    """PRB 3R inverse kinematics. Qx, Qy normalised by L. Returns (T1,T2,T3) or None."""
    g0, g1, g2, g3 = gamma
    px  = Qx - g3*math.cos(th0) - g0
    py  = Qy - g3*math.sin(th0)
    arg = (px**2 + py**2 - g1**2 - g2**2) / (2*g1*g2)
    if abs(arg) > 1.0:
        return None
    T2  = math.acos(max(-1.0, min(1.0, arg)))
    d   = g1**2 + g2**2 + 2*g1*g2*math.cos(T2)
    if d < 1e-12:
        return None
    T1  = math.atan2(py*(g1+g2*math.cos(T2)) - px*g2*math.sin(T2),
                     px*(g1+g2*math.cos(T2)) + py*g2*math.sin(T2))
    return T1, T2, th0-T1-T2

def prb_fk(gamma, T1, T2, T3):
    """PRB 3R forward kinematics. Returns (Qx, Qy) normalised by L."""
    g0, g1, g2, g3 = gamma
    Qx = g0 + g1*math.cos(T1) + g2*math.cos(T1+T2) + g3*math.cos(T1+T2+T3)
    Qy =      g1*math.sin(T1) + g2*math.sin(T1+T2) + g3*math.sin(T1+T2+T3)
    return Qx, Qy

def prb_statics(gamma, k_list, xi, phi, beta, max_iter=200, tol=1e-10):
    """
    PRB forward statics: given load (xi, phi, beta), find joint angles by
    fixed-point iteration  T_i <- tau_i(T) / k_i  until converged.
    Returns (T1,T2,T3, Qx,Qy) or None.
    """
    g0, g1, g2, g3 = gamma
    k1, k2, k3 = k_list
    Wx = 2*xi*math.cos(phi)
    Wy = 2*xi*math.sin(phi)

    # Initial guess: linearise about undeformed state
    T1 = beta/k1 if k1 > 1e-12 else 0.0
    T2 = beta/k2 if k2 > 1e-12 else 0.0
    T3 = beta/k3 if k3 > 1e-12 else 0.0

    for _ in range(max_iter):
        th0 = T1+T2+T3
        c0   = math.cos(th0);        s0   = math.sin(th0)
        c12  = math.cos(T1+T2);      s12  = math.sin(T1+T2)
        c1   = math.cos(T1);         s1   = math.sin(T1)

        # Jacobian transpose rows (dQx/dTi, dQy/dTi, 1)
        # tau_i = J_i^T . W where W = [Wx, Wy, beta]
        tau1 = (-(g1*s1+g2*s12+g3*s0))*Wx + (g1*c1+g2*c12+g3*c0)*Wy + beta
        tau2 = (-(g2*s12+g3*s0))*Wx       + (g2*c12+g3*c0)*Wy       + beta
        tau3 = (-g3*s0)*Wx                + (g3*c0)*Wy               + beta

        T1n = tau1/k1 if k1 > 1e-12 else 0.0
        T2n = tau2/k2 if k2 > 1e-12 else 0.0
        T3n = tau3/k3 if k3 > 1e-12 else 0.0

        if max(abs(T1n-T1), abs(T2n-T2), abs(T3n-T3)) < tol:
            T1, T2, T3 = T1n, T2n, T3n
            break
        T1, T2, T3 = T1n, T2n, T3n

    Qx, Qy = prb_fk(gamma, T1, T2, T3)
    return T1, T2, T3, Qx, Qy

# -- Load cases ----------------------------------------------------------------
# Pure end-moment: M0 = EI*theta0/L  =>  beta = theta0 exactly
# Exact solution (circular arc): Qx = sin(t)/t, Qy = (1-cos(t))/t

theta0_M = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
LOAD_CASES = []

for th in theta0_M:
    M0 = EI * th / L
    LOAD_CASES.append({
        "id":       "M_%03.0fdeg" % math.degrees(th),
        "Fy_N":     0.0, "Mz_Nmm":  M0,
        "xi":       0.0, "phi":     math.pi/2, "beta": th,
        "th0_exact": th,
        "Qx_exact":  math.sin(th)/th,
        "Qy_exact":  (1-math.cos(th))/th,
        "type":     "pure_moment",
    })

for xi_val in [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]:
    Fy = 2 * xi_val * EI / L**2
    LOAD_CASES.append({
        "id":      "F_xi%.2f" % xi_val,
        "Fy_N":    Fy, "Mz_Nmm":  0.0,
        "xi":      xi_val, "phi": math.pi/2, "beta": 0.0,
        "th0_exact": None, "Qx_exact": None, "Qy_exact": None,
        "type":    "pure_force",
    })

print("\n%d load cases." % len(LOAD_CASES))

# -- CalculiX .inp writer ------------------------------------------------------
def make_inp(job_name, Fy_N, Mz_Nmm):
    """
    Write CalculiX .inp for a B32R cantilever beam.
    Beam axis = global x.  Bending in x-y plane.
    Fixed at node 1 (x=0).  Load at tip node (x=L).
    Local 2-axis = (0,0,1) => bending moment is DOF 6.
    """
    n_nodes  = 2 * N_el + 1
    tip_node = n_nodes

    lines = []

    # Nodes: uniformly spaced from x=0 to x=L
    lines.append("*NODE, NSET=NALL")
    for i in range(n_nodes):
        x = i * L / (n_nodes - 1)
        lines.append("%d, %.6f, 0.0, 0.0" % (i+1, x))

    # Elements (B32R: nodes n1-n2-n3 per element)
    lines.append("*ELEMENT, TYPE=B32R, ELSET=EBEAM")
    for e in range(N_el):
        n1 = 2*e + 1
        n2 = 2*e + 2
        n3 = 2*e + 3
        lines.append("%d, %d, %d, %d" % (e+1, n1, n2, n3))

    # Beam section: RECT h x b; local-2 = z => bending in x-y about z
    lines.append("*BEAM SECTION, ELSET=EBEAM, MATERIAL=STEEL, SECTION=RECT")
    lines.append("%.4f, %.4f" % (h, b))
    lines.append("0.0, 0.0, 1.0")

    # Material
    lines.append("*MATERIAL, NAME=STEEL")
    lines.append("*ELASTIC")
    lines.append("%.1f, %.2f" % (E, nu))
    lines.append("*DENSITY")
    lines.append("%.6e" % rho)

    # Node sets
    lines.append("*NSET, NSET=NFIX")
    lines.append("1")
    lines.append("*NSET, NSET=NTIP")
    lines.append("%d" % tip_node)

    # Boundary condition: all 6 DOF fixed at root
    lines.append("*BOUNDARY")
    lines.append("NFIX, 1, 6, 0.0")

    # Step: large-deflection static
    # INC=500 with small initial increment improves convergence for large rotations
    lines.append("*STEP, NLGEOM=YES, INC=500")
    lines.append("*STATIC")
    lines.append("0.005, 1.0, 1e-6, 0.05")   # init, total, min, max

    # Applied loads at tip
    loads_written = False
    if abs(Fy_N) > 1e-12 or abs(Mz_Nmm) > 1e-12:
        lines.append("*CLOAD")
        loads_written = True
    if abs(Fy_N) > 1e-12:
        lines.append("NTIP, 2, %.8f" % Fy_N)     # DOF 2 = y force
    if abs(Mz_Nmm) > 1e-12:
        lines.append("NTIP, 6, %.8f" % Mz_Nmm)   # DOF 6 = moment about z

    # Output: all-node displacements so we can compute tip slope via FD
    lines.append("*NODE PRINT, NSET=NALL, FREQUENCY=9999")
    lines.append("U")
    lines.append("*END STEP")

    inp_path = os.path.join(OUT, job_name + ".inp")
    with open(inp_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return inp_path, tip_node

# -- Run CalculiX --------------------------------------------------------------
def run_ccx(job_name):
    try:
        r = subprocess.run(
            [CCX, "-i", job_name],
            cwd=OUT,
            capture_output=True,
            text=True,
            timeout=180
        )
        if r.returncode != 0:
            print("  ccx stderr: " + r.stderr[-600:])
            return False
        return True
    except Exception as ex:
        print("  subprocess error: " + str(ex))
        return False

# -- Parse .dat results --------------------------------------------------------
def parse_dat(job_name):
    """
    Scan all-nodes displacement output.  Return (Ux_tip, Uy_tip, urz_approx)
    where urz_approx is estimated from finite difference of nodes 61 and 59.
    Key fix: do NOT reset mode on blank lines (CalculiX puts blank between
    header and data block).
    """
    dat = os.path.join(OUT, job_name + ".dat")
    if not os.path.exists(dat):
        print("  .dat not found")
        return None

    with open(dat) as f:
        raw_lines = f.readlines()

    n_nodes  = 2 * N_el + 1   # 61
    tip      = n_nodes         # 61
    near_tip = n_nodes - 2     # 59 = first node of last element

    node_disp = {}  # node_id -> (Ux, Uy)
    mode = None

    for line in raw_lines:
        ls = line.strip()
        ll = ls.lower()

        # Section header detection
        if 'displacements' in ll and ('vx' in ll or 'ux' in ll):
            mode = 'U'
            continue  # <-- do NOT reset on blank, only on new header

        # Data lines: try to parse node number + 3 floats
        if mode == 'U' and ls:
            parts = ls.split()
            if len(parts) >= 4:
                try:
                    nid = int(parts[0])
                    ux  = float(parts[1])
                    uy  = float(parts[2])
                    node_disp[nid] = (ux, uy)
                except ValueError:
                    pass  # header/other text - ignore

    if tip not in node_disp:
        print("  tip node %d not in parsed data. Found nodes: %s" % (
            tip, sorted(node_disp.keys())[:5]))
        return None

    ux_tip, uy_tip = node_disp[tip]

    # Tip slope from finite difference of deformed positions
    # Node index i (1-based) has undeformed x = (i-1)*L/(n_nodes-1)
    if near_tip in node_disp:
        ux_n, uy_n = node_disp[near_tip]
        x_tip = L + ux_tip                            # tip deformed x
        y_tip = uy_tip
        x_n   = (near_tip - 1) * L / (n_nodes - 1) + ux_n   # FIX: was near_tip*...
        y_n   = uy_n
        urz   = math.atan2(y_tip - y_n, x_tip - x_n)
    else:
        urz = 0.0

    return (ux_tip, uy_tip, urz, node_disp)

# -- Main comparison loop ------------------------------------------------------
results = []

for lc in LOAD_CASES:
    jname = lc["id"]
    print("\n--- %s   Fy=%.4fN   Mz=%.4fN.mm ---" % (jname, lc["Fy_N"], lc["Mz_Nmm"]))

    make_inp(jname, lc["Fy_N"], lc["Mz_Nmm"])
    ok = run_ccx(jname)
    if not ok:
        print("  SOLVER FAILED")
        continue

    disp = parse_dat(jname)
    if disp is None:
        print("  Parse failed")
        continue

    ux, uy, urz, node_disp = disp
    Qx_fea  = (L + ux) / L
    Qy_fea  = uy / L
    th0_fea = urz if urz is not None else 0.0

    print("  FEA: Qx/L=%.4f  Qy/L=%.4f  th0=%.2fdeg" % (
        Qx_fea, Qy_fea, math.degrees(th0_fea)))

    row = {
        "id":       jname,
        "type":     lc["type"],
        "Fy_N":     lc["Fy_N"],
        "Mz_Nmm":   lc["Mz_Nmm"],
        "Qx_fea":   Qx_fea,
        "Qy_fea":   Qy_fea,
        "th0_fea_deg": math.degrees(th0_fea),
    }

    if lc["Qx_exact"] is not None:
        row["Qx_exact"] = lc["Qx_exact"]
        row["Qy_exact"] = lc["Qy_exact"]
        ex = abs(Qx_fea - lc["Qx_exact"]) / max(abs(lc["Qx_exact"]), 1e-9) * 100
        ey = abs(Qy_fea - lc["Qy_exact"]) / max(abs(lc["Qy_exact"]), 1e-9) * 100
        print("  Exact: Qx/L=%.4f  Qy/L=%.4f" % (lc["Qx_exact"], lc["Qy_exact"]))
        print("  Error vs exact: Qx %.2f%%   Qy %.2f%%" % (ex, ey))
        row["errQx_exact"] = ex
        row["errQy_exact"] = ey

    # PRB forward statics: given the applied load, solve equilibrium, compare tip
    for mname in MODELS:
        mdl = MODELS[mname]
        sol = prb_statics(mdl["gamma"], mdl["k"],
                          lc["xi"], lc["phi"], lc["beta"])
        if sol:
            T1, T2, T3, Qx_p, Qy_p = sol
            ex = abs(Qx_p - Qx_fea) / max(abs(Qx_fea), 1e-9) * 100
            ey = abs(Qy_p - Qy_fea) / max(abs(Qy_fea), 1e-9) * 100
            print("  PRB-%s: Qx/L=%.4f  Qy/L=%.4f   err Qx %.3f%%  Qy %.3f%%" % (
                mname, Qx_p, Qy_p, ex, ey))
            row["Qx_" + mname] = Qx_p
            row["Qy_" + mname] = Qy_p
            row["errQx_" + mname] = ex
            row["errQy_" + mname] = ey
        else:
            print("  PRB-%s: statics did not converge" % mname)

    row["node_disp"] = node_disp   # store for animation
    results.append(row)

# -- Save CSV ------------------------------------------------------------------
if results:
    csv_path = os.path.join(OUT, "prb_fea_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nCSV: " + csv_path)

# -- Summary plot (tip locus) --------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG, PANEL = "#0d0d1a", "#111128"
    fig, axes = plt.subplots(2, 1, figsize=(7, 10), facecolor=BG)
    fig.suptitle("FEA vs PRB 3R Model -- Cantilever Beam Tip Locus",
                 color="white", fontsize=13, fontweight="bold")

    for ax, ltype, title in zip(axes,
        ["pure_moment", "pure_force"],
        ["Pure End Moment (beta=theta0)", "Pure Vertical Force (kappa=0)"]):

        ax.set_facecolor(PANEL)
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel("Qx / l", color="#99aabb")
        ax.set_ylabel("Qy / l", color="#99aabb")
        ax.tick_params(colors="#99aabb")
        for sp in ax.spines.values():
            sp.set_color("#334")
        ax.grid(True, color="#1e2240", lw=0.5)

        sub = [r for r in results if r["type"] == ltype]
        if not sub:
            continue

        ax.plot([r["Qx_fea"] for r in sub],
                [r["Qy_fea"] for r in sub],
                "o-", color="white", lw=2, ms=7, label="FEA (CalculiX)")

        if ltype == "pure_moment" and "Qx_exact" in sub[0]:
            ax.plot([r["Qx_exact"] for r in sub],
                    [r["Qy_exact"] for r in sub],
                    "--", color="#aaffaa", lw=1.5, label="Exact (arc)")
        elif ltype == "pure_force":
            exact_x = [1.00000,0.99934,0.99742,0.99423,0.98978,0.98408,0.97712,0.96891,0.95946,0.94878,0.93686,0.92373,0.90937,0.89381,0.87704,0.85907,0.83989,0.81952,0.79793,0.77511,0.75105,0.72571,0.69903,0.67095,0.64135,0.61007,0.57688,0.54140,0.50302,0.46065]
            exact_y = [0.00067,0.03314,0.06556,0.09790,0.13012,0.16217,0.19401,0.22560,0.25691,0.28789,0.31851,0.34874,0.37852,0.40785,0.43667,0.46497,0.49271,0.51988,0.54646,0.57242,0.59776,0.62247,0.64656,0.67006,0.69298,0.71538,0.73737,0.75907,0.78076,0.80286]
            
            # Trim the exact curve so it doesn't overshoot the data points
            max_y = max(r["Qy_fea"] for r in sub) * 1.05
            ex_x = [x for x, y in zip(exact_x, exact_y) if y <= max_y]
            ex_y = [y for y in exact_y if y <= max_y]
            
            ax.plot(ex_x, ex_y, "--", color="#aaffaa", lw=1.5, label="Exact (Elliptic Integrals)")

        fea_pts = [(r["Qx_fea"], r["Qy_fea"]) for r in sub]
        mean_x = sum(p[0] for p in fea_pts) / len(fea_pts)
        mean_y = sum(p[1] for p in fea_pts) / len(fea_pts)
        SST = sum((p[0] - mean_x)**2 + (p[1] - mean_y)**2 for p in fea_pts)

        for mname, color in [("Su2009", "#4C9BE8"), ("OurModel", "#E87C4C")]:
            key = "Qx_" + mname
            pts = [(r[key], r["Qy_" + mname]) for r in sub if key in r]
            if pts:
                SSE = sum((p[0] - f[0])**2 + (p[1] - f[1])**2 for p, f in zip(pts, fea_pts))
                R2 = 1.0 - (SSE / SST) if SST > 0 else 1.0
                
                label_str = f"PRB {mname} ($R^2={R2:.5f}$)"
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        "s--", color=color, lw=1.8, ms=6, label=label_str)

        ax.legend(facecolor="#1a1a3a", edgecolor="#334",
                  labelcolor="white", fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(OUT, "prb_fea_comparison.png")
    plt.savefig(fig_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print("Summary plot: " + fig_path)

except Exception as e:
    print("Summary plot skipped: " + str(e))

# -- Per-case beam shape images ------------------------------------------------
# Each image shows the full deformed beam (FEA) vs both PRB 3R link chains.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG, PANEL = "#0d0d1a", "#111128"
    n_nodes = 2 * N_el + 1
    IMG_DIR = os.path.join(OUT, "frames")
    os.makedirs(IMG_DIR, exist_ok=True)

    # Helper: PRB beam shape in mm from joint angles
    def prb_shape_mm(gamma, T1, T2, T3):
        g0, g1, g2, g3 = gamma
        pts = [(0.0, 0.0)]
        x, y = g0*L, 0.0;  pts.append((x, y))
        x += g1*L*math.cos(T1);       y += g1*L*math.sin(T1);       pts.append((x, y))
        x += g2*L*math.cos(T1+T2);    y += g2*L*math.sin(T1+T2);    pts.append((x, y))
        x += g3*L*math.cos(T1+T2+T3); y += g3*L*math.sin(T1+T2+T3); pts.append((x, y))
        return [p[0] for p in pts], [p[1] for p in pts]

    print("\nExporting per-case beam shape images...")

    for idx, r in enumerate(results):
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
        ax.set_facecolor(PANEL)

        # Title
        ltype = r["type"]
        if ltype == "pure_moment":
            load_str = "Pure Moment  M = %.0f N.mm  (beta = %.2f rad)" % (
                r["Mz_Nmm"], r["Mz_Nmm"]*L/EI)
        else:
            load_str = "Pure Force  F = %.1f N  (xi = %.2f)" % (
                r["Fy_N"], r["Fy_N"]*L**2/(2*EI))

        ax.set_title("Case %d/%d: %s\n%s" % (idx+1, len(results), r["id"], load_str),
                      color="white", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("x  [mm]", color="#99aabb", fontsize=10)
        ax.set_ylabel("y  [mm]", color="#99aabb", fontsize=10)
        ax.tick_params(colors="#99aabb")
        for sp in ax.spines.values():
            sp.set_color("#334")
        ax.grid(True, color="#1e2240", lw=0.5, alpha=0.5)

        # Undeformed reference
        ax.plot([0, L], [0, 0], "--", color="#555577", lw=1.5, label="Undeformed")

        # FEA deformed shape (all nodes)
        nd = r.get("node_disp", {})
        fea_x, fea_y = [], []
        for nid in sorted(nd.keys()):
            ux_i, uy_i = nd[nid]
            x_ref = (nid - 1) * L / (n_nodes - 1)
            fea_x.append(x_ref + ux_i)
            fea_y.append(uy_i)
        if fea_x:
            ax.plot(fea_x, fea_y, "-", color="white", lw=3.0, label="FEA (CalculiX)", zorder=4)
            ax.plot(fea_x[-1], fea_y[-1], "D", color="white", ms=8, zorder=5)

        # PRB models
        prb_colors = {"Su2009": "#4C9BE8", "OurModel": "#E87C4C"}
        prb_labels = {"Su2009": "PRB Su (2009)", "OurModel": "PRB Our Model"}

        err_lines = []
        for mname in MODELS:
            mdl = MODELS[mname]
            sol = prb_statics(mdl["gamma"], mdl["k"],
                              r.get("xi_val", 0) if "xi_val" in r else
                              (r["Fy_N"]*L**2/(2*EI) if r["Fy_N"] > 0 else 0.0),
                              math.pi/2,
                              r["Mz_Nmm"]*L/EI if r["Mz_Nmm"] > 0 else 0.0)
            if sol is None:
                continue
            T1, T2, T3, Qx_p, Qy_p = sol
            px, py = prb_shape_mm(mdl["gamma"], T1, T2, T3)
            c = prb_colors[mname]

            ax.plot(px, py, "-o", color=c, lw=2.2, ms=7, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=prb_labels[mname])
            ax.plot(px[-1], py[-1], "s", color=c, ms=9, zorder=5)

            # Error text
            ex = abs(Qx_p - r["Qx_fea"]) / max(abs(r["Qx_fea"]), 1e-9) * 100
            ey = abs(Qy_p - r["Qy_fea"]) / max(abs(r["Qy_fea"]), 1e-9) * 100
            err_lines.append("%s:  err Qx=%.3f%%  Qy=%.3f%%" % (prb_labels[mname], ex, ey))

        # FEA tip info
        info = "FEA tip: Qx/L=%.4f  Qy/L=%.4f  th0=%.1f deg" % (
            r["Qx_fea"], r["Qy_fea"], r["th0_fea_deg"])
        err_text = info + "\n" + "\n".join(err_lines)

        ax.text(0.02, 0.97, err_text, transform=ax.transAxes,
                color="#ccddee", fontsize=8.5, va="top", family="monospace",
                bbox=dict(facecolor="#1a1a3a", edgecolor="#334",
                          alpha=0.9, boxstyle="round,pad=0.4"))

        # Fixed-end marker
        ax.plot(0, 0, "^", color="white", ms=10, zorder=6)
        ax.annotate("O (fixed)", (0, 0), xytext=(3, -8),
                    textcoords="offset points", color="#aaa", fontsize=7)

        ax.set_aspect("equal")
        ax.legend(loc="lower left", facecolor="#1a1a3a", edgecolor="#334",
                  labelcolor="white", fontsize=9, framealpha=0.9)

        plt.tight_layout()
        img_name = "case_%02d_%s.png" % (idx+1, r["id"])
        img_path = os.path.join(IMG_DIR, img_name)
        plt.savefig(img_path, dpi=150, facecolor=BG, bbox_inches="tight")
        plt.close()
        print("  Saved: " + img_name)

    print("All case images saved to: " + IMG_DIR)

except Exception as e:
    import traceback
    print("Per-case export error: " + str(e))
    traceback.print_exc()

print("\n=== Done ===")

# ─────────────────────────────────────────────────────────────────────────────
# FreeCAD GUI ANIMATION
# Creates 3D wire objects in the FreeCAD viewport and cycles through each
# solved load case, updating the deformed beam shape in real time.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import FreeCADGui, Part, time
    from FreeCAD import Base

    if not results:
        raise RuntimeError("No results to animate")

    n_nodes = 2 * N_el + 1

    # ── Create or reuse document ──────────────────────────────────────────────
    doc_name = "PRB_FEA_Anim"
    if doc_name in FreeCAD.listDocuments():
        FreeCAD.closeDocument(doc_name)
    doc = FreeCAD.newDocument(doc_name)

    # ── Helper: 3D wire from list of (x,y) points (z=0) ──────────────────────
    def make_wire(name, xy_pairs, color=(1,1,1)):
        pts = [Base.Vector(x, y, 0) for x, y in xy_pairs]
        if len(pts) < 2:
            return None
        obj = doc.addObject("Part::Feature", name)
        obj.Shape = Part.makePolygon(pts)
        obj.ViewObject.LineColor = color
        obj.ViewObject.LineWidth = 2.0
        return obj

    # ── Static geometry: undeformed beam and ground ───────────────────────────
    make_wire("Undeformed", [(0, 0), (L, 0)], color=(0.4, 0.4, 0.4))
    # Fixed-end triangle marker
    tri = doc.addObject("Part::Feature", "FixedEnd")
    tri.Shape = Part.makePolygon([
        Base.Vector(0, 0, 0),
        Base.Vector(-6, 5, 0),
        Base.Vector(-6, -5, 0),
        Base.Vector(0, 0, 0),
    ])
    tri.ViewObject.LineColor = (1, 1, 1)

    # ── Animated objects (will be updated each frame) ─────────────────────────
    fea_wire  = make_wire("FEA_Deformed",    [(0,0),(L,0)], color=(1.0, 1.0, 1.0))
    prb_wires = {
        "Su2009":   make_wire("PRB_Su2009",   [(0,0),(L,0)], color=(0.30, 0.61, 0.91)),
        "OurModel": make_wire("PRB_OurModel", [(0,0),(L,0)], color=(0.91, 0.49, 0.30)),
    }

    # Tip markers
    tip_fea = doc.addObject("Part::Feature", "Tip_FEA")
    tip_fea.ViewObject.PointColor  = (1, 1, 1)
    tip_fea.ViewObject.PointSize   = 8.0

    # Label annotation (Draft text if available, else skip)
    label_obj = None
    try:
        import Draft
        label_obj = Draft.makeText(["Initialising..."], point=Base.Vector(-5, 45, 0))
        label_obj.ViewObject.FontSize   = 5
        label_obj.ViewObject.TextColor  = (1, 1, 1)
    except Exception:
        pass

    doc.recompute()
    FreeCADGui.ActiveDocument.ActiveView.fitAll()

    # ── PRB forward-kinematics beam shape ────────────────────────────────────
    def prb_beam_xy(gamma, T1, T2, T3):
        """Return list of (x,y) in mm for the 5 PRB chain nodes."""
        g0, g1, g2, g3 = gamma
        pts = [(0.0, 0.0)]
        x, y = g0 * L, 0.0
        pts.append((x, y))
        x += g1 * L * math.cos(T1);          y += g1 * L * math.sin(T1)
        pts.append((x, y))
        x += g2 * L * math.cos(T1+T2);       y += g2 * L * math.sin(T1+T2)
        pts.append((x, y))
        x += g3 * L * math.cos(T1+T2+T3);    y += g3 * L * math.sin(T1+T2+T3)
        pts.append((x, y))
        return pts

    def update_wire(obj, xy_pairs):
        pts = [Base.Vector(x, y, 0) for x, y in xy_pairs]
        if len(pts) >= 2:
            obj.Shape = Part.makePolygon(pts)

    # ── Animation loop using Qt timer ─────────────────────────────────────────
    from PySide6 import QtCore

    anim_idx   = [0]
    anim_repeat = 3        # number of full cycles before stopping
    anim_cycle  = [0]
    INTERVAL_MS = 1200     # ms per frame

    def anim_step():
        idx = anim_idx[0]
        if idx >= len(results):
            anim_idx[0]  = 0
            anim_cycle[0] += 1
            if anim_cycle[0] >= anim_repeat:
                timer.stop()
                print("Animation complete.")
                return
            idx = 0

        r = results[idx]
        nd = r.get("node_disp", {})

        # FEA deformed beam
        fea_pts = []
        for nid in sorted(nd.keys()):
            ux_i, uy_i = nd[nid]
            x_ref = (nid - 1) * L / (n_nodes - 1)
            fea_pts.append((x_ref + ux_i, uy_i))
        if fea_pts:
            update_wire(fea_wire, fea_pts)

        # PRB predictions
        for mname, wire_obj in prb_wires.items():
            key = "Qx_" + mname
            if key in r:
                gamma = MODELS[mname]["gamma"]
                # Reconstruct T1,T2,T3 from stored Qx/Qy and th0
                th0 = math.radians(r["th0_fea_deg"])
                sol = prb_ik(gamma, r["Qx_fea"], r["Qy_fea"], th0)
                if sol:
                    T1, T2, T3 = sol
                    xy = prb_beam_xy(gamma, T1, T2, T3)
                    update_wire(wire_obj, xy)

        # Label
        ltype  = r["type"]
        load_s = ("Mz=%.0fN.mm" % r["Mz_Nmm"]) if ltype == "pure_moment" \
                 else ("Fy=%.1fN" % r["Fy_N"])
        info = [
            "Case %d/%d: %s" % (idx+1, len(results), r["id"]),
            "Load: " + load_s,
            "FEA:  Qx/L=%.4f  Qy/L=%.4f  th0=%.1fdeg" % (
                r["Qx_fea"], r["Qy_fea"], r["th0_fea_deg"]),
        ]
        for mname in MODELS:
            if "Qx_" + mname in r:
                info.append("PRB-%s: err Qx %.2f%%  Qy %.2f%%" % (
                    mname,
                    r.get("errQx_" + mname, 0),
                    r.get("errQy_" + mname, 0)))
            else:
                info.append("PRB-%s: out of reach" % mname)

        if label_obj:
            try:
                label_obj.Text = info
            except Exception:
                pass

        doc.recompute()
        FreeCADGui.updateGui()
        anim_idx[0] += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(anim_step)
    timer.start(INTERVAL_MS)
    print("Animation started (%d cases, %d cycles, %.1fs/frame)." % (
        len(results), anim_repeat, INTERVAL_MS/1000.0))
    print("The FreeCAD 3D view will update automatically.")
    print("Tip: use View > Standard Views > Home to reset camera.")

except ImportError as e:
    print("GUI animation skipped (no FreeCADGui): " + str(e))
except Exception as e:
    import traceback
    print("Animation error: " + str(e))
    traceback.print_exc()

