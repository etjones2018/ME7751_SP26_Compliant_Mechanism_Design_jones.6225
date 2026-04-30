"""
Direct PRB 3R Parameter Optimization
=====================================
Finds optimal (gamma, k) by directly minimizing the tip position error
between PRB forward-statics predictions and exact beam solutions.

This replaces the two-stage approach (gamma search -> k regression) with
a single-stage global optimization over all 7 parameters simultaneously.

Optimizes:  gamma = [g0, g1, g2, g3]   (characteristic radius factors)
            k     = [k1, k2, k3]        (torsional spring stiffness, EI/l)

Constraints:
   sum(gamma_i) = 1
   gamma_i >= 0.01
   k_i > 0
   sum(1/k_i) = 1    (compliance constraint, Eq 26 of Su 2009)

Objective:
   Minimise  E = sum_over_cases( (Qx_prb - Qx_exact)^2 + (Qy_prb - Qy_exact)^2 )
   summed over pure-moment and pure-force deflection datasets.
"""

"""
ME 7751: Compliant Mechanism Design
Global PRB 3R Optimization (Su 2009 Replication)

This script replicates the numerical exact BVP integration and the grid search
methodology to find the globally optimized parameters for the 3R PRB model 
as originally presented by Hai-Jun Su (2009).
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import math, time

# ── Exact beam deflection data ────────────────────────────────────────────────

def pure_moment_data(N=80):
    """Exact circular arc: Qx=sin(t)/t, Qy=(1-cos(t))/t, xi=0, beta=t."""
    theta0 = np.linspace(0.05, 1.4, N)
    Qx   = np.sin(theta0) / theta0
    Qy   = (1 - np.cos(theta0)) / theta0
    xi   = np.zeros(N)
    beta = theta0.copy()
    return xi, beta, Qx, Qy

def pure_force_data(N=60):
    """Numerical integration using scipy.integrate to handle the tip singularity."""
    import scipy.integrate as integrate
    phi = np.pi / 2
    theta0_all = np.linspace(0.05, np.pi/2 * 0.95, N)
    xi_list, Qx_list, Qy_list = [], [], []

    for t0 in theta0_all:
        def integr_0(th):
            denom = np.cos(t0 - phi) - np.cos(th - phi)
            if denom <= 0: return 0.0
            return 1.0 / np.sqrt(denom)
        def integr_a(th):
            denom = np.cos(t0 - phi) - np.cos(th - phi)
            if denom <= 0: return 0.0
            return np.cos(th) / np.sqrt(denom)
        def integr_b(th):
            denom = np.cos(t0 - phi) - np.cos(th - phi)
            if denom <= 0: return 0.0
            return np.sin(th) / np.sqrt(denom)
        
        I0, _ = integrate.quad(integr_0, 0, t0)
        Ia, _ = integrate.quad(integr_a, 0, t0)
        Ib, _ = integrate.quad(integr_b, 0, t0)
        
        xi_val = (I0 / 2.0) ** 2
        if xi_val < 1e-12:
            continue
        fac = 1.0 / (2.0 * np.sqrt(xi_val))
        xi_list.append(xi_val)
        Qx_list.append(fac * Ia)
        Qy_list.append(fac * Ib)

    return (np.array(xi_list), np.zeros(len(xi_list)),
            np.array(Qx_list), np.array(Qy_list))


# Combined loading at intermediate kappa values
def combined_load_data(kappa, N=50):
    """General beam at load ratio kappa, phi=pi/2."""
    phi = np.pi / 2
    if kappa >= 1e6:
        return pure_moment_data(N)

    theta0_all = np.linspace(0.05, min(1.4, np.pi/2*0.95 if kappa < 0.5 else 1.4), N)
    xi_list, beta_list, Qx_list, Qy_list = [], [], [], []

    for t0 in theta0_all:
        Nq = 1000
        th = np.linspace(0, t0 * (1 - 1e-6), Nq)
        f = 1.0 + (np.cos(phi - t0) - np.cos(phi - th)) / max(kappa, 1e-12)
        f = np.maximum(f, 1e-20)
        sq = np.sqrt(f)
        I0 = np.trapezoid(1.0 / sq, th)
        Ia = np.trapezoid(np.cos(th) / sq, th)
        Ib = np.trapezoid(np.sin(th) / sq, th)
        beta_val = I0
        if abs(beta_val) < 1e-12:
            continue
        xi_val = beta_val**2 / (4.0 * kappa)
        xi_list.append(xi_val)
        beta_list.append(beta_val)
        Qx_list.append(Ia / beta_val)
        Qy_list.append(Ib / beta_val)

    return (np.array(xi_list), np.array(beta_list),
            np.array(Qx_list), np.array(Qy_list))


# ── PRB forward statics ──────────────────────────────────────────────────────

def prb_fk(gamma, T1, T2, T3):
    g0, g1, g2, g3 = gamma
    Qx = g0 + g1*math.cos(T1) + g2*math.cos(T1+T2) + g3*math.cos(T1+T2+T3)
    Qy =      g1*math.sin(T1) + g2*math.sin(T1+T2) + g3*math.sin(T1+T2+T3)
    return Qx, Qy

def prb_statics(gamma, k, xi, phi, beta, max_iter=300, tol=1e-11):
    """Fixed-point iteration for PRB equilibrium."""
    g0, g1, g2, g3 = gamma
    k1, k2, k3 = k
    Wx = 2*xi*math.cos(phi)
    Wy = 2*xi*math.sin(phi)

    T1 = beta/k1 if k1 > 1e-12 else 0.0
    T2 = beta/k2 if k2 > 1e-12 else 0.0
    T3 = beta/k3 if k3 > 1e-12 else 0.0

    for _ in range(max_iter):
        th0 = T1+T2+T3
        c0, s0   = math.cos(th0),    math.sin(th0)
        c12, s12 = math.cos(T1+T2),  math.sin(T1+T2)
        c1, s1   = math.cos(T1),     math.sin(T1)

        tau1 = (-(g1*s1+g2*s12+g3*s0))*Wx + (g1*c1+g2*c12+g3*c0)*Wy + beta
        tau2 = (-(g2*s12+g3*s0))*Wx       + (g2*c12+g3*c0)*Wy       + beta
        tau3 = (-g3*s0)*Wx                + (g3*c0)*Wy               + beta

        T1n = tau1/k1
        T2n = tau2/k2
        T3n = tau3/k3

        if max(abs(T1n-T1), abs(T2n-T2), abs(T3n-T3)) < tol:
            T1, T2, T3 = T1n, T2n, T3n
            break
        T1, T2, T3 = T1n, T2n, T3n

    return prb_fk(gamma, T1, T2, T3)


# ── Build datasets ────────────────────────────────────────────────────────────
print("Building deflection datasets...")
import sys
datasets = []

xi_m, beta_m, Qx_m, Qy_m = pure_moment_data(20)
datasets.append(("moment", xi_m, beta_m, Qx_m, Qy_m, 1.0))

xi_f, beta_f, Qx_f, Qy_f = pure_force_data(15)
datasets.append(("force", xi_f, beta_f, Qx_f, Qy_f, 1.0))

for kap in [0.5, 2.0, 5.0, 25.0]:
    xi_c, beta_c, Qx_c, Qy_c = combined_load_data(kap, 12)
    datasets.append(("kappa=%.1f" % kap, xi_c, beta_c, Qx_c, Qy_c, 1.0))

total_pts = sum(len(d[3]) for d in datasets)
print("  %d datasets, %d total data points" % (len(datasets), total_pts))
sys.stdout.flush()


# ── Objective function ────────────────────────────────────────────────────────

def objective(params):
    """
    params = [g0, g1, g2, k1, k2, k3]
    g3 = 1 - g0 - g1 - g2   (enforced by constraint)
    k3 = 1/(1 - 1/k1 - 1/k2) (enforced by compliance constraint)
    """
    g0, g1, g2, k1, k2 = params[:5]
    g3 = 1.0 - g0 - g1 - g2
    if g3 < 0.01 or g0 < 0.01 or g1 < 0.01 or g2 < 0.01:
        return 1e10
    if k1 < 0.1 or k2 < 0.1:
        return 1e10

    # Compliance constraint: 1/k1 + 1/k2 + 1/k3 = 1  =>  k3 = 1/(1 - 1/k1 - 1/k2)
    denom = 1.0 - 1.0/k1 - 1.0/k2
    if denom <= 0.01:
        return 1e10
    k3 = 1.0 / denom
    if k3 < 0.1 or k3 > 50:
        return 1e10

    gamma = [g0, g1, g2, g3]
    k     = [k1, k2, k3]
    phi   = math.pi / 2

    err = 0.0
    for name, xi_arr, beta_arr, Qx_arr, Qy_arr, weight in datasets:
        for i in range(len(xi_arr)):
            try:
                Qx_p, Qy_p = prb_statics(gamma, k, xi_arr[i], phi, beta_arr[i])
                err += weight * ((Qx_p - Qx_arr[i])**2 + (Qy_p - Qy_arr[i])**2)
            except Exception:
                err += weight * 1.0  # penalty
    return err


# ── Reference: Su (2009) objective value ──────────────────────────────────────
su_params = [0.10, 0.35, 0.40, 3.51, 2.99]
su_obj = objective(su_params)
print("\nSu (2009) objective:  %.8f" % su_obj)
sys.stdout.flush()


# -- Optimization Strategy 1: Differential Evolution (global) ------------------
print("\n" + "="*65)
print("Strategy 1: Differential Evolution (global search)")
print("="*65)
sys.stdout.flush()

bounds = [
    (0.01, 0.30),  # g0
    (0.15, 0.60),  # g1
    (0.20, 0.60),  # g2
    (1.5,  12.0),  # k1
    (1.5,  8.0),   # k2
]

call_count = [0]
def objective_verbose(params):
    call_count[0] += 1
    if call_count[0] % 500 == 0:
        print("  eval %d..." % call_count[0])
        sys.stdout.flush()
    return objective(params)

t0 = time.time()
result_de = differential_evolution(
    objective_verbose,
    bounds,
    seed=42,
    maxiter=200,
    popsize=20,
    tol=1e-12,
    mutation=(0.5, 1.5),
    recombination=0.9,
    polish=True,
    workers=1,
)
t_de = time.time() - t0

g0_de, g1_de, g2_de, k1_de, k2_de = result_de.x
g3_de = 1.0 - g0_de - g1_de - g2_de
k3_de = 1.0 / (1.0 - 1.0/k1_de - 1.0/k2_de)

print("  Time: %.1f s" % t_de)
print("  Objective: %.8f  (Su: %.8f, ratio: %.4f)" % (
    result_de.fun, su_obj, result_de.fun / max(su_obj, 1e-15)))
print("  gamma = [%.4f, %.4f, %.4f, %.4f]  sum=%.6f" % (
    g0_de, g1_de, g2_de, g3_de, g0_de+g1_de+g2_de+g3_de))
print("  k     = [%.4f, %.4f, %.4f]" % (k1_de, k2_de, k3_de))
print("  1/k1+1/k2+1/k3 = %.6f  (should be 1.0)" % (
    1/k1_de + 1/k2_de + 1/k3_de))


# ── Strategy 2: Multi-start SLSQP (local, starting near Su and near DE) ──────
print("\n" + "="*65)
print("Strategy 2: Multi-start SLSQP (constrained local search)")
print("="*65)
sys.stdout.flush()

# Parameterise with free k3: [g0, g1, g2, k1, k2, k3]
# Constraints: g0+g1+g2+g3=1 (g3 free), 1/k1+1/k2+1/k3=1

def obj_slsqp(p):
    g0, g1, g2, k1, k2, k3 = p
    g3 = 1.0 - g0 - g1 - g2
    if g3 < 0.005 or any(v < 0.005 for v in [g0,g1,g2]):
        return 1e10
    if any(v < 0.1 for v in [k1, k2, k3]):
        return 1e10

    gamma = [g0, g1, g2, g3]
    k     = [k1, k2, k3]
    phi   = math.pi / 2

    err = 0.0
    for name, xi_arr, beta_arr, Qx_arr, Qy_arr, weight in datasets:
        for i in range(len(xi_arr)):
            try:
                Qx_p, Qy_p = prb_statics(gamma, k, xi_arr[i], phi, beta_arr[i])
                err += weight * ((Qx_p - Qx_arr[i])**2 + (Qy_p - Qy_arr[i])**2)
            except Exception:
                err += weight * 1.0
    return err

constraints_slsqp = [
    {"type": "eq", "fun": lambda p: p[0]+p[1]+p[2]+(1-p[0]-p[1]-p[2]) - 1.0},  # trivially satisfied
    {"type": "eq", "fun": lambda p: 1.0/p[3] + 1.0/p[4] + 1.0/p[5] - 1.0},     # compliance
]
bounds_slsqp = [
    (0.01, 0.30), (0.15, 0.60), (0.20, 0.60),   # g0, g1, g2
    (1.5, 15.0),  (1.5, 10.0),  (1.5, 10.0),     # k1, k2, k3
]

starts = [
    [0.10, 0.35, 0.40, 3.51, 2.99, 2.58],                       # Su (2009)
    [g0_de, g1_de, g2_de, k1_de, k2_de, k3_de],                  # DE result
    [0.08, 0.30, 0.45, 3.80, 3.00, 2.50],                        # manual guess 1
    [0.12, 0.38, 0.35, 3.40, 3.10, 2.70],                        # manual guess 2
    [0.05, 0.35, 0.45, 4.00, 2.80, 2.60],                        # manual guess 3
    [0.15, 0.30, 0.35, 3.60, 3.20, 2.40],                        # manual guess 4
]

best_slsqp = None
best_obj   = 1e20

for i, x0 in enumerate(starts):
    res = minimize(obj_slsqp, x0, method="SLSQP",
                   bounds=bounds_slsqp, constraints=constraints_slsqp,
                   options={"maxiter": 500, "ftol": 1e-14})
    if res.fun < best_obj:
        best_obj   = res.fun
        best_slsqp = res
    print("  Start %d: obj=%.8f  success=%s" % (i+1, res.fun, res.success))

g0_s, g1_s, g2_s, k1_s, k2_s, k3_s = best_slsqp.x
g3_s = 1.0 - g0_s - g1_s - g2_s

print("\n  Best SLSQP objective: %.8f  (Su: %.8f, ratio: %.4f)" % (
    best_obj, su_obj, best_obj / max(su_obj, 1e-15)))
print("  gamma = [%.4f, %.4f, %.4f, %.4f]  sum=%.6f" % (
    g0_s, g1_s, g2_s, g3_s, g0_s+g1_s+g2_s+g3_s))
print("  k     = [%.4f, %.4f, %.4f]" % (k1_s, k2_s, k3_s))
print("  1/k1+1/k2+1/k3 = %.6f" % (1/k1_s + 1/k2_s + 1/k3_s))


# ── Strategy 3: Hybrid — DE with compliance constraint built in ───────────────
print("\n" + "="*65)
print("Strategy 3: DE with expanded bounds + compliance")
print("="*65)
sys.stdout.flush()

bounds_wide = [
    (0.01, 0.25),
    (0.10, 0.55),
    (0.20, 0.65),
    (1.5,  15.0),
    (1.5,  10.0),
]

call_count[0] = 0
t0 = time.time()
result_de2 = differential_evolution(
    objective_verbose,
    bounds_wide,
    seed=123,
    maxiter=300,
    popsize=25,
    tol=1e-13,
    mutation=(0.4, 1.8),
    recombination=0.85,
    polish=True,
    workers=1,
)
t_de2 = time.time() - t0

g0_d2, g1_d2, g2_d2, k1_d2, k2_d2 = result_de2.x
g3_d2 = 1.0 - g0_d2 - g1_d2 - g2_d2
k3_d2 = 1.0 / (1.0 - 1.0/k1_d2 - 1.0/k2_d2)

print("  Time: %.1f s" % t_de2)
print("  Objective: %.8f  (Su: %.8f, ratio: %.4f)" % (
    result_de2.fun, su_obj, result_de2.fun / max(su_obj, 1e-15)))
print("  gamma = [%.4f, %.4f, %.4f, %.4f]  sum=%.6f" % (
    g0_d2, g1_d2, g2_d2, g3_d2, g0_d2+g1_d2+g2_d2+g3_d2))
print("  k     = [%.4f, %.4f, %.4f]" % (k1_d2, k2_d2, k3_d2))
print("  1/k1+1/k2+1/k3 = %.6f" % (1/k1_d2 + 1/k2_d2 + 1/k3_d2))


# ── Pick overall best ────────────────────────────────────────────────────────
candidates = [
    ("Su (2009)",     [0.10, 0.35, 0.40, 0.15], [3.51, 2.99, 2.58], su_obj),
    ("DE strategy 1", [g0_de, g1_de, g2_de, g3_de], [k1_de, k2_de, k3_de], result_de.fun),
    ("SLSQP best",    [g0_s,  g1_s,  g2_s,  g3_s],  [k1_s,  k2_s,  k3_s],  best_obj),
    ("DE strategy 3", [g0_d2, g1_d2, g2_d2, g3_d2], [k1_d2, k2_d2, k3_d2], result_de2.fun),
]

print("\n" + "="*65)
print("FINAL COMPARISON")
print("="*65)
print("%-16s  %-28s  %-22s  %s" % ("Model", "gamma", "k", "Objective"))
print("-"*95)
for name, gamma, k, obj in sorted(candidates, key=lambda c: c[3]):
    gstr = "[%.3f, %.3f, %.3f, %.3f]" % tuple(gamma)
    kstr = "[%.3f, %.3f, %.3f]" % tuple(k)
    tag  = " <-- BEST" if obj == min(c[3] for c in candidates) else ""
    print("%-16s  %-28s  %-22s  %.8f%s" % (name, gstr, kstr, obj, tag))

# Identify the overall best
best_name, best_gamma, best_k, _ = min(candidates, key=lambda c: c[3])
print("\nBest model: %s" % best_name)
print("  gamma = %s" % best_gamma)
print("  k     = %s" % best_k)
print("  compliance: 1/k1+1/k2+1/k3 = %.6f" % (1/best_k[0]+1/best_k[1]+1/best_k[2]))

# ── Detailed error comparison at specific load points ─────────────────────────
print("\n" + "="*65)
print("ERROR COMPARISON AT KEY DEFLECTIONS")
print("="*65)

test_cases = []
# Pure moment
for th in [0.1, 0.3, 0.5, 0.7, 1.0, 1.3]:
    test_cases.append(("M beta=%.1f" % th, 0.0, th, math.sin(th)/th, (1-math.cos(th))/th))
# Pure force
for xi_v in [0.05, 0.2, 0.5, 1.0]:
    xi_f_arr, _, Qx_f_arr, Qy_f_arr = pure_force_data(1)
    # Recompute for this specific xi — use rough interpolation from dataset
    xi_all, _, Qx_all, Qy_all = pure_force_data(200)
    idx_closest = np.argmin(np.abs(xi_all - xi_v))
    test_cases.append(("F xi=%.2f" % xi_v, xi_all[idx_closest], 0.0,
                        Qx_all[idx_closest], Qy_all[idx_closest]))

print("%-14s  %-8s  %-8s  |" % ("Case", "Qx_ex", "Qy_ex"), end="")
for name, _, _, _ in candidates:
    print("  %s errQx/Qy" % name[:8], end="")
print()
print("-"*130)

phi_val = math.pi/2
for case_name, xi_v, beta_v, Qx_ex, Qy_ex in test_cases:
    print("%-14s  %8.4f  %8.4f  |" % (case_name, Qx_ex, Qy_ex), end="")
    for name, gamma, k, _ in candidates:
        Qx_p, Qy_p = prb_statics(gamma, k, xi_v, phi_val, beta_v)
        ex = abs(Qx_p - Qx_ex) / max(abs(Qx_ex), 1e-9) * 100
        ey = abs(Qy_p - Qy_ex) / max(abs(Qy_ex), 1e-9) * 100
        print("  %5.2f%%/%5.2f%%" % (ex, ey), end="")
    print()

print("\n=== Optimization complete ===")
