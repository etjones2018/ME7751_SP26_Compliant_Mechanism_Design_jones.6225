"""
ME 7751: Compliant Mechanism Design
Task-Specific PRB 3R Optimization

This script uses Differential Evolution to find the optimal segment lengths 
and spring stiffnesses for a 3-Revolute Pseudo-Rigid-Body model. Unlike global 
methods, it is constrained exclusively to the geometric trajectory of the 
target 4-bar linkage.
"""
import numpy as np
import time
from scipy.optimize import minimize_scalar, differential_evolution, minimize
from scipy.integrate import solve_bvp

# ── Linkage constants ─────────────────────────────────────────────────────────
SQ2 = np.sqrt(2)
B   = np.array([2 - SQ2, 1/SQ2])
r   = 1 - SQ2/2
AQu = -1/SQ2
AQv = 1/SQ2

def crank_A(psi):
    return B + r * np.array([-np.cos(psi), np.sin(psi)])

def Q_from(A, th0):
    dx = AQu*np.cos(th0) - AQv*np.sin(th0)
    dy = AQu*np.sin(th0) + AQv*np.cos(th0)
    return A - np.array([dx, dy])

# ── PRB Kinematics ────────────────────────────────────────────────────────────
def ik(gamma, Qx, Qy, th0, elbow="down"):
    g0, g1, g2, g3 = gamma
    px = Qx - g3*np.cos(th0) - g0
    py = Qy - g3*np.sin(th0)
    arg = (px**2 + py**2 - g1**2 - g2**2) / (2*g1*g2)
    if abs(arg) > 1.0: return None
    T2 = np.arccos(np.clip(arg, -1.0, 1.0))
    if elbow == "up": T2 = -T2
    d = g1**2 + g2**2 + 2*g1*g2*np.cos(T2)
    if d < 1e-12: return None
    T1 = np.arctan2(py*(g1 + g2*np.cos(T2)) - px*g2*np.sin(T2),
                    px*(g1 + g2*np.cos(T2)) + py*g2*np.sin(T2))
    T3 = th0 - T1 - T2
    return T1, T2, T3

def prb_solve(A, gamma, k):
    def energy(th0):
        Q = Q_from(A, th0)
        sol = ik(gamma, Q[0], Q[1], th0)
        if sol is None: return 1e10
        T1, T2, T3 = sol
        return 0.5 * (k[0]*T1**2 + k[1]*T2**2 + k[2]*T3**2)
    
    th_grid = np.linspace(-np.pi/4, np.pi/4, 30)
    E_grid  = np.array([energy(th) for th in th_grid])
    if not (E_grid < 1e9).any(): return None
    idx = np.argmin(E_grid)
    lo  = th_grid[max(0, idx - 2)]
    hi  = th_grid[min(len(th_grid)-1, idx + 2)]
    
    res = minimize_scalar(energy, bounds=(lo, hi), method='bounded', options={'xatol': 1e-7})
    if res.fun > 1e9: return None
    th0 = res.x
    Q = Q_from(A, th0)
    return Q

# ── Exact BVP Solver ──────────────────────────────────────────────────────────
def get_exact_Qs(psis):
    def fun(s, Y, p):
        Fx, Fy, th0 = p
        x, y, th, M = Y
        return np.vstack((np.cos(th), np.sin(th), M, Fx*np.sin(th) - Fy*np.cos(th)))

    exact_Qs = []
    s = np.linspace(0, 1, 30)
    Y = np.zeros((4, s.size)); Y[0] = s; Y[2] = 0.1*s; Y[3] = 0.1
    p = np.array([0.0, 0.5, 0.1])
    
    for psi in psis:
        A = crank_A(psi)
        def bc(ya, yb, p):
            Fx, Fy, th0 = p
            Qx = A[0] - (AQu*np.cos(th0) - AQv*np.sin(th0))
            Qy = A[1] - (AQu*np.sin(th0) + AQv*np.cos(th0))
            return np.array([ya[0], ya[1], ya[2], yb[0]-Qx, yb[1]-Qy, yb[2]-th0, yb[3]-((A[0]-Qx)*Fy-(A[1]-Qy)*Fx)])
        
        res = solve_bvp(fun, bc, s, Y, p=p, tol=1e-3, max_nodes=500)
        if res.success:
            Y = res.y; p = res.p; s = res.x
            exact_Qs.append(np.array([Y[0][-1], Y[1][-1]]))
        else:
            print("Failed BVP at psi =", psi)
            exact_Qs.append(None)
    return exact_Qs

# ── Optimization ──────────────────────────────────────────────────────────────
psis = np.linspace(0, 2*np.pi, 36, endpoint=False) # 36 frames
print("Pre-computing exact BVP tip positions...")
exact_Qs = get_exact_Qs(psis)

crank_As = [crank_A(psi) for psi in psis]

def objective(params):
    g0, g1, g2, k1, k2 = params
    g3 = 1.0 - g0 - g1 - g2
    if g3 < 0.01: return 1e10
    
    denom = 1.0 - 1.0/k1 - 1.0/k2
    if denom <= 0.01: return 1e10
    k3 = 1.0 / denom
    if k3 < 0.1 or k3 > 50: return 1e10
    
    gamma = [g0, g1, g2, g3]
    k     = [k1, k2, k3]
    
    err = 0.0
    for A, ex_Q in zip(crank_As, exact_Qs):
        if ex_Q is None: continue
        prb_Q = prb_solve(A, gamma, k)
        if prb_Q is None:
            err += 1.0 # severe penalty
        else:
            err += (prb_Q[0] - ex_Q[0])**2 + (prb_Q[1] - ex_Q[1])**2
    return err

su_params = [0.10, 0.35, 0.40, 3.51, 2.99]
su_err = objective(su_params)
print(f"Su (2009) SSE: {su_err:.8f}")

bounds = [(0.01, 0.30), (0.15, 0.60), (0.20, 0.60), (1.5, 12.0), (1.5, 8.0)]
eval_count = 0
def obj_verbose(p):
    global eval_count
    eval_count += 1
    if eval_count % 100 == 0:
        print(f"Eval {eval_count}...")
    return objective(p)

print("Starting task-based optimization via Differential Evolution...")
t0 = time.time()
res = differential_evolution(obj_verbose, bounds, maxiter=50, popsize=10, tol=1e-5, workers=1, mutation=(0.5, 1.5))
print(f"Optimization finished in {time.time()-t0:.1f} sec")
print("Best Params:", res.x)
print("Best SSE:", res.fun)

print("Refining locally...")
res_loc = minimize(objective, res.x, bounds=bounds, method='L-BFGS-B')
best_p = res_loc.x
g0, g1, g2, k1, k2 = best_p
g3 = 1.0 - g0 - g1 - g2
k3 = 1.0 / (1.0 - 1.0/k1 - 1.0/k2)
print("\nFinal Task-Optimal Parameters:")
print(f"  gamma = [{g0:.5f}, {g1:.5f}, {g2:.5f}, {g3:.5f}]")
print(f"  k     = [{k1:.5f}, {k2:.5f}, {k3:.5f}]")
print(f"  SSE   = {res_loc.fun:.8f}")
