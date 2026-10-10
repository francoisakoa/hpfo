#import os
#os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
#os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
from pyomo.environ import (
    ConcreteModel, Var, Objective, RangeSet, Reals, minimize, value
)
from pyomo.opt import SolverFactory
import warnings
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp

from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, RangeSet, SolverFactory, minimize
from typing import Tuple, Sequence
# === NOUVELLES IMPORTATIONS ===
import pyomo.environ as pyo

import logging

logging.getLogger('pyomo.core').setLevel(logging.ERROR)
    
# === Décompositions DC du potentiel de Lennard-Jones ===
def get_dc_decomposition(n, d, version=None):
    def f(x):
        X = x.reshape((n, d))
        E = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                r = (jnp.linalg.norm(X[i] - X[j])**2 + 1e-8)**0.5
                E += 1.0 / r**12
                
        return E

    def g(x):
        X = x.reshape((n, d))
        G = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                r = (jnp.linalg.norm(X[i] - X[j])**2 + 1e-8)**0.5
                G += 2.0 / r**6
                
        return G

    return f, g, jax.grad(g)

# === Adaptation de solver_lj_cyipopt pour résoudre le sous-problème DC ===
def solver_lj_cvx_dc(n, d, lb, ub, x0, seed=0, tol=1e-8, maxit=5000, disp=0, dc_version=None, alpha=1e-4):
    N = n * d
    xk = np.array(x0)
    f_fn, g_fn, grad_g_fn = get_dc_decomposition(n, d, dc_version)
    grad_g_xk = np.array(grad_g_fn(jnp.array(xk)))

    model = pyo.ConcreteModel()
    model.N = pyo.RangeSet(0, N - 1)
    model.x = pyo.Var(model.N, domain=pyo.Reals, initialize={i: float(xk[i]) for i in range(N)})

    def f_expr_pyomo():
        expr = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                r2 = sum((model.x[i*d + k] - model.x[j*d + k])**2 for k in range(d)) + 1e-8
                r = (r2 + 1e-8) ** 0.5
                expr += 1.0 / r ** 12 + 2.0 / r ** 6
                
        return expr

    dot_g = pyo.quicksum(grad_g_xk[i] * model.x[i] for i in range(N))
    x_norm_sq = pyo.quicksum(model.x[i]**2 for i in range(N))
    x_dot_xk = pyo.quicksum(model.x[i] * xk[i] for i in range(N))
    quad = alpha * (x_norm_sq - 2 * x_dot_xk)
    # removed: x_dot_xk now part of quad term

    model.obj = pyo.Objective(expr=f_expr_pyomo() - dot_g + quad, sense=pyo.minimize)

    for i in range(N):
        model.x[i].setlb(lb[i])
        model.x[i].setub(ub[i])

    solver = pyo.SolverFactory('ipopt')
    solver.options['tol'] = tol
    solver.options['max_iter'] = maxit
    solver.options['print_level'] = disp

    try:
        results = solver.solve(model, tee=False)
        if results.solver.termination_condition not in ['optimal', 'acceptable']:
            print("Warning: IPOPT did not find an optimal solution. Termination condition:",
                  results.solver.termination_condition)
            x_opt = np.array([value(model.x[i]) for i in model.N])
        else:
            x_opt = np.array([value(model.x[i]) for i in model.N])
        obj_opt = value(model.obj)
        return x_opt, obj_opt
    except Exception as e:
        print("Warning: Exception in solver_lj_cyipopt (DC):", e)
        fallback_obj = float(f_fn(jnp.array(x0)) - g_fn(jnp.array(x0)))
        return x0, fallback_obj


# ─────────────────────────────────────────────────────────────────────────────
# 🔧 CYIPOPT‑STYLE SOLVER WITH PER‑ATOM SPHERICAL CONSTRAINT
# ─────────────────────────────────────────────────────────────────────────────

def solver_lj_cyipopt(n: int, d: int, lb: Sequence[float] | None, 
                      ub: Sequence[float] | None, x0: Sequence[float], *, 
                      radius: float | None = None, tol: float = 1e-6, maxit: int = 5000, 
                      disp: int = 0, seed:int=237, verbose=False) -> tuple[np.ndarray, float]:
    x0 = np.asarray(x0, dtype=float)
    N = n * d
    assert x0.size == N
    lb = np.full(N, -np.inf) if lb is None else np.asarray(lb, dtype=float)
    ub = np.full(N,  np.inf) if ub is None else np.asarray(ub, dtype=float)
    model = ConcreteModel()
    model.N = RangeSet(0, N-1)
    model.x = Var(model.N, bounds=lambda m,i: (float(lb[i]), float(ub[i])), initialize=lambda m,i: float(x0[i]))
    eps = 1e-8
    def obj_rule(m):
        e=0.0
        for i in range(n):
            for j in range(i+1, n):
                r2 = sum((m.x[i*d+k]-m.x[j*d+k])**2 for k in range(d)) + eps
                inv_r6 = r2**-3
                e += inv_r6**2 - 2*inv_r6
        return e
    model.obj = Objective(rule=obj_rule, sense=minimize)
    if radius is not None and np.isfinite(radius):
        r2_limit = radius**2
        model.ball = ConstraintList()
        for atom in range(n):
            model.ball.add(sum(model.x[atom*d + k]**2 for k in range(d)) <= r2_limit)
    solver = SolverFactory('ipopt')
    solver.options.update({'tol': tol, 'max_iter': maxit, 'print_level': disp})
    try:
        res = solver.solve(model, tee=False)
        status = str(res.solver.termination_condition).lower()
# =============================================================================
#         if verbose and 'optimal' not in status and 'acceptable' not in status:
#             print('[Warning] IPOPT status:', res.solver.termination_condition)
# =============================================================================
        x_opt = np.array([pyo.value(model.x[i]) for i in model.N])
        return x_opt, 0.0 #pyo.value(model.obj)
    except Exception as exc:
        #print('[Warning] Exception in solver_lj_cyipopt:', exc)
        if verbose:
          print("IPOPT failed, switching to L-BFGS-B: ", exc)
# =============================================================================
#         try:
#             return solver_lj_lbfgsb(n, d, lb, ub, x0, tol=tol, maxit=maxit)
#         except Exception as ee:
#             print("L-BFGS-B also failed, switching to Numba fallback: ", ee)
# =============================================================================

        fallback = 0.0
# =============================================================================
#         for i in range(n):
#             for j in range(i+1, n):
#                 r2 = np.sum((x0[i*d:i*d+d]-x0[j*d:j*d+d])**2) + eps
#                 inv_r6 = r2**-3
#                 fallback += inv_r6**2 - 2*inv_r6
# =============================================================================
        return x0.copy(), fallback


def solver_lj_cyipopt_no_ball_constraints(n, d, lb, ub, x0, seed=0, tol=1e-8, maxit=5000, disp=0, radius=None):
    """
    Solve the Lennard–Jones minimization problem for n atoms in d dimensions
    using Pyomo and IPOPT.
    
    If the solver fails, a warning is printed and the initial solution (x0)
    along with its objective value is returned.
    """
    N = n * d
    eps = 1e-8

# =============================================================================
#     lb = -np.inf * np.ones(n * d)
#     ub = np.inf * np.ones(n * d)
#        
# =============================================================================
    # Build the Pyomo model.
    model = ConcreteModel()
    model.N = RangeSet(0, N - 1)
    
    def var_bounds(m, i):
        return (lb[i], ub[i])
    
    # Use a dictionary for variable initialization with x0.
    init_dict = {i: float(x0[i]) for i in range(N)}
    model.x = Var(model.N, domain=Reals, bounds=var_bounds, initialize=init_dict)
    
    # Define the Lennard–Jones objective function.
    def objective_rule(m):
        obj = 0.0
        for i in range(n):
            for j in range(i+1, n):
                r2 = sum((m.x[i*d + k] - m.x[j*d + k])**2 for k in range(d))
                r = (r2 + eps)**0.5
                obj += 1/(r**12) - 2/(r**6)
        return obj
    model.obj = Objective(rule=objective_rule, sense=minimize)
    
    # Create the IPOPT solver.
    opt = SolverFactory('ipopt')
    opt.options['tol'] = tol
    opt.options['max_iter'] = maxit
    opt.options['print_level'] = disp
    # Optional: enable warm start to help with convergence.
    #opt.options['warm_start_init_point'] = 'yes'
    
    try:
        results = opt.solve(model, tee=False)
        if results.solver.termination_condition not in ['optimal', 'acceptable']:
            print("Warning: IPOPT did not find an optimal solution. Termination condition:",
                  results.solver.termination_condition)
            # Attempt to extract the current iterate regardless
            x_opt = np.array([value(model.x[i]) for i in model.N])
        else:
            x_opt = np.array([value(model.x[i]) for i in model.N])
        obj_opt = value(model.obj)
        return x_opt, obj_opt
    except Exception as e:
        print("IPOPT failed, switching to L-BFGS-B: ", e)
        try:
            return solver_lj_lbfgsb(n, d, lb, ub, x0, tol=tol, maxit=maxit)
        except Exception as ee:
            print("L-BFGS-B also failed, switching to Numba fallback: ", ee)

        # Compute the objective value for x0 manually as a fallback.
        fallback_obj = 0.0
        for i in range(n):
            for j in range(i+1, n):
                r2 = sum((x0[i*d + k] - x0[j*d + k])**2 for k in range(d))
                r = (r2 + eps)**0.5
                fallback_obj += 1/(r**12) - 2/(r**6)
        return x0, fallback_obj

from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, RangeSet, Reals, minimize, value
import numpy as np
from numba import njit
from scipy.optimize import minimize as scipy_minimize

@njit(fastmath=True)
def lj_potential_numba(x, n, d, eps=1e-8):
    obj = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r2 = 0.0
            for k in range(d):
                diff = x[i * d + k] - x[j * d + k]
                r2 += diff * diff
            r = (r2 + eps)**0.5
            obj += 1.0 / r**12 - 2.0 / r**6
    return obj
from numba import njit, prange
import numpy as np


@njit(fastmath=True, parallel=False)
def lj_potential_numba___(x, n, d, eps=1e-8):
    X = x.reshape((n, d))
    partial_sums = np.zeros(n)

    for i in prange(n - 1):
        s = 0.0
        for j in range(i + 1, n):
            r2 = 0.0
            for k in range(d):
                diff = X[i, k] - X[j, k]
                r2 += diff * diff
            r = (r2 + eps) ** 0.5
            s += 1.0 / r**12 - 2.0 / r**6
        partial_sums[i] = s

    return np.sum(partial_sums)

def solver_lj_lbfgsb(n, d, lb, ub, x0, tol=1e-8, maxit=5000):
    bounds = [(lb[i], ub[i]) for i in range(n * d)]
    fun = lambda x: lj_potential_numba(x, n, d)
    res = scipy_minimize(fun, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': maxit, 'ftol': tol, 'disp': False})
    return res.x, res.fun

def solver_lj_cyipopt__(n, d, lb, ub, x0, seed=1240, tol=1e-8, maxit=5000, disp=0, use_lbfgsb=False):
    """
    Solve the Lennard–Jones minimization problem for n atoms in d dimensions
    using Pyomo and IPOPT. Falls back to L-BFGS-B or Numba if IPOPT fails.
    """
    # Using LBFGS-B
    if use_lbfgsb:
        return solver_lj_lbfgsb(n, d, lb, ub, x0, tol=tol, maxit=maxit)

    # Using Ipopt
    N = n * d
    eps = 1e-8
    model = ConcreteModel()
    model.N = RangeSet(0, N - 1)

    def var_bounds(m, i):
        return (lb[i], ub[i])

    init_dict = {i: float(x0[i]) for i in range(N)}
    model.x = Var(model.N, domain=Reals, bounds=var_bounds, initialize=init_dict)

    def objective_rule(m):
        expr = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                r2 = sum((m.x[i * d + k] - m.x[j * d + k])**2 for k in range(d)) + eps
                r = r2 ** 0.5
                expr += 1.0 / r**12 - 2.0 / r**6
        return expr

    model.obj = Objective(rule=objective_rule, sense=minimize)

    opt = SolverFactory('ipopt')
    opt.options['tol'] = tol
    opt.options['max_iter'] = maxit
    opt.options['print_level'] = disp
    #opt.options['linear_solver'] = "mumps"

    try:
        results = opt.solve(model, tee=False)
        x_opt = np.array([value(model.x[i]) for i in model.N])
        obj_opt = value(model.obj)

        if (not np.all(np.isfinite(x_opt)) or
            not np.isfinite(obj_opt) or
            np.isnan(obj_opt) or
            np.abs(obj_opt) > 1e10):
            raise ValueError("Solution numériquement instable ou invalide.")

        if results.solver.termination_condition not in ['optimal', 'acceptable']:
            raise ValueError("IPOPT failed: " + str(results.solver.termination_condition))

        return x_opt, obj_opt

    except Exception as e:
        print("IPOPT failed, switching to L-BFGS-B: ", e)
        try:
            return solver_lj_lbfgsb(n, d, lb, ub, x0, tol=tol, maxit=maxit)
        except Exception as ee:
            print("L-BFGS-B also failed, switching to Numba fallback: ", ee)
            fallback_obj = lj_potential_numba(np.asarray(x0), n, d, eps)
            return x0, fallback_obj
