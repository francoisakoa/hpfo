#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
solve_nlp_local_gpu.py - Raffinement local GPU via JAX

Ce module remplace IPOPT par un raffinement gradient-based sur GPU.
Utilise JAX pour l'autodiff et optionnellement jaxopt pour L-BFGS.

API unifiée :
    local_refine(x0, lb, ub, model_params, cfg) -> (x_opt, obj_opt)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
import warnings

# === Configuration JAX ===
# Note: jax_enable_x64 est contrôlé par cfg["jax_x64"]

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from functools import partial

# Tentative d'import jaxopt (optionnel, pour L-BFGS)
_JAXOPT_AVAILABLE = False
try:
    import jaxopt
    from jaxopt import ProjectedGradient, LBFGS, LBFGSB
    _JAXOPT_AVAILABLE = True
except ImportError:
    jaxopt = None

# IPOPT via cyipopt (optionnel)
_IPOPT_AVAILABLE = False
cyipopt = None
try:
    import cyipopt
    _IPOPT_AVAILABLE = True
except ImportError:
    pass


# ==============================================================================
# Potentiel de Lennard-Jones (JAX)
# ==============================================================================

def lj_energy_jax(x: jnp.ndarray, n: int, d: int, eps: float = 1e-8) -> float:
    """
    Calcule le potentiel de Lennard-Jones en JAX (différentiable).
    
    Note: Cette fonction n'est PAS JIT-compilée directement car n et d sont dynamiques.
    Utilisez make_lj_objective() pour obtenir une version JIT-compilée.
    
    Args:
        x: Vecteur de positions (n*d,)
        n: Nombre d'atomes
        d: Dimension spatiale
        eps: Régularisation pour éviter division par zéro
        
    Returns:
        Énergie LJ (scalaire)
    """
    X = x.reshape((n, d))
    
    # Calcul vectorisé des distances pairwise
    diff = X[:, None, :] - X[None, :, :]  # (n, n, d)
    
    # r2[i,j] = ||X[i] - X[j]||^2
    r2 = jnp.sum(diff ** 2, axis=-1) + eps  # (n, n)
    
    # Masque pour i < j (triangle supérieur)
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    
    # inv_r6 = 1/r^6
    inv_r6 = r2 ** (-3)
    
    # Énergie LJ: sum_{i<j} (1/r^12 - 2/r^6)
    energy = jnp.sum(jnp.where(mask, inv_r6 ** 2 - 2 * inv_r6, 0.0))
    
    return energy


def make_lj_objective(n: int, d: int, eps: float = 1e-8):
    """
    Crée une fonction objective JIT-compilée pour n atomes en d dimensions.
    
    Args:
        n: Nombre d'atomes (fixé)
        d: Dimension spatiale (fixé)
        eps: Régularisation
        
    Returns:
        Fonction objective JIT-compilée: f(x) -> energy
    """
    @jit
    def objective(x):
        X = x.reshape((n, d))
        diff = X[:, None, :] - X[None, :, :]
        r2 = jnp.sum(diff ** 2, axis=-1) + eps
        mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
        inv_r6 = r2 ** (-3)
        return jnp.sum(jnp.where(mask, inv_r6 ** 2 - 2 * inv_r6, 0.0))
    
    return objective


def lj_energy_numpy(x: np.ndarray, n: int, d: int, eps: float = 1e-8) -> float:
    """Version NumPy du potentiel LJ (pour fallback CPU)."""
    X = x.reshape((n, d))
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r2 = np.sum((X[i] - X[j]) ** 2) + eps
            inv_r6 = r2 ** (-3)
            energy += inv_r6 ** 2 - 2 * inv_r6
    return float(energy)


# ==============================================================================
# Projection sur les bornes (clipping)
# ==============================================================================

@jit
def project_bounds(x: jnp.ndarray, lb: jnp.ndarray, ub: jnp.ndarray) -> jnp.ndarray:
    """Projette x sur [lb, ub]."""
    return jnp.clip(x, lb, ub)


# ==============================================================================
# Raffinement via Gradient Descent projeté (fallback si jaxopt non dispo)
# ==============================================================================

def projected_gradient_descent(
    f: Callable,
    x0: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[jnp.ndarray, float]:
    """
    Gradient descent projeté simple.
    
    Args:
        f: Fonction objective (JAX)
        x0: Point initial
        lb, ub: Bornes
        lr: Learning rate
        max_iter: Iterations max
        tol: Tolérance convergence
        verbose: Afficher progression
        
    Returns:
        (x_opt, f_opt)
    """
    grad_f = jit(grad(f))
    
    x = x0.copy()
    f_prev = float('inf')
    
    for i in range(max_iter):
        f_val = f(x)
        g = grad_f(x)
        
        # Gradient step
        x_new = x - lr * g
        
        # Projection
        x_new = project_bounds(x_new, lb, ub)
        
        # Adaptive learning rate (simple backtracking)
        f_new = f(x_new)
        if f_new > f_val:
            lr *= 0.5
            continue
        
        # Convergence check
        if abs(f_prev - f_new) < tol:
            if verbose:
                print(f"[PGD] Converged at iter {i+1}, f={f_new:.8f}")
            break
        
        x = x_new
        f_prev = f_new
        
        if verbose and i % 100 == 0:
            print(f"[PGD] Iter {i+1}: f={f_new:.8f}, |g|={jnp.linalg.norm(g):.6f}")
    
    return x, float(f(x))


# ==============================================================================
# Raffinement via L-BFGS-B (jaxopt)
# ==============================================================================

def lbfgsb_refine(
    f: Callable,
    x0: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[jnp.ndarray, float]:
    """
    Raffinement via L-BFGS-B (jaxopt).
    
    Args:
        f: Fonction objective (JAX)
        x0: Point initial
        lb, ub: Bornes
        max_iter: Iterations max
        tol: Tolérance convergence
        verbose: Afficher progression
        
    Returns:
        (x_opt, f_opt)
    """
    if not _JAXOPT_AVAILABLE:
        raise ImportError("jaxopt non disponible, utiliser projected_gradient_descent")
    
    # Définir les bornes pour LBFGSB
    bounds = (lb, ub)
    
    # Créer le solver
    solver = LBFGSB(
        fun=f,
        maxiter=max_iter,
        tol=tol,
        verbose=verbose
    )
    
    # Résoudre
    result = solver.run(x0, bounds=bounds)
    x_opt = result.params
    f_opt = float(f(x_opt))
    
    return x_opt, f_opt


def lbfgs_with_projection(
    f: Callable,
    x0: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[jnp.ndarray, float]:
    """
    L-BFGS avec projection post-hoc (alternative si LBFGSB pose problème).
    """
    if not _JAXOPT_AVAILABLE:
        raise ImportError("jaxopt non disponible")
    
    # Fonction avec pénalité pour les bornes
    def f_penalized(x):
        # Pénalité quadratique pour violation des bornes
        penalty_lb = jnp.sum(jnp.maximum(0, lb - x) ** 2)
        penalty_ub = jnp.sum(jnp.maximum(0, x - ub) ** 2)
        return f(x) + 1e6 * (penalty_lb + penalty_ub)
    
    solver = LBFGS(
        fun=f_penalized,
        maxiter=max_iter,
        tol=tol,
    )
    
    result = solver.run(x0)
    x_opt = project_bounds(result.params, lb, ub)
    f_opt = float(f(x_opt))
    
    return x_opt, f_opt


# ==============================================================================
# Raffinement via Adam (alternative robuste)
# ==============================================================================

def adam_refine(
    f: Callable,
    x0: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    lr: float = 0.01,
    max_iter: int = 2000,
    tol: float = 1e-7,
    verbose: bool = False
) -> Tuple[jnp.ndarray, float]:
    """
    Raffinement via Adam avec projection.
    """
    grad_f = jit(grad(f))
    
    # Adam parameters
    beta1, beta2 = 0.9, 0.999
    eps_adam = 1e-8
    
    x = x0.copy()
    m = jnp.zeros_like(x)  # First moment
    v = jnp.zeros_like(x)  # Second moment
    
    f_prev = float('inf')
    
    for i in range(max_iter):
        g = grad_f(x)
        
        # Update moments
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))
        
        # Update
        x = x - lr * m_hat / (jnp.sqrt(v_hat) + eps_adam)
        
        # Projection
        x = project_bounds(x, lb, ub)
        
        f_val = float(f(x))
        
        # Convergence check
        if abs(f_prev - f_val) < tol and i > 100:
            if verbose:
                print(f"[Adam] Converged at iter {i+1}, f={f_val:.8f}")
            break
        
        f_prev = f_val
        
        if verbose and i % 200 == 0:
            print(f"[Adam] Iter {i+1}: f={f_val:.8f}, |g|={jnp.linalg.norm(g):.6f}")
    
    return x, float(f(x))


# ==============================================================================
# Raffinement via Pyomo/IPOPT (CPU) - Version utilisateur
# ==============================================================================

# Numba-accelerated LJ potential (pour fallback L-BFGS-B)
try:
    from numba import njit
    
    @njit(fastmath=True)
    def _lj_potential_numba(x, n, d, eps=1e-8):
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
    
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    
    def _lj_potential_numba(x, n, d, eps=1e-8):
        """Fallback sans Numba."""
        return lj_energy_numpy(x, n, d)


def _solver_lj_lbfgsb(n, d, lb, ub, x0, tol=1e-8, maxit=5000):
    """Solver L-BFGS-B avec scipy (rapide, utilise Numba si dispo)."""
    from scipy.optimize import minimize as scipy_minimize
    
    bounds = [(lb[i], ub[i]) for i in range(n * d)]
    fun = lambda x: _lj_potential_numba(x, n, d)
    res = scipy_minimize(fun, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': maxit, 'ftol': tol, 'disp': False})
    return res.x, res.fun


def _solver_lj_cyipopt(n, d, lb, ub, x0, tol=1e-8, maxit=5000, disp=0, radius=None, verbose=False):
    """
    Solve the Lennard-Jones minimization problem for n atoms in d dimensions
    using Pyomo and IPOPT. Falls back to L-BFGS-B if IPOPT fails.
    
    Version originale de l'utilisateur.
    """
    from pyomo.environ import (
        ConcreteModel, Var, Objective, ConstraintList, RangeSet, Reals, 
        minimize, value
    )
    from pyomo.opt import SolverFactory
    import pyomo.environ as pyo
    
    x0 = np.asarray(x0, dtype=float)
    N = n * d
    assert x0.size == N
    lb = np.full(N, -np.inf) if lb is None else np.asarray(lb, dtype=float)
    ub = np.full(N,  np.inf) if ub is None else np.asarray(ub, dtype=float)
    
    model = ConcreteModel()
    model.N = RangeSet(0, N-1)
    model.x = Var(model.N, bounds=lambda m,i: (float(lb[i]), float(ub[i])), 
                  initialize=lambda m,i: float(x0[i]))
    
    eps = 1e-8
    
    def obj_rule(m):
        e = 0.0
        for i in range(n):
            for j in range(i+1, n):
                r2 = sum((m.x[i*d+k] - m.x[j*d+k])**2 for k in range(d)) + eps
                inv_r6 = r2**-3
                e += inv_r6**2 - 2*inv_r6
        return e
    
    model.obj = Objective(rule=obj_rule, sense=minimize)
    
    # Contrainte sphérique optionnelle par atome
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
        x_opt = np.array([pyo.value(model.x[i]) for i in model.N])
        obj_opt = pyo.value(model.obj)
        return x_opt, obj_opt
    except Exception as exc:
        if verbose:
            print(f"[IPOPT] Failed ({exc}), switching to L-BFGS-B")
        # Fallback L-BFGS-B
        return _solver_lj_lbfgsb(n, d, lb, ub, x0, tol=tol, maxit=maxit)


def _ipopt_refine(
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    n: int,
    d: int,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Raffinement via Pyomo/IPOPT (CPU) avec fallback L-BFGS-B + Numba.
    
    Utilise le solver Pyomo/IPOPT de l'utilisateur.
    
    Args:
        x0: Point initial
        lb, ub: Bornes
        n: Nombre d'atomes
        d: Dimension spatiale
        max_iter: Iterations max
        tol: Tolérance
        verbose: Afficher progression
        
    Returns:
        (x_opt, f_opt)
    """
    try:
        return _solver_lj_cyipopt(
            n, d, lb, ub, np.asarray(x0), 
            tol=tol, maxit=max_iter, disp=5 if verbose else 0, 
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"[IPOPT] Import/execution failed ({e}), using L-BFGS-B")
        return _solver_lj_lbfgsb(n, d, lb, ub, np.asarray(x0), tol=tol, maxit=max_iter)


# ==============================================================================
# Interface unifiée de raffinement local
# ==============================================================================

def get_default_refine_cfg() -> Dict:
    """Configuration par défaut pour le raffinement local."""
    return {
        "use_gpu": True,
        "method": "lbfgsb",  # "lbfgsb", "lbfgs", "adam", "pgd", "ipopt"
        "max_iter": 500,
        "tol": 1e-6,
        "lr": 0.01,  # Pour Adam/PGD
        "jax_x64": True,  # Précision float64
        "verbose": False,
    }


def local_refine(
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    model_params: Dict,
    cfg: Dict = None
) -> Tuple[np.ndarray, float]:
    """
    Raffinement local unifié (GPU via JAX ou CPU fallback).
    
    Args:
        x0: Point initial (numpy array)
        lb: Bornes inférieures
        ub: Bornes supérieures
        model_params: Dict avec 'n' (atomes) et 'd' (dimension)
        cfg: Configuration (voir get_default_refine_cfg())
        
    Returns:
        (x_opt, obj_opt): Solution raffinée et sa valeur objective
    """
    if cfg is None:
        cfg = get_default_refine_cfg()
    
    n = model_params.get('n', len(x0) // 3)
    d = model_params.get('d', 3)
    method = cfg.get("method", "lbfgsb")
    max_iter = cfg.get("max_iter", 500)
    tol = cfg.get("tol", 1e-6)
    lr = cfg.get("lr", 0.01)
    verbose = cfg.get("verbose", False)
    use_gpu = cfg.get("use_gpu", True)
    jax_x64 = cfg.get("jax_x64", True)
    
    # Configurer la précision JAX
    jax.config.update("jax_enable_x64", jax_x64)
    
    # === IPOPT/SCIPY PATH (CPU avec gradient JAX) ===
    if method == "ipopt":
        return _ipopt_refine(x0, lb, ub, n, d, max_iter, tol, verbose)
    
    # === JAX PATH (GPU) ===
    # Convertir en JAX arrays
    dtype = jnp.float64 if jax_x64 else jnp.float32
    x0_jax = jnp.array(x0, dtype=dtype)
    lb_jax = jnp.array(lb, dtype=dtype)
    ub_jax = jnp.array(ub, dtype=dtype)
    
    # Projeter x0 dans les bornes
    x0_jax = project_bounds(x0_jax, lb_jax, ub_jax)
    
    # Créer la fonction objective JIT-compilée
    objective = make_lj_objective(n, d)
    
    try:
        if method == "lbfgsb" and _JAXOPT_AVAILABLE:
            x_opt, f_opt = lbfgsb_refine(
                objective, x0_jax, lb_jax, ub_jax,
                max_iter=max_iter, tol=tol, verbose=verbose
            )
        elif method == "lbfgs" and _JAXOPT_AVAILABLE:
            x_opt, f_opt = lbfgs_with_projection(
                objective, x0_jax, lb_jax, ub_jax,
                max_iter=max_iter, tol=tol, verbose=verbose
            )
        elif method == "adam":
            x_opt, f_opt = adam_refine(
                objective, x0_jax, lb_jax, ub_jax,
                lr=lr, max_iter=max_iter, tol=tol, verbose=verbose
            )
        else:
            # Fallback: Projected Gradient Descent
            if verbose and method in ["lbfgsb", "lbfgs"]:
                print(f"[local_refine] jaxopt non dispo, fallback vers PGD")
            x_opt, f_opt = projected_gradient_descent(
                objective, x0_jax, lb_jax, ub_jax,
                lr=lr, max_iter=max_iter, tol=tol, verbose=verbose
            )
        
        # Convertir en numpy
        x_opt_np = np.array(x_opt)
        
        # Vérifier la validité
        if not np.all(np.isfinite(x_opt_np)) or not np.isfinite(f_opt):
            if verbose:
                print("[local_refine] Solution invalide, retour x0")
            return x0.copy(), lj_energy_numpy(x0, n, d)
        
        return x_opt_np, f_opt
        
    except Exception as e:
        if verbose:
            print(f"[local_refine] Exception: {e}, fallback numpy")
        # Fallback: retourner x0 avec son énergie
        return x0.copy(), lj_energy_numpy(x0, n, d)


# ==============================================================================
# Wrapper compatible avec l'ancienne interface IPOPT
# ==============================================================================

def local_refine_ipopt_compatible(
    n: int,
    d: int,
    lb: np.ndarray,
    ub: np.ndarray,
    x0: np.ndarray,
    radius: float = None,
    seed: int = 0,
    tol: float = 1e-6,
    maxit: int = 500,
    disp: int = 0,
    cfg: Dict = None
) -> Tuple[np.ndarray, float]:
    """
    Wrapper avec signature compatible solver_lj_cyipopt.
    
    Cette fonction peut être utilisée comme drop-in replacement.
    """
    if cfg is None:
        cfg = get_default_refine_cfg()
    
    cfg["max_iter"] = maxit
    cfg["tol"] = tol
    cfg["verbose"] = disp > 0
    
    model_params = {'n': n, 'd': d}
    
    # Appliquer la contrainte de rayon si spécifiée
    if radius is not None and np.isfinite(radius):
        # Projeter x0 dans la boule
        x0_reshaped = x0.reshape((n, d))
        norms = np.linalg.norm(x0_reshaped, axis=1)
        scale = np.minimum(1.0, radius / (norms + 1e-8))
        x0 = (x0_reshaped * scale[:, None]).flatten()
    
    return local_refine(x0, lb, ub, model_params, cfg)


# ==============================================================================
# Test du module
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU MODULE solve_nlp_local_gpu.py")
    print("=" * 60)
    
    print(f"\njaxopt disponible: {_JAXOPT_AVAILABLE}")
    
    # Test avec petit système
    np.random.seed(42)
    n_atoms, d = 7, 3
    num_vars = n_atoms * d
    
    lb = -3.0 * np.ones(num_vars)
    ub = 3.0 * np.ones(num_vars)
    x0 = np.random.uniform(-2, 2, size=num_vars)
    
    model_params = {'n': n_atoms, 'd': d}
    
    print(f"\nTest avec {n_atoms} atomes ({num_vars} variables)")
    print(f"Énergie initiale: {lj_energy_numpy(x0, n_atoms, d):.6f}")
    
    # Test différentes méthodes
    methods = ["pgd", "adam"]
    if _JAXOPT_AVAILABLE:
        methods.extend(["lbfgsb", "lbfgs"])
    
    for method in methods:
        print(f"\n--- Méthode: {method} ---")
        cfg = get_default_refine_cfg()
        cfg["method"] = method
        cfg["max_iter"] = 1000
        cfg["verbose"] = False
        
        import time
        t0 = time.time()
        x_opt, f_opt = local_refine(x0.copy(), lb, ub, model_params, cfg)
        elapsed = time.time() - t0
        
        print(f"  Énergie finale: {f_opt:.6f}")
        print(f"  Temps: {elapsed:.3f}s")
        
        # Vérifier que l'énergie recalculée correspond
        f_check = lj_energy_numpy(x_opt, n_atoms, d)
        print(f"  Vérification: {f_check:.6f} (diff={abs(f_opt - f_check):.2e})")
    
    # Test wrapper compatible IPOPT
    print("\n--- Test wrapper IPOPT-compatible ---")
    x_opt, f_opt = local_refine_ipopt_compatible(
        n=n_atoms, d=d, lb=lb, ub=ub, x0=x0.copy(),
        radius=3.0, tol=1e-6, maxit=500, disp=0
    )
    print(f"  Énergie: {f_opt:.6f}")
    
    print("\n" + "=" * 60)
    print("TESTS TERMINÉS")
    print("=" * 60)
