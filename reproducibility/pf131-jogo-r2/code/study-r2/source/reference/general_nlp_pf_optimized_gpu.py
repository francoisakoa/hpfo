#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
general_nlp_pf_optimized_gpu.py - Version GPU du Particle Filter Optimizer

Migration GPU complète avec switch CPU/GPU configurable.
La logique métier est STRICTEMENT IDENTIQUE à la version originale.

Configuration backend via:
    backend_cfg = {
        "use_gpu": True,           # Master switch
        "gpu_clustering": True,    # KMeans/GMM/DBSCAN via cuML
        "gpu_pdist": True,         # pdist via CuPy
        "gpu_dtype": "float32",    # dtype pour GPU
        "pdist_chunk_size": 2000,  # Seuil pour chunking pdist
        "verbose": False,          # Logs backend
    }
"""

# === OPTIMISATION: Variables d'environnement (AVANT TOUT IMPORT) ===
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.60"  # ajuste 0.5–0.8
# JAX platform - sera configuré dynamiquement selon backend_cfg
# os.environ["JAX_PLATFORMS"] = "cpu"  # Commenté pour permettre GPU

import cupy as cp
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

import logging
logging.getLogger('jax._src.xla_bridge').setLevel(logging.ERROR)
# === FIN OPTIMISATION ===

import sys
import time
import warnings
import multiprocessing as mp

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm

# Import Pyomo (OPTIONNEL - seulement si IPOPT fallback nécessaire)
_PYOMO_AVAILABLE = False
try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
    from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, RangeSet, minimize
    from pyomo.opt import SolverStatus, TerminationCondition
    _PYOMO_AVAILABLE = True
except ImportError:
    pyo = None
    SolverFactory = None

from numba import njit, prange
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers, stax

# === IMPORT DU MODULE BACKEND GPU/CPU ===
from backend_ops import (
    cluster_fit_predict,
    pdist_like,
    squareform_like,
    compute_sharing_factors_backend,
    compute_diversity_backend,
    compute_distance_matrix,
    standardize_cluster_labels,
    print_backend_status,
    get_backend_info,
    DEFAULT_BACKEND_CFG,
    to_np,
    to_xp,
)

# === Raffinement local: GPU (JAX) + fallback optionnel CPU ===
try:
    from solve_nlp_local_gpu import local_refine, get_default_refine_cfg, make_lj_objective
except ImportError as e:
    raise ImportError(
        "Impossible d'importer solve_nlp_local_gpu.py (local_refine). "
        "Vérifiez que solve_nlp_local_gpu.py est présent."
    ) from e

from jax.experimental.ode import odeint

warnings.filterwarnings("ignore")
if _PYOMO_AVAILABLE:
    warnings.filterwarnings('ignore', category=UserWarning, module='pyomo')

# JAX config: une seule fois au niveau module (avant toute compilation)
jax.config.update("jax_enable_x64", False)

import pickle
import gzip
from pathlib import Path
from typing import Any, Optional, Dict

# ==============================================================================
# Configuration Backend par défaut
# ==============================================================================

def get_default_backend_cfg() -> Dict:
    """Retourne la configuration backend par défaut."""
    return {
        "use_gpu": True,
        "gpu_clustering": True,
        "gpu_pdist": True,
        "gpu_dtype": "float32",
        "pdist_chunk_size": 2000,
        "verbose": False,
    }


# ==============================================================================
# Fonctions utilitaires (identiques à l'original)
# ==============================================================================

def draw_hypersphere_candidates(rng: np.random.Generator, n: int, d: int, R: float, num_points: int = 1) -> np.ndarray:
    """
    Draw random candidate configurations uniformly inside a hypersphere of radius R in ℝ^{n×d}.
    """
    if n < 1 or d < 1 or R <= 0 or num_points < 1:
        raise ValueError("n, d, R and num_points must be positive")

    total_dim = n * d
    z = rng.normal(0, 1, size=(num_points, total_dim)).astype(np.float32)
    norms = np.linalg.norm(z, axis=1)
    norms = np.where(norms == 0, 1e-10, norms)
    u = z / norms[:, np.newaxis]

    U = rng.uniform(0, 1, size=num_points)
    radii = R * U**(1.0 / total_dim)
    scaled = u * radii[:, np.newaxis]

    return scaled.reshape(num_points, n, d)


def save_pickle(obj: Any, filename: str | Path, compress: bool = False) -> None:
    """Sauvegarde obj dans filename avec compression optionnelle."""
    filename = Path(filename)
    if compress and filename.suffix != ".gz":
        filename = filename.with_suffix(filename.suffix + ".gz")

    opener = gzip.open if filename.suffix == ".gz" else open
    with opener(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: str | Path) -> Any:
    """Charge un pickle avec détection automatique de compression."""
    filename = Path(filename)
    opener = gzip.open if filename.suffix == ".gz" else open
    with opener(filename, "rb") as f:
        return pickle.load(f)


def lj_objective(x, n, d):
    """
    Computes the Lennard–Jones potential (CPU/NumPy).
    
    NOTE: Cette fonction est conservée pour debug/vérification uniquement.
    En production, utiliser make_lj_objective() de solve_nlp_local_gpu.py
    qui génère une version JAX JIT-compilée (GPU).
    """
    X = np.reshape(x, (n, d))
    E = 0.0
    J = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diff = X[i] - X[j]
            r = np.linalg.norm(diff)
            r = max(r, 1e-8)
            E += 2.0 / (r**6)
            J += 1.0 / (r**12)
    return J - E


# ==============================================================================
# Invertible Mapping Functions (identiques à l'original)
# ==============================================================================

def linear_mapping(x, cluster_best, alpha):
    return (1 - alpha) * x + alpha * cluster_best


def affine_coupling_mapping(x, cluster_best, alpha, clip_value=2.0):
    d = len(x)
    mid = d // 2
    x_a = x[:mid]
    x_b = x[mid:]
    s = alpha * (cluster_best[:mid] - x_a)
    s = np.clip(s, -clip_value, clip_value)
    t = alpha * (cluster_best[mid:] - x_b)
    y_a = x_a
    y_b = x_b * np.exp(s) + t
    return np.concatenate([y_a, y_b])


def residual_mapping(x, cluster_best, alpha, mlp_params=None):
    f = alpha * np.tanh(cluster_best - x)
    return x + f


def cnf_mapping(x, cluster_best, alpha):
    def ode_func(y, t):
        return alpha * jnp.tanh(cluster_best - y)
    t = jnp.linspace(0, 1, 20)
    y_out = odeint(ode_func, x, t)
    return np.array(y_out[-1])


def invertible_mapping(x, cluster_best, alpha, method='linear', clip_value=2.0):
    if method == 'linear':
        return linear_mapping(x, cluster_best, alpha)
    elif method == 'affine':
        return affine_coupling_mapping(x, cluster_best, alpha, clip_value=2.0)
    elif method == 'residual':
        return residual_mapping(x, cluster_best, alpha)
    elif method == 'cnf':
        return cnf_mapping(x, cluster_best, alpha)
    else:
        raise ValueError("Unknown mapping method.")


# ==============================================================================
# Helper Clustering Function - VERSION GPU
# ==============================================================================

def perform_clustering(particles_array: np.ndarray, clustering_algo: str, 
                       random_state: int, backend_cfg: Dict = None) -> np.ndarray:
    """
    Clustering unifié avec support GPU via backend_ops.
    
    Args:
        particles_array: Array (n_samples, n_features)
        clustering_algo: "kmeans", "gmm", ou "dbscan"
        random_state: Graine aléatoire
        backend_cfg: Configuration backend (utilise défaut si None)
        
    Returns:
        labels: Array numpy int32
    """
    if backend_cfg is None:
        backend_cfg = get_default_backend_cfg()
    
    n_samples = particles_array.shape[0]
    n_clusters = max(2, int(np.sqrt(n_samples)))
    
    # Utiliser le wrapper unifié
    return cluster_fit_predict(
        particles_array,
        method=clustering_algo.lower(),
        n_clusters=n_clusters,
        random_state=random_state,
        cfg=backend_cfg,
        # Paramètres DBSCAN par défaut
        eps=0.5,
        min_samples=2
    )


# ==============================================================================
# Helper Functions for Diversity Enhancement - VERSION GPU
# ==============================================================================

def compute_sharing_factors(particles, sigma_share, backend_cfg: Dict = None) -> np.ndarray:
    """
    Calcule les facteurs de sharing avec support GPU.
    """
    arr = np.array(particles)
    if len(arr) < 2:
        return np.zeros(len(arr))
    return compute_sharing_factors_backend(arr, sigma_share, cfg=backend_cfg)


def compute_novelty(particles, archive):
    """Compute novelty based on archive (vectorized)."""
    if not archive:
        return np.zeros(len(particles))
    P = np.asarray(particles)
    A = np.asarray(archive)
    # Vectorized pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(P, A)   # (N, A)
    return np.mean(dists, axis=1)


def compute_crowding_distance(particles, k=2):
    """Compute crowding distance (vectorized)."""
    arr = np.array(particles)
    from scipy.spatial.distance import cdist
    D = cdist(arr, arr)
    np.fill_diagonal(D, np.inf)
    # Sort each row, take k nearest
    D_sorted = np.sort(D, axis=1)[:, :k]
    return np.mean(D_sorted, axis=1)


def expected_improvement(x, f_best, model):
    """Expected improvement for surrogate model (inchangé)."""
    x = np.array(x).reshape(1, -1)
    mu, sigma = model.predict(x, return_std=True)
    mu = mu[0]
    sigma = sigma[0]
    if sigma < 1e-8:
        return 0.0
    Z = (f_best - mu) / sigma
    return (f_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)


# ==============================================================================
# Problem Classes (identiques à l'original)
# ==============================================================================

class ThompsonEnergyPyomo:
    def __init__(self, N=50):
        self.N = N
        self.n = 3 * N
        self.m = N
        self.eps = 1e-8

    def build_model(self, x0, lb, ub, **kwargs):
        model = pyo.ConcreteModel()
        model.E = pyo.RangeSet(1, self.N)
        model.D = pyo.RangeSet(1, 3)
        init_dict = {}
        for j in range(1, 4):
            for i in range(1, self.N + 1):
                index = (j - 1) * self.N + (i - 1)
                init_dict[(i, j)] = x0[index]
        def var_bounds(model, i, j):
            index = (j - 1) * self.N + (i - 1)
            return (lb[index], ub[index])
        model.x = pyo.Var(model.E, model.D, domain=pyo.Reals, bounds=var_bounds, initialize=init_dict)
        def obj_rule(model):
            energy = 0.0
            for i in range(1, self.N):
                for j in range(i + 1, self.N + 1):
                    dx = model.x[i, 1] - model.x[j, 1]
                    dy = model.x[i, 2] - model.x[j, 2]
                    dz = model.x[i, 3] - model.x[j, 3]
                    r = pyo.sqrt(dx**2 + dy**2 + dz**2 + self.eps)
                    energy += 1.0 / r
            return energy
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        def sphere_rule(model, i):
            return model.x[i, 1]**2 + model.x[i, 2]**2 + model.x[i, 3]**2 == 1.0
        model.SphereConstr = pyo.Constraint(model.E, rule=sphere_rule)
        return model

    def obj(self, x):
        energy = 0.0
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                dx = x[i] - x[j]
                dy = x[self.N + i] - x[self.N + j]
                dz = x[2 * self.N + i] - x[2 * self.N + j]
                r = np.sqrt(dx**2 + dy**2 + dz**2 + self.eps)
                energy += 1.0 / r
        return energy

    @property
    def cl(self):
        return np.zeros(self.m)
    
    @property
    def cu(self):
        return np.zeros(self.m)


def thompson_obj_func(x, n, d):
    return ThompsonEnergyPyomo(N=50).obj(x)


def build_thompson_model(x0, lb, ub, **params):
    N = params.get('N', 50)
    return ThompsonEnergyPyomo(N=N).build_model(x0, lb, ub)


class LennardJonesPyomo:
    def __init__(self, n=10, d=3):
        self.n_atoms = n
        self.d = d
        self.n = n * d
        self.m = 0
        self.eps = 1e-8

    def build_model(self, x0, lb, ub, **kwargs):
        model = pyo.ConcreteModel()
        model.A = pyo.RangeSet(1, self.n_atoms)
        model.D = pyo.RangeSet(1, self.d)
        def var_bounds(model, i, j):
            return (-np.inf, np.inf)
        init_dict = {}
        for j in range(1, self.d+1):
            for i in range(1, self.n_atoms+1):
                index = (j-1)*self.n_atoms + (i-1)
                init_dict[(i,j)] = x0[index]
        model.x = pyo.Var(model.A, model.D, domain=pyo.Reals, bounds=var_bounds, initialize=init_dict)
        def obj_rule(model):
            expr = 0.0
            for i in model.A:
                for j in model.A:
                    if j > i:
                        r_sq = sum((model.x[i,k] - model.x[j,k])**2 for k in model.D) + self.eps
                        r = pyo.sqrt(r_sq)
                        expr += r**(-12) - 2*r**(-6)
            return expr
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        return model

    def obj(self, x):
        X = np.reshape(x, (self.n_atoms, self.d))
        energy = 0.0
        for i in range(self.n_atoms-1):
            for j in range(i+1, self.n_atoms):
                diff = X[i] - X[j]
                r = np.linalg.norm(diff)
                if r < self.eps:
                    r = self.eps
                energy += r**(-12) - 2*r**(-6)
        return energy

    @property
    def cl(self):
        return np.array([])
    
    @property
    def cu(self):
        return np.array([])


def lj_obj_func(x, n, d):
    return LennardJonesPyomo(n=n, d=d).obj(x)


def build_lj_model(x0, lb, ub, **params):
    n = params.get('n', 10)
    d = params.get('d', 3)
    return LennardJonesPyomo(n=n, d=d).build_model(x0, lb, ub)


# ==============================================================================
# Local Solver using Pyomo (identique à l'original)
# ==============================================================================

def solver_pyomo(model_builder, x0, lb, ub, model_params, ipopt_options):
    model = model_builder(x0, lb, ub, **model_params)
    opt = SolverFactory('ipopt')
    for key, val in ipopt_options.items():
        opt.options[key] = val
    try:
        results = opt.solve(model, tee=False)
        if results.solver.termination_condition not in ['optimal', 'acceptable']:
            print("Warning: IPOPT did not find an optimal solution. Termination condition:",
                  results.solver.termination_condition)
        sol = []
        if hasattr(model, 'x'):
            if hasattr(model.x, 'keys'):
                for key in sorted(model.x.keys()):
                    sol.append(pyo.value(model.x[key]))
            else:
                for i in model.I:
                    sol.append(pyo.value(model.x[i]))
        else:
            raise ValueError("Model does not have variable 'x'")
        x_opt = np.array(sol)
        obj_opt = float(pyo.value(model.obj))
        return x_opt, obj_opt
    except Exception as e:
        print("Pyomo solve failed with exception:", e)
        fallback_obj = 1e20
        return x0, fallback_obj


# ==============================================================================
# PF_NLP_Solver: Particle Filter Global Optimizer - VERSION GPU
# ==============================================================================

class PF_NLP_Solver:
    """
    Particle Filter Global Optimizer avec support GPU.
    
    La logique métier est IDENTIQUE à la version originale.
    Seuls les calculs de clustering et distances sont accélérés par GPU.
    """
    
    def __init__(self, model_builder, obj_func, x0, lb, ub, model_params,
                 pf_max_iter=1000, num_particles=50, ipopt_options=None, refine_top_k=5,
                 best_known_obj=-1e20, use_cma=False, use_fast_flow=True, 
                 use_invertible_flow=False, radius=5, use_differentiable_pf=False, 
                 clustering_algo="kmeans", stopping_criterion='median_mad', 
                 epsilon_stop=1e-6, mapping_method='linear', n_jobs=-1, 
                 output_dir=None, max_time=None, ipopt_refine_all=False,
                 no_improve_stop: int = 400,
                 # === NOUVEAU: Configuration backend GPU ===
                 backend_cfg: Dict = None):
        
        self.model_builder = model_builder
        self.model_params = model_params
        self.obj_func = obj_func
        self.x0 = np.array(x0)
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.pf_max_iter = pf_max_iter
        self.ipopt_refine_all = ipopt_refine_all
        self.num_particles = num_particles
        self.mapping_method = mapping_method
        self.refine_top_k = refine_top_k
        self.best_particle = None
        self.best_known_obj = best_known_obj
        self.archive = []
        self.use_cma = use_cma
        self.use_fast_flow = use_fast_flow
        self.use_invertible_flow = use_invertible_flow
        self.use_differentiable_pf = use_differentiable_pf
        self.clustering_algo = clustering_algo
        self.stopping_criterion = stopping_criterion
        self.mutation_sigma_init = 0.05
        self.mutation_sigma = self.mutation_sigma_init
        self.sa_initial_temp = 5000
        self.sa_cooling_rate = 0.998
        self.sa_max_iter = 100
        self.beta = self.beta_min = self.beta_min_init = 1e-4
        self.beta_max = self.beta_max_init = 5.
        self.beta_factor = 1.0
        self.epsilon_stop = epsilon_stop
        self.sigma_share = 0.1 * np.linalg.norm(self.ub - self.lb)
        self.lambda_share = 0.01
        self.lambda_novelty = 0.01
        self.lambda_crowding = 0.01
        self.radius = radius
        self.p_crossover = 0.1
        self.p_pso = 0.1
        self.dict_stats = dict()
        self.diversity_threshold = 0.05 * np.linalg.norm(self.ub - self.lb)
        self.n_jobs = n_jobs
        self.obj_eval_count = 0
        self.output_dir = output_dir
        self.max_time = max_time
        self.no_improve_stop = no_improve_stop   # 0 = désactivé
        
        # === NOUVEAU: Configuration backend GPU ===
        if backend_cfg is None:
            self.backend_cfg = get_default_backend_cfg()
        else:
            # Fusionner avec défauts
            self.backend_cfg = get_default_backend_cfg()
            self.backend_cfg.update(backend_cfg)
        # === NOUVEAU: Configuration du raffinement local (GPU JAX ou CPU IPOPT) ===
        # L'algorithme global PF reste inchangé ; seul le backend de raffinement change.
        self.refine_cfg = get_default_refine_cfg()
        # Suivre le master switch GPU du backend
        self.refine_cfg["use_gpu"] = bool(self.backend_cfg.get("use_gpu", True))
        # Aligner la précision JAX sur le dtype GPU (float32 recommandé pour perf)
        gpu_dtype = str(self.backend_cfg.get("gpu_dtype", "float32")).lower()
        self.refine_cfg["jax_x64"] = (gpu_dtype in ["float64", "fp64", "double"])
        # Méthode de raffinement: configurable via backend_cfg["local_refine_method"]
        self.refine_cfg["method"] = self.backend_cfg.get("local_refine_method", "lbfgsb")

        # === NOUVEAU: Objective JAX batch (GPU) ===
        self._n = int(self.model_params.get("n", len(self.x0)))
        self._d = int(self.model_params.get("d", 1))
        self._jax_obj = make_lj_objective(self._n, self._d)
        self._jax_obj_batch = jax.jit(jax.vmap(self._jax_obj))
        self._jax_dtype = jnp.float64 if self.refine_cfg["jax_x64"] else jnp.float32
        
        if ipopt_options is None:
            self.ipopt_options = {"tol": 1e-8, "print_level": 0, "hessian_approximation": "exact"}
        else:
            self.ipopt_options = ipopt_options.copy()
            self.ipopt_options.setdefault("hessian_approximation", "exact")
        
        # Initialize neural network for differentiable PF if requested
        if self.use_differentiable_pf:
            self.nn_init, self.nn_apply = stax.serial(
                stax.Dense(64), stax.Relu,
                stax.Dense(64), stax.Relu,
                stax.Dense(self.x0.shape[0])
            )
            rng_nn = jax.random.PRNGKey(0)
            out_shape, init_params = self.nn_init(rng_nn, (-1, self.x0.shape[0]))
            self.nn_params = init_params
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-3)
            self.opt_state = self.opt_init(self.nn_params)

    def set_backend_cfg(self, **kwargs):
        """
        Met à jour la configuration backend à la volée.
        
        Exemple:
            solver.set_backend_cfg(use_gpu=False, gpu_clustering=False)
        """
        self.backend_cfg.update(kwargs)
        
    def get_backend_info(self) -> Dict:
        """Retourne les informations sur le backend actuel."""
        return {
            "config": self.backend_cfg.copy(),
            "system": get_backend_info()
        }

    def display_options(self):
        print("\n==============================================================")
        print("Starting Particle Filter Global Optimization (GPU-Enabled)")
        print("  Number of atoms:", self.model_params['n'])
        print("Selected Options:")
        print("  PF max iterations:", self.pf_max_iter)
        print("  Number of particles:", self.num_particles)
        print("  Refinement top-k:", self.refine_top_k)
        print("  Mutation sigma init:", self.mutation_sigma_init)
        print("  SA initial temperature:", self.sa_initial_temp)
        print("  Beta:", self.beta)
        print("  Mapping method:", self.mapping_method)
        print("  Clustering algorithm:", self.clustering_algo.upper())
        print("--- Backend Configuration ---")
        print(f"  GPU enabled: {self.backend_cfg['use_gpu']}")
        print(f"  GPU clustering: {self.backend_cfg['gpu_clustering']}")
        print(f"  GPU pdist: {self.backend_cfg['gpu_pdist']}")
        print(f"  GPU dtype: {self.backend_cfg['gpu_dtype']}")
        print(f"  Local refinement: {self.refine_cfg.get('method', 'lbfgsb')}")
        if self.use_differentiable_pf:
            print("  Particle Flow Method: Differentiable PF (with NN)")
        elif self.use_invertible_flow:
            print("  Particle Flow Method: Invertible PF")
        elif self.use_fast_flow:
            print("  Particle Flow Method: Fast Particle Filtering via Clustering")
        else:
            print("  Particle Flow Method: Standard Mutation")
        print("=============================================================\n")

    def objective_total(self, x):
        """Objective (scalaire) via JAX (GPU si dispo)."""
        self.obj_eval_count += 1
        xj = jnp.asarray(np.asarray(x), dtype=self._jax_dtype)
        return float(jax.device_get(self._jax_obj(xj)))

    def objective_batch(self, particles):
        """Objective batch JAX (GPU): retourne un np.ndarray (B,)."""
        X = np.asarray(particles)
        if X.ndim == 1:
            X = X[None, :]
        self.obj_eval_count += X.shape[0]
        # Guard: replace any NaN/Inf inputs with zeros to avoid JAX crash
        bad_mask = ~np.all(np.isfinite(X), axis=1)
        if bad_mask.any():
            X = X.copy()
            X[bad_mask] = 0.0
        Xj = jnp.asarray(X, dtype=self._jax_dtype)
        vals = self._jax_obj_batch(Xj)
        result = np.asarray(jax.device_get(vals), dtype=float)
        # Mark bad inputs as inf
        result[bad_mask] = 1e20
        return result

    def local_optimization(self, x):
        """
        Raffinement local d'une particule.
        
        - Mode 'ipopt': utilise solver_lj_cyipopt (Pyomo/IPOPT) - compatible avec Parallel
        - Autres modes: utilise local_refine (JAX)
        """
        method = self.refine_cfg.get("method", "lbfgsb")
        
        if method == "ipopt":
            # Utiliser le solver IPOPT original (compatible Parallel/joblib)
            from solve_nlp_ipopt import solver_lj_cyipopt
            sol, _ = solver_lj_cyipopt(
                n=int(self.model_params['n']),
                d=int(self.model_params['d']),
                lb=self.lb,
                ub=self.ub,
                x0=np.asarray(x),
                radius=self.radius,
                tol=1e-8,
                maxit=5000,
                disp=0
            )
            return sol
        else:
            # Utiliser local_refine (JAX) - attention: pas optimal avec Parallel
            sol, _ = local_refine(
                x0=np.asarray(x),
                lb=self.lb,
                ub=self.ub,
                model_params=self.model_params,
                cfg=self.refine_cfg,
            )
            return sol

    def _random_box_point(self, rng):
        finite_lb = np.where(np.isinf(self.lb), -1.0, self.lb)
        finite_ub = np.where(np.isinf(self.ub), 1.0, self.ub)
        return rng.uniform(low=finite_lb, high=finite_ub)

    def _mutate_particles_adaptive(self, particles, pf_iter, max_pf_iter, rng):
        self.mutation_sigma = self.mutation_sigma_init * max(0.1, 1.0 - (pf_iter/max_pf_iter))
        cov_matrix = None
        if self.use_cma:
            pop_array = np.array(particles)
            if pop_array.shape[0] > 1:
                cov_matrix = np.cov(pop_array.T)
        return Parallel(n_jobs=self.n_jobs)(
            delayed(mutate_wrapper)(self, p, self.mutation_sigma, self.lb, self.ub, rng, particles, cov_matrix)
            for p in particles
        )

    def advanced_mutate_wrapper(self, x, lb, ub, rng, population):
        x = np.asarray(x)
        r = rng.random()
        if r < 0.4 and len(population) >= 3:
            indices = rng.choice(len(population), 2, replace=False)
            r1 = np.asarray(population[indices[0]])
            r2 = np.asarray(population[indices[1]])
            F = 0.8
            mutant = x + F * (r1 - r2)
            new_candidate = mutant + rng.normal(scale=self.mutation_sigma, size=x.shape)
        elif r < 0.7:
            best_particle = np.asarray(self.best_particle) if self.best_particle is not None else x
            directed = x + 0.3 * (best_particle - x)
            new_candidate = directed + rng.normal(scale=self.mutation_sigma, size=x.shape)
        elif r < 0.85:
            random_point = rng.uniform(low=self.lb, high=self.ub, size=x.shape)
            alpha = rng.random()
            blended = alpha * x + (1 - alpha) * random_point
            new_candidate = blended + rng.normal(scale=self.mutation_sigma, size=x.shape)
        elif r < 0.95:
            if self.archive:
                partner = np.asarray(self.archive[rng.integers(0, len(self.archive))])
            else:
                partner = rng.uniform(low=self.lb, high=self.ub, size=x.shape)
            new_candidate = 0.5 * (x + partner) + rng.normal(scale=self.mutation_sigma, size=x.shape)
        else:
            best_particle = np.asarray(self.best_particle) if self.best_particle is not None else x
            new_candidate = x + 0.5 * (best_particle - x) + rng.normal(scale=self.mutation_sigma, size=x.shape)
        return np.clip(new_candidate, self.lb, self.ub)

    def simulated_annealing(self, initial_solution, population, adaptive_step=True, verbose=False):
        current_solution = initial_solution.copy()
        current_obj = self.objective_total(current_solution)
        best_solution = current_solution.copy()
        best_obj = current_obj
        temperature = self.sa_initial_temp
        step_size = np.std(initial_solution) if adaptive_step else 1.0
        variance_window = []
        window_size = 10
        variance_temp_factor = 1e-6
        rng_sa = np.random.default_rng()
        for iter in range(self.sa_max_iter):
            new_solution = self.advanced_mutate_wrapper(current_solution, self.lb, self.ub, rng_sa, population)
            new_solution, new_obj = local_refine(
                x0=np.asarray(new_solution),
                lb=self.lb,
                ub=self.ub,
                model_params=self.model_params,
                cfg=self.refine_cfg,
            )
            
            if new_obj < current_obj:
                acceptance_prob = 1.0
            else:
                acceptance_prob = np.exp((current_obj - new_obj) / max(temperature, 1e-12))
            if acceptance_prob > rng_sa.random():
                current_solution = new_solution.copy()
                current_obj = new_obj
                if new_obj < best_obj:
                    best_solution = new_solution.copy()
                    best_obj = new_obj
            temperature *= self.sa_cooling_rate
            if adaptive_step:
                step_size = np.std(current_solution)
            variance_window.append(current_obj)
            if len(variance_window) > window_size:
                variance_window.pop(0)
            if len(variance_window) == window_size and np.var(variance_window) < variance_temp_factor * (temperature**2):
                if verbose:
                    print(f"SA iter {iter+1}: Convergence reached.")
                break
            if verbose:
                print(f"SA iter {iter+1}: Obj = {current_obj:.6e}, Best = {best_obj:.6e}, Temp = {temperature:.6e}")
        return best_solution, best_obj

    def _compute_weights(self, objs, beta, particles):
        # === UTILISE LE BACKEND GPU POUR compute_sharing_factors ===
        sharing = compute_sharing_factors(particles, self.sigma_share, self.backend_cfg)
        novelty = compute_novelty(particles, self.archive)
        effective_objs = objs + self.lambda_share * sharing - self.lambda_novelty * novelty
        std_obj = np.std(effective_objs)
        std_obj = max(std_obj, 1e-8)
        # --- LOG-SUM-EXP TRICK: prevents overflow when beta is large ---
        log_w = -beta * (effective_objs - np.min(effective_objs)) / std_obj
        log_w -= np.max(log_w)          # shift so max(log_w) = 0
        weights = np.exp(log_w)
        weights = np.clip(weights, 1e-8, None)   # same floor as original
        weights /= (np.sum(weights) + 1e-12)
        ess = 1.0 / np.sum(weights**2)
        return weights, ess

    def _resample_particles(self, particles, current_num, rng):
        if len(particles) > current_num:
            indices = rng.choice(len(particles), current_num, replace=False)
            particles = [particles[i] for i in indices]
        return particles

    def _resample_particles_by_weights(self, particles, weights, rng):
        N = len(particles)
        # --- Robust stratified resampling via searchsorted ---
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 0, None)
        w_sum = w.sum()
        if w_sum < 1e-300 or not np.isfinite(w_sum):
            # Weights are degenerate — uniform resample
            w = np.ones(N) / N
        else:
            w = w / w_sum
        cumulative_sum = np.cumsum(w)
        cumulative_sum[-1] = 1.0  # ensure last element is exactly 1
        positions = (rng.random(N) + np.arange(N)) / N   # stratified (N offsets)
        indexes = np.searchsorted(cumulative_sum, positions, side='left')
        indexes = np.clip(indexes, 0, N - 1)
        new_particles = [particles[i].copy() for i in indexes]
        jitter_scale = 0.01 * (self.ub - self.lb)
        new_particles = [np.clip(p + rng.normal(scale=jitter_scale, size=p.shape), self.lb, self.ub)
                         for p in new_particles]
        return new_particles

    def _update_best_enkf(self, objs, particles, best_obj, best_particle):
        idx_best = np.argmin(objs)
        if objs[idx_best] < best_obj:
            best_obj = objs[idx_best]
            best_particle = particles[idx_best].copy()
            self.archive.append(best_particle.copy())
            if len(self.archive) > 10:
                self.archive = self.archive[-10:]
        # Skip EnKF adjustment if dimension is too large (O(D³) cost)
        D = len(best_particle)
        if D > 500 or len(particles) < 5:
            return best_obj, best_particle
        particle_matrix = np.array(particles)
        mean_particle = np.mean(particle_matrix, axis=0)
        perturbations = particle_matrix - mean_particle
        cov_perturbations = np.cov(perturbations, rowvar=False)
        if cov_perturbations.ndim < 2:
            return best_obj, best_particle
        cov_norm = np.linalg.norm(cov_perturbations)
        if cov_norm < 1e-8 or not np.isfinite(cov_norm):
            return best_obj, best_particle
        # Use solve() instead of inv() — more stable and O(D²) vs O(D³)
        reg = 1e-6 * np.eye(D)
        rhs = (best_particle - mean_particle).reshape(-1, 1)
        try:
            kalman_rhs = np.linalg.solve(cov_perturbations + np.eye(D) + reg, rhs)
            adjustment = 0.1 * (cov_perturbations @ kalman_rhs).flatten()
            updated_candidate = best_particle + adjustment
        except np.linalg.LinAlgError:
            updated_candidate = best_particle
        return best_obj, updated_candidate

    def _check_convergence(self, objs, tol=1e-6, threshold=0.5, min_particles=50):
        if len(objs) < min_particles:
            return False
        best_val = np.min(objs)
        close_count = np.sum(np.abs(objs - best_val) < tol)
        return (close_count / len(objs)) >= threshold

    def adjust_beta(self, pf_iter, max_pf_iter, weights, beta, particles=None):
        ess = 1.0 / np.sum(weights**2)
        target_ess = 0.5 * self.num_particles
        # NOTE: use diversity_threshold as default (original tested behavior)
        # Actual diversity is computed separately in tune_parameters
        diversity = self.diversity_threshold
        diversity_factor = np.clip(diversity / self.diversity_threshold, 0.1, 1.0)
        progress = pf_iter / max(max_pf_iter - 1, 1)
        self.beta_min = self.beta_min_init * (1 - 0.5 * progress)
        self.beta_max = self.beta_max_init * (1 + 0.5 * (1 - diversity_factor))
        ess_factor = np.clip(ess / target_ess, 0.1, 1.0)
        beta_new = beta + self.beta_factor * (self.beta_max - beta) * (progress * ess_factor * diversity_factor)
        return float(np.clip(beta_new, self.beta_min, self.beta_max))

    def compute_diversity(self, particles):
        """Calcule la diversité avec support GPU."""
        arr = np.array(particles)
        if len(arr) < 2:
            return np.inf
        # === UTILISE LE BACKEND GPU POUR pdist ===
        return compute_diversity_backend(arr, cfg=self.backend_cfg)

    def island_exchange(self, particles, num_islands):
        islands = [[] for _ in range(num_islands)]
        for i, p in enumerate(particles):
            islands[i % num_islands].append(p)
        new_particles = []
        for island in islands:
            if island:
                objs = list(self.objective_batch(island))
                best_idx = np.argmin(objs)
                new_particles.append(island[best_idx])
        merged = particles + new_particles
        rng_local = np.random.default_rng()
        N_merged = len(merged)
        idx = rng_local.choice(N_merged, self.num_particles,
                               replace=(N_merged < self.num_particles))
        return [merged[i] for i in idx]

    def tune_parameters(self, pf_iter, no_improvement_counter, ess, diversity):
        if no_improvement_counter >= 10:
            self.mutation_sigma = min(self.mutation_sigma * 1.2, 1.0)
            self.sa_initial_temp *= 1.1
        else:
            self.mutation_sigma = self.mutation_sigma_init * max(0.1, 1.0 - (pf_iter / self.pf_max_iter))
        if ess < 0.05 * self.num_particles:
            self.beta_factor = min(self.beta_factor * 1.1, 5.0)
        else:
            self.beta_factor = max(self.beta_factor * 0.95, 1.0)
        if diversity < self.diversity_threshold:
            num_to_replace = int(0.2 * self.num_particles)
            rng_local = np.random.default_rng()
            new_particles = [self._random_box_point(rng_local) for _ in range(num_to_replace)]
            return new_particles
        return []

    def _final_local_optimization(self, particles, discard_top_k, best_obj, best_particle):
        final_objs = list(self.objective_batch(particles))
        sorted_idx = np.argsort(final_objs)
        top_candidates = [particles[i] for i in sorted_idx[:discard_top_k]]
        for candidate in top_candidates:
            refined_candidate = self.local_optimization(candidate)
            refined_obj = self.objective_total(refined_candidate)
            if refined_obj < best_obj:
                best_obj = refined_obj
                best_particle = refined_candidate
        return best_particle, best_obj

    def differentiable_particle_flow(self, particles, rng, pf_iter, total_iter):
        updated_particles = []
        for p in particles:
            p_flat = p.reshape(1, -1)
            delta = np.array(self.nn_apply(self.nn_params, p_flat))[0]
            updated_particles.append(np.clip(p + delta, self.lb, self.ub))
        loss = float(np.mean(self.objective_batch(updated_particles)))
        
        def loss_fn(params):
            preds = [self.nn_apply(params, p.reshape(1, -1))[0] for p in particles]
            new_particles = [np.clip(p + pred, self.lb, self.ub) for p, pred in zip(particles, preds)]
            return float(np.mean(self.objective_batch(new_particles)))
        
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(self.nn_params)
        self.opt_state = self.opt_update(pf_iter, grads, self.opt_state)
        self.nn_params = self.get_params(self.opt_state)
        jitter_scale = 0.02 * (self.ub - self.lb)
        new_particles = [np.clip(p + rng.normal(scale=jitter_scale, size=p.shape), self.lb, self.ub) for p in updated_particles]
        return new_particles

    def invertible_particle_flow(self, particles, rng, pf_iter, total_iter):
        """
        Invertible Particle Flow Update via Clustering - VERSION GPU
        """
        alpha_base = 0.98
        alpha = alpha_base * (1 - pf_iter/total_iter) + 0.05
        particles_array = np.array(particles)
        random_state = rng.integers(0, 10000)
        
        # === UTILISE LE BACKEND GPU POUR LE CLUSTERING ===
        labels = perform_clustering(particles_array, self.clustering_algo, random_state, self.backend_cfg)
        
        new_particles = []
        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            if len(idx) == 0:
                continue
            cluster_particles = [particles[i] for i in idx]
            objs_cluster = np.array(self.objective_batch(cluster_particles))
            best_idx = np.argmin(objs_cluster)
            cluster_best = cluster_particles[best_idx]
            for p in cluster_particles:
                updated = invertible_mapping(p, cluster_best, alpha, method=self.mapping_method)
                new_particles.append(np.clip(updated, self.lb, self.ub))
        jitter_scale = 0.02 * (self.ub - self.lb)
        new_particles = [np.clip(p + rng.normal(scale=jitter_scale, size=p.shape), self.lb, self.ub)
                         for p in new_particles]
        return new_particles

    def fast_particle_flow(self, particles, rng, pf_iter, total_iter):
        """
        Fast Particle Flow Update via Clustering - VERSION GPU
        """
        base_flow_factor = 0.98
        decay = pf_iter / total_iter
        adaptive_flow = base_flow_factor * (1 - decay) + 0.05
        particles_array = np.array(particles)
        random_state = rng.integers(0, 10000)
        
        # === UTILISE LE BACKEND GPU POUR LE CLUSTERING ===
        labels = perform_clustering(particles_array, self.clustering_algo, random_state, self.backend_cfg)
        
        new_particles = []
        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            if len(idx) == 0:
                continue
            cluster_particles = [particles[i] for i in idx]
            objs_cluster = np.array(self.objective_batch(cluster_particles))
            best_idx = np.argmin(objs_cluster)
            cluster_best = cluster_particles[best_idx]
            for p in cluster_particles:
                updated = p + adaptive_flow * (cluster_best - p)
                new_particles.append(np.clip(updated, self.lb, self.ub))
        jitter_scale = 0.04 * (self.ub - self.lb)
        new_particles = [np.clip(p + rng.normal(scale=jitter_scale, size=p.shape), self.lb, self.ub)
                         for p in new_particles]
        return new_particles

    def _solve_pf_loop(self, seed=42, verbose=True, min_particles=5, run_num=0, progress_file=None):
        import sys
        self.obj_eval_count = 0
        
        def log_progress(msg):
            print(msg, flush=True)
            sys.stdout.flush()
            if progress_file:
                try:
                    with open(progress_file, 'a') as f:
                        f.write(f"{time.strftime('%H:%M:%S')} | {msg}\n")
                        f.flush()
                except:
                    pass
        
        log_progress(f"[Init] Generating {self.num_particles} particles...")
        log_progress(f"[Init] Target (best_known_obj): {self.best_known_obj:.6f}, epsilon_stop: {self.epsilon_stop}")
        log_progress(f"[Init] Backend: GPU={self.backend_cfg['use_gpu']}, Clustering GPU={self.backend_cfg['gpu_clustering']}, pdist GPU={self.backend_cfg['gpu_pdist']}")
        
        rng = np.random.default_rng(seed)
        if verbose:
            self.display_options()
        particles = [self._random_box_point(rng) for _ in range(self.num_particles)]
        particles = [p.flatten() for p in particles]
        
        log_progress(f"[Init] Starting initial local optimization of {len(particles)} particles (this may take a while)...")
        
        best_obj = float('inf')
        best_particle = None
        global_best_obj = float('inf')
        global_best = None
        prev_best_obj = float('inf')
        no_improvement_counter = 0
        global_no_improve = 0    # compteur global — NE SE REMET PAS à 0 lors des kicks
        best_obj_ever = float('inf')
        overall_start = time.time()
        prev_stop_value = None
        
        step_size = 0.38
        kick_factor = 10
        best_config = None
        num_vars = self.model_params["n"] * self.model_params["d"]
        
        self.dict_stats['diversity'] = []
        self.dict_stats['kick'] = []
        self.dict_stats['objs'] = []
        self.dict_stats['weights'] = []
        self.dict_stats['ess'] = []
        self.dict_stats['beta'] = []
        self.dict_stats['beta_min'] = []
        self.dict_stats['beta_max'] = []
        
        if verbose:
            print("\n" + "-"*88)
            print(f"{'Iter':^5}|{'Best f':^13}|{'Current f':^13}|{'#Part':^6}|{'Beta':^6}|{'Sigma':^6}|{'Radius':^6}|{'ESS':^8}|{'Beta Min':^6}|{'Beta Max':^6}|")
            print("-"*88)
        
        # Import KMeans pour utilisation dans la boucle (avec fallback si GPU non dispo)
        for pf_iter in range(self.pf_max_iter):
            current_num = max(min_particles, int(self.num_particles * (1 - pf_iter / self.pf_max_iter)))
            particles = self._resample_particles(particles, current_num, rng)
            
            if pf_iter % 10 == 0 and pf_iter > 0:
                particles = self.island_exchange(particles, num_islands=3)
            
            if pf_iter % 20 == 0 and pf_iter > 0:
                # === UTILISE LE BACKEND GPU POUR LE CLUSTERING ===
                particles_array = np.array(particles)
                labels_temp = cluster_fit_predict(
                    particles_array, 
                    method="kmeans", 
                    n_clusters=3, 
                    random_state=42,
                    cfg=self.backend_cfg
                )
                reps = []
                for cl in range(3):
                    idx = np.where(labels_temp == cl)[0]
                    if len(idx) > 0:
                        cl_particles = [particles[i] for i in idx]
                        objs_cl = list(self.objective_batch(cl_particles))
                        reps.append(cl_particles[np.argmin(objs_cl)])
                particles.extend(reps)
            
            # Choose the update method based on selected options
            if self.use_differentiable_pf:
                particles = self.differentiable_particle_flow(particles, rng, pf_iter, self.pf_max_iter)
            elif self.use_invertible_flow:
                particles = self.invertible_particle_flow(particles, rng, pf_iter, self.pf_max_iter)
            elif self.use_fast_flow:
                particles = self.fast_particle_flow(particles, rng, pf_iter, self.pf_max_iter)
            else:
                particles = self._mutate_particles_adaptive(particles, pf_iter, self.pf_max_iter, rng)
            
            # --- Raffinement local ---
            method = self.refine_cfg.get("method", "adam")
            
            if method == "ipopt":
                if self.ipopt_refine_all:
                    # ancien comportement (toutes particules)
                    particles = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.local_optimization)(p) for p in particles
                    )
                else:
                    # IPOPT (CPU): top-k seulement (au lieu de toutes les particules)
                    if pf_iter == 0:
                        log_progress(f"[Iter 1] Local refinement of top-{self.refine_top_k} particles (IPOPT/Parallel)...")
                
                    # 1) choisir les top-k (on peut utiliser objective_batch si disponible)
                    objs_pre = self.objective_batch(particles)   # (N,) float
                    idx_ref = np.argsort(objs_pre)[:self.refine_top_k]
                
                    # 2) raffiner uniquement ces particules (CPU, donc Parallel OK)
                    refined = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.local_optimization)(particles[i]) for i in idx_ref
                    )
                
                    # 3) réinjecter
                    for i, x_new in zip(idx_ref, refined):
                        particles[i] = x_new
            
            else:
                # JAX (GPU): top-k séquentiel (évite concurrence GPU)
                if pf_iter == 0:
                    log_progress(f"[Iter 1] Local refinement of top-{self.refine_top_k} particles (JAX/GPU)...")
            
                objs_pre = self.objective_batch(particles)
                idx_ref = np.argsort(objs_pre)[:self.refine_top_k]
            
                for i in idx_ref:
                    particles[i] = self.local_optimization(particles[i])

            
            # --- Évaluation objective (batch JAX/GPU) ---
            objs = self.objective_batch(particles)
            objs = np.nan_to_num(objs, nan=1e20)
            
            idx_best = np.argmin(objs)
            current_iter_best = objs[idx_best]
            
            if current_iter_best < global_best_obj:
                global_best_obj = current_iter_best
                global_best = particles[idx_best].copy()
            
            best_obj, best_particle = self._update_best_enkf(objs, particles, best_obj, best_particle)
            
            weights, ess = self._compute_weights(objs, self.beta, particles)
            
            if verbose:
                print(f"{pf_iter+1:^5}|{global_best_obj:^13.7f}|{best_obj:^13.7f}|{len(objs):^6}|{self.beta:^6.4f}|{self.mutation_sigma:^6.3f}|{self.radius:^6.3f}|{ess:^8.3f}|{self.beta_min:^8.4f}|{self.beta_max:^8.4f}|", flush=True)
            elif pf_iter % 10 == 0 or pf_iter == self.pf_max_iter - 1:
                elapsed = time.time() - overall_start
                log_progress(f"  [Iter {pf_iter+1:4d}/{self.pf_max_iter}] global_best={global_best_obj:.6f} | iter_best={current_iter_best:.6f} | particles={len(objs)} | time={elapsed:.1f}s")

            # Condition d'arrêt sur l'optimum connu
            if self.best_known_obj > -1e10:
                gap_to_known = abs(global_best_obj - self.best_known_obj)
                if global_best_obj <= self.best_known_obj + self.epsilon_stop:
                    elapsed = time.time() - overall_start
                    log_progress(f"  [STOP] Best known objective reached!")
                    log_progress(f"         global_best={global_best_obj:.8f}, known={self.best_known_obj:.8f}, gap={gap_to_known:.2e}")
                    print(f"[STOP] Best known reached: {global_best_obj:.8f} <= {self.best_known_obj:.8f} + {self.epsilon_stop}", flush=True)
                    break
            
            # Condition d'arrêt sur le temps maximum
            if self.max_time is not None:
                elapsed = time.time() - overall_start
                if elapsed >= self.max_time:
                    log_progress(f"  [TIMEOUT] Max time ({self.max_time}s) reached after {pf_iter+1} iterations")
                    log_progress(f"            global_best={global_best_obj:.8f}, elapsed={elapsed:.1f}s")
                    print(f"[TIMEOUT] Max time {self.max_time}s reached. Best: {global_best_obj:.8f}", flush=True)
                    break
            
            if ess < 0.1 * self.num_particles:
                particles = self._resample_particles_by_weights(particles, weights, rng)
                if ess < 0.05 * self.num_particles:
                    additional_particles = [self._random_box_point(rng) for _ in range(int(0.1 * self.num_particles))]
                    particles.extend(additional_particles)
                    N_cur = len(particles)
                    if N_cur != self.num_particles:
                        idx = rng.choice(N_cur, self.num_particles, replace=(N_cur < self.num_particles))
                        particles = [particles[i] for i in idx]
            
            self.beta = self.adjust_beta(pf_iter, self.pf_max_iter, weights, self.beta, particles)
            
            if abs(best_obj - prev_best_obj) < 1e-6:
                no_improvement_counter += 1
                kick_factor += 0.5
            else:
                no_improvement_counter = 0
                kick_factor = 10.5

            # Compteur global de stagnation (survit aux kicks)
            if best_obj < best_obj_ever - 1e-6:
                best_obj_ever = best_obj
                global_no_improve = 0
            else:
                global_no_improve += 1

            # Condition d'arrêt par stagnation globale
            if self.no_improve_stop > 0 and global_no_improve >= self.no_improve_stop:
                elapsed = time.time() - overall_start
                log_progress(f"  [STOP] No improvement for {global_no_improve} iterations "
                             f"(threshold={self.no_improve_stop})")
                log_progress(f"         global_best={global_best_obj:.8f}, elapsed={elapsed:.1f}s")
                print(f"[STOP] No improvement for {global_no_improve} iters. "
                      f"Best: {global_best_obj:.8f}", flush=True)
                break
            
            diversity = self.compute_diversity(particles)
            extra_particles = self.tune_parameters(pf_iter, no_improvement_counter, ess, diversity)
            if extra_particles:
                particles.extend(extra_particles)
                if verbose:
                    print(f"Adaptive reinitialization: Injected {len(extra_particles)} particles (diversity={diversity:.3e}).")
            prev_best_obj = best_obj
            
            with_kick = 0
            if no_improvement_counter >= 10:
                with_kick = 1
                kick_factor = min(100.0, kick_factor + 0.5 * no_improvement_counter + 0.01 * self.model_params['n'] * 3)
                self.radius = max(0.98 * self.radius, 3.)
                self.lb = -self.radius * np.ones(num_vars)
                self.ub = self.radius * np.ones(num_vars)
                new_particles = []
                for p in particles:
                    new_particles.append(np.clip(p + rng.uniform(-2.0, 2.0, size=p.shape) * kick_factor * step_size, -self.radius, self.radius))
                particles = new_particles
                no_improvement_counter = 0
            
            self.dict_stats['diversity'].append(diversity)
            self.dict_stats['kick'].append(with_kick)
            self.dict_stats['objs'].append(objs)
            self.dict_stats['weights'].append(weights)
            self.dict_stats['ess'].append(ess)
            self.dict_stats['beta'].append(self.beta)
            self.dict_stats['beta_min'].append(self.beta_min)
            self.dict_stats['beta_max'].append(self.beta_max)
        
        # Sauvegarde des stats
        pkl_filename = f"dict_stats_LJ_v0_INVF_{self.use_invertible_flow}_n{self.model_params['n']}_Run{run_num}.pkl"
        if self.output_dir:
            pkl_path = Path(self.output_dir) / pkl_filename
        else:
            pkl_path = pkl_filename
        save_pickle(self.dict_stats, pkl_path)
        
        elapsed_time = time.time() - overall_start
        if not verbose:
            print(f" [RUN {run_num}]|{pf_iter+1:^5}|{global_best_obj:^13.7f}|{best_obj:^13.7f}|{len(objs):^6}|{self.beta:^6.4f}|{self.mutation_sigma:^6.3f}|{self.radius:^6.3f}|{ess:^8.3f}|{self.beta_min:^8.4f}|{self.beta_max:^8.4f}|{elapsed_time:^8.2f}secs|{self.obj_eval_count:^10}evals|")
        else:
            print(f"Total PF time: {elapsed_time:.2f} sec | Objective evaluations: {self.obj_eval_count}")
        
        final_best, final_obj = self._final_local_optimization(particles, self.refine_top_k, best_obj, best_particle)
        if final_obj < global_best_obj:
            return final_best, final_obj
        else:
            return global_best, global_best_obj

    def solve_particle_filter(self, seed=42, verbose=True, min_particles=5, run_num=0, progress_file=None):
        return self._solve_pf_loop(seed=seed, verbose=verbose, min_particles=min_particles, run_num=run_num, progress_file=progress_file)


# ==============================================================================
# Helper Mutation Function (identique à l'original)
# ==============================================================================

def mutate_wrapper(solver, x, sigma, lb, ub, rng, population, cov_matrix=None):
    x = np.asarray(x)
    r = rng.random()
    
    def sample_noise(shape):
        if cov_matrix is not None:
            reg_cov = cov_matrix + 1e-8 * np.eye(shape[0])
            return np.random.multivariate_normal(np.zeros(shape[0]), (sigma**2) * reg_cov)
        else:
            return rng.normal(scale=sigma, size=shape)
    
    if r < 0.4 and len(population) >= 3:
        indices = rng.choice(len(population), 2, replace=False)
        r1 = np.asarray(population[indices[0]])
        r2 = np.asarray(population[indices[1]])
        F = 0.8
        mutant = x + F * (r1 - r2)
        noise = sample_noise(x.shape)
        new_candidate = mutant + noise
    elif r < 0.7:
        best_particle = np.asarray(solver.best_particle) if solver.best_particle is not None else x
        directed = x + 0.3 * (best_particle - x)
        noise = sample_noise(x.shape)
        new_candidate = directed + noise
    elif r < 0.85:
        random_point = rng.uniform(low=lb, high=ub, size=x.shape)
        alpha = rng.random()
        blended = alpha * x + (1 - alpha) * random_point
        noise = sample_noise(x.shape)
        new_candidate = blended + noise
    elif r < 0.95:
        if solver.archive:
            partner = np.asarray(solver.archive[rng.integers(0, len(solver.archive))])
        else:
            partner = rng.uniform(low=lb, high=ub, size=x.shape)
        noise = sample_noise(x.shape)
        new_candidate = 0.5 * (x + partner) + noise
    else:
        best_particle = np.asarray(solver.best_particle) if solver.best_particle is not None else x
        noise = sample_noise(x.shape)
        new_candidate = x + 0.5 * (best_particle - x) + noise
    
    return np.clip(new_candidate, lb, ub)


# ==============================================================================
# Best Known Solutions
# ==============================================================================

BEST_KNOWN_SOL = {
    38: -173.928427,
    75: -397.492331,
    76: -402.894866,
    77: -409.083517,
    98: -543.665361,
    102: -569.363653,
    103: -575.766131,
    104: -582.086642,
}


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import hashlib
    import json
    import csv
    from datetime import datetime
    
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    def make_seed(base: int, n: int, m: int, offset: int = 0) -> int:
        s = f"{base}_{n}_{m}_{offset}"
        h = hashlib.md5(s.encode()).hexdigest()
        return int(h[:8], 16)
    
    def setup_logging(log_file: str):
        import logging as log_module
        formatter = log_module.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = log_module.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_module.INFO)
        console_handler = log_module.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_module.INFO)
        logger = log_module.getLogger('PF_LJ_GPU')
        logger.setLevel(log_module.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        original_info = logger.info
        def info_with_flush(msg, *args, **kwargs):
            original_info(msg, *args, **kwargs)
            for handler in logger.handlers:
                handler.flush()
        logger.info = info_with_flush
        return logger
    
    # =========================================================================
    # PARSING ARGUMENTS
    # =========================================================================
    parser = argparse.ArgumentParser(description="PF-NLP Optimizer for Lennard-Jones (GPU-Enabled)")
    parser.add_argument("--atoms", type=int, nargs="+", default=[38,75,76,77,98,102,103,104], help="List of atom counts")
    parser.add_argument("--runs", type=int, default=30, help="Number of runs per atom count")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max PF iterations")
    parser.add_argument("--max-time", type=int, default=20000, help="Max time per run (seconds)")
    parser.add_argument("--particles", type=int, default=None, help="Number of particles (auto if None)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output-dir", type=str, default="results_gpu", help="Output directory")
    parser.add_argument("--seed", type=int, default=2024, help="Base random seed")
    parser.add_argument("--refine-top-k", type=int, default=5, help="top k particules for refinement")
    
    parser.add_argument("--ipopt-refine-all",action="store_true", help="Refine all particles or not")
    # Arguments GPU
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU (use CPU only)")
    parser.add_argument("--no-gpu-clustering", action="store_true", help="Disable GPU clustering only")
    parser.add_argument("--no-gpu-pdist", action="store_true", help="Disable GPU pdist only")
    parser.add_argument("--local-refine", type=str, default="ipopt", 
                        choices=["lbfgsb", "adam", "pgd", "ipopt"],
                        help="Local refinement method: lbfgsb (GPU), adam (GPU), pgd (GPU), ipopt (CPU)")
    
    args = parser.parse_args()
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    ATOM_COUNTS = args.atoms
    NUM_RUNS = args.runs
    MAX_ITER = args.max_iter
    MAX_TIME = args.max_time
    N_JOBS = args.n_jobs
    VERBOSE = args.verbose
    BASE_SEED = args.seed
    STOP_TOL = 1e-4
    d = 3
    
    # Configuration backend GPU
    backend_cfg = {
        "use_gpu": not args.no_gpu,
        "gpu_clustering": not args.no_gpu_clustering and not args.no_gpu,
        "gpu_pdist": not args.no_gpu_pdist and not args.no_gpu,
        "gpu_dtype": "float32",
        "pdist_chunk_size": 2000,
        "verbose": False,
        "local_refine_method": args.local_refine,  # lbfgsb, adam, pgd, ipopt
    }
    
    # =========================================================================
    # CRÉATION DU RÉPERTOIRE DE SORTIE
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "optimization.log"
    csv_file = output_dir / "results.csv"
    json_file = output_dir / "results.json"
    summary_file = output_dir / "summary.txt"
    progress_file = output_dir / "progress.txt"
    
    logger = setup_logging(str(log_file))
    
    # =========================================================================
    # AFFICHAGE DE LA CONFIGURATION
    # =========================================================================
    logger.info("=" * 80)
    logger.info("PF-NLP OPTIMIZER FOR LENNARD-JONES (GPU-ENABLED)")
    logger.info("=" * 80)
    logger.info(f"Atoms: {ATOM_COUNTS}")
    logger.info(f"Runs per atom: {NUM_RUNS}")
    logger.info(f"Max iterations: {MAX_ITER}")
    logger.info(f"Max time per run: {MAX_TIME}s")
    logger.info(f"N_jobs: {N_JOBS}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("-" * 40)
    logger.info("Backend Configuration:")
    for k, v in backend_cfg.items():
        logger.info(f"  {k}: {v}")
    
    # Afficher le statut GPU
    print_backend_status(backend_cfg)
    
    config = {
        'atoms': ATOM_COUNTS,
        'num_runs': NUM_RUNS,
        'max_iter': MAX_ITER,
        'max_time': MAX_TIME,
        'n_jobs': N_JOBS,
        'base_seed': BASE_SEED,
        'stop_tol': STOP_TOL,
        'backend_cfg': backend_cfg,
        'timestamp': timestamp,
    }
    
    # =========================================================================
    # CSV HEADER
    # =========================================================================
    csv_headers = ['n', 'run', 'objective', 'best_known', 'gap', 'gap_pct',
                   'iterations', 'obj_evals', 'time_sec', 'success', 'seed_init', 'seed_pf']
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
    
    # =========================================================================
    # BOUCLE PRINCIPALE
    # =========================================================================
    all_results = []
    results_summary = []
    
    for n in ATOM_COUNTS:
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING OPTIMIZATION FOR n={n} ATOMS")
        logger.info(f"{'='*80}")
        
        num_vars = n * d
        best_known = BEST_KNOWN_SOL.get(n, -1e20)
        num_particles = args.particles if args.particles else max(3 * n + 56, 50)
        refine_top_k = args.refine_top_k
        ipopt_refine_all=args.ipopt_refine_all
        
        logger.info(f"  Variables: {num_vars}")
        logger.info(f"  Best known: {best_known}")
        logger.info(f"  Particles: {num_particles}")
        
        run_results = []
        
        for m in range(1, NUM_RUNS + 1):
            run_start = time.time()
            seed_init = make_seed(BASE_SEED, n, m, offset=0)
            seed_pf = make_seed(BASE_SEED, n, m, offset=1000)
            
            rng = np.random.default_rng(seed_init)
            # NOTE: np.random.seed removed — use only rng for reproducibility
            
            lb = -5.0 * np.ones(num_vars)
            ub = 5.0 * np.ones(num_vars)
            margin = 0.1 * (ub - lb)
            x0 = rng.uniform(lb + margin, ub - margin, size=num_vars)
            
            pf_solver = PF_NLP_Solver(
                model_builder=build_lj_model,
                obj_func=lj_obj_func,
                x0=x0, lb=lb, ub=ub,
                model_params={'n': n, 'd': d},
                pf_max_iter=MAX_ITER,
                best_known_obj=best_known,
                ipopt_refine_all=ipopt_refine_all,
                num_particles=num_particles,
                ipopt_options={"tol": 1e-8, "print_level": 0, "hessian_approximation": "exact"},
                epsilon_stop=STOP_TOL,
                refine_top_k=refine_top_k,
                use_cma=False,
                use_fast_flow=False,
                use_invertible_flow=True,
                use_differentiable_pf=False,
                mapping_method='linear',
                clustering_algo="kmeans",
                n_jobs=N_JOBS,
                output_dir=output_dir,
                max_time=MAX_TIME,
                # === NOUVEAU: Configuration backend GPU ===
                backend_cfg=backend_cfg
            )
            
            with open(progress_file, 'a') as f:
                f.write(f"\n[n={n}][Run {m}/{NUM_RUNS}] Starting optimization...\n")
                f.flush()
            
            logger.info(f"  [n={n}][Run {m:3d}/{NUM_RUNS}] Starting optimization...")
            
            best_solution, best_obj = pf_solver.solve_particle_filter(
                seed=seed_pf, verbose=VERBOSE, min_particles=5, run_num=m,
                progress_file=str(progress_file)
            )
            
            run_time = time.time() - run_start
            gap = abs(best_obj - best_known) if best_known > -1e10 else None
            gap_pct = (gap / abs(best_known) * 100) if gap and best_known != 0 else None
            success = gap < 1e-4 if gap else False
            
            run_result = {
                'n': n,
                'run': m,
                'objective': best_obj,
                'best_known': best_known if best_known > -1e10 else None,
                'gap': gap,
                'gap_pct': gap_pct,
                'iterations': pf_solver.pf_max_iter,
                'obj_evals': pf_solver.obj_eval_count,
                'time_sec': run_time,
                'success': success,
                'seed_init': seed_init,
                'seed_pf': seed_pf
            }
            run_results.append(run_result)
            all_results.append(run_result)
            
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writerow(run_result)
            
            status = "✓" if success else "✗"
            logger.info(f"  [n={n}][Run {m:3d}/{NUM_RUNS}] obj={best_obj:14.6f} | gap={gap:12.6f} | evals={pf_solver.obj_eval_count:8d} | {run_time:6.1f}s | {status}")
            
            with open(progress_file, 'a') as f:
                f.write(f"[DONE] n={n} Run {m}/{NUM_RUNS} | obj={best_obj:.6f} | gap={gap:.6f} | {run_time:.1f}s | {status}\n")
                f.flush()
        
        # Résumé pour ce n
        objs = [r['objective'] for r in run_results]
        gaps = [r['gap'] for r in run_results if r['gap'] is not None]
        evals = [r['obj_evals'] for r in run_results]
        times = [r['time_sec'] for r in run_results]
        success_count = sum(1 for r in run_results if r['success'])
        
        summary = {
            'n': n,
            'best': min(objs),
            'worst': max(objs),
            'mean': np.mean(objs),
            'std': np.std(objs),
            'median': np.median(objs),
            'success_count': success_count,
            'success_rate': success_count / NUM_RUNS,
            'avg_gap': np.mean(gaps) if gaps else None,
            'avg_evals': np.mean(evals),
            'avg_time': np.mean(times),
            'total_time': sum(times)
        }
        results_summary.append(summary)
        
        logger.info(f"  --- Summary n={n}: best={summary['best']:.6f}, success={success_count}/{NUM_RUNS} ({summary['success_rate']*100:.1f}%), avg_time={summary['avg_time']:.1f}s")
    
    # =========================================================================
    # SAUVEGARDER RÉSULTATS COMPLETS (JSON)
    # =========================================================================
    final_results = {
        'config': config,
        'summary': results_summary,
        'all_runs': all_results
    }
    with open(json_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # =========================================================================
    # RÉSUMÉ FINAL
    # =========================================================================
    summary_lines = []
    summary_lines.append("=" * 90)
    summary_lines.append("FINAL SUMMARY (GPU-ENABLED)")
    summary_lines.append("=" * 90)
    summary_lines.append(f"{'n':>5} | {'Best':>14} | {'Known':>14} | {'Gap':>12} | {'Success':>10} | {'Avg Evals':>10} | {'Avg Time':>10}")
    summary_lines.append("-" * 90)
    
    for s in results_summary:
        known = BEST_KNOWN_SOL.get(s['n'], None)
        gap = abs(s['best'] - known) if known else float('nan')
        known_str = f"{known:.6f}" if known else "N/A"
        summary_lines.append(
            f"{s['n']:>5} | {s['best']:>14.6f} | {known_str:>14} | {gap:>12.6f} | "
            f"{s['success_count']:>4}/{NUM_RUNS:<4} | {s['avg_evals']:>10.0f} | {s['avg_time']:>9.1f}s"
        )
    
    summary_lines.append("=" * 90)
    summary_lines.append(f"Total runs: {len(all_results)}")
    summary_lines.append(f"Output directory: {output_dir}")
    summary_lines.append(f"Backend: GPU={backend_cfg['use_gpu']}, Clustering={backend_cfg['gpu_clustering']}, pdist={backend_cfg['gpu_pdist']}")
    summary_lines.append("=" * 90)
    
    summary_text = "\n".join(summary_lines)
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    
    for line in summary_lines:
        logger.info(line)
    
    logger.info(f"")
    logger.info(f"Results saved to:")
    logger.info(f"  - {csv_file}")
    logger.info(f"  - {json_file}")
    logger.info(f"  - {summary_file}")
    logger.info(f"  - {log_file}")
    logger.info(f"")
    logger.info("OPTIMIZATION COMPLETED")
