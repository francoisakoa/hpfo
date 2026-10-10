"""Float64 Lennard-Jones oracles, without hidden changes of length convention.

pf131: q=(r^2+1e-8)^-1; pair=q^3*(q^3-2), matching provided PF C++ 1.3.1.
standard: 4*(r^-12-r^-6), provided for explicit, separately labelled experiments.
No clipping/centering is done inside an energy or a derivative.
"""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np

MODELS = ('pf131', 'standard')

@lru_cache(maxsize=64)
def pairs(n):
    # Same i,j ordering as src/problem.cpp (j < i).
    i, j = np.tril_indices(n, -1)
    return i, j

def coordinates(x, n_atoms=None):
    a = np.asarray(x, dtype=np.float64)
    n = int(n_atoms) if n_atoms is not None else a.size // 3
    if n < 2 or a.size != 3*n or not np.isfinite(a).all():
        raise ValueError('Coordinates must contain 3*N finite values, N >= 2')
    return a.reshape(n, 3)

class LJOracle:
    def __init__(self, n_atoms, model='pf131', backend='cpu'):
        self.n_atoms = int(n_atoms)
        if self.n_atoms < 2 or model not in MODELS or backend not in ('cpu', 'cupy'):
            raise ValueError('Invalid atoms, potential or backend')
        self.model, self.backend = model, backend
        self.i, self.j = pairs(self.n_atoms)
        self.cp = None
        self.device = None
        if backend == 'cupy':
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError('No CUDA device')
                self.cp = cp
                self.gi, self.gj = cp.asarray(self.i), cp.asarray(self.j)
                props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
                name = props.get('name', b'unknown')
                self.device = name.decode() if isinstance(name, bytes) else str(name)
                # Fail before starting a run rather than silently fall back.
                cp.zeros(1, dtype=cp.float64).sum().item()
            except Exception as exc:
                raise RuntimeError('CuPy/CUDA requested but unavailable: '+str(exc)) from exc

    def _terms(self, r2, xp):
        if self.model == 'pf131':
            q = 1.0 / (r2 + 1e-8)
            q3 = q*q*q
            return q3*(q3-2.0)
        q = 1.0 / r2
        q3 = q*q*q
        return 4.0*q3*(q3-1.0)

    def value(self, x):
        r = coordinates(x, self.n_atoms)
        d = r[self.i] - r[self.j]
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            terms = self._terms(np.sum(d*d, axis=1), np)
            value = float(np.sum(terms, dtype=np.float64))
        # +inf denotes a rejected collision/overflow, never a finite replacement.
        return value if math.isfinite(value) else math.inf

    def batch(self, X):
        a = np.asarray(X, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 3*self.n_atoms or not np.isfinite(a).all():
            raise ValueError('Batch must be a finite (B,3*N) array')
        if not len(a):
            return np.empty(0, dtype=np.float64)
        xp = np if self.cp is None else self.cp
        c = xp.asarray(a, dtype=xp.float64).reshape(-1, self.n_atoms, 3)
        i, j = (self.i, self.j) if self.cp is None else (self.gi, self.gj)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            d = c[:, i, :] - c[:, j, :]
            e = xp.sum(self._terms(xp.sum(d*d, axis=2), xp), axis=1, dtype=xp.float64)
        v = np.asarray(e) if self.cp is None else self.cp.asnumpy(e)
        v = np.asarray(v, dtype=np.float64)
        v[~np.isfinite(v)] = np.inf
        return v

    def gradient(self, x):
        r = coordinates(x, self.n_atoms)
        d = r[self.i]-r[self.j]
        r2 = np.sum(d*d, axis=1)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            q = 1.0/(r2 + (1e-8 if self.model == 'pf131' else 0.0))
            q3 = q*q*q
            # d/dx, not the force = -gradient.
            factor = 12.0*q*q3*(1.0-q3) if self.model == 'pf131' else 24.0*q*q3*(1.0-2.0*q3)
            term = factor[:, None]*d
        g = np.zeros_like(r)
        np.add.at(g, self.i, term)
        np.add.at(g, self.j, -term)
        if not np.isfinite(g).all():
            raise FloatingPointError('Nonfinite analytic LJ gradient')
        return g.ravel()

    def reference(self, x):
        """Independent scalar/Kahan reporting audit; not a search call."""
        r = coordinates(x, self.n_atoms)
        total = comp = 0.0
        try:
            for i in range(1, self.n_atoms):
                for j in range(i):
                    r2 = sum(float(r[i,k]-r[j,k])**2 for k in range(3))
                    q = 1.0/(r2+(1e-8 if self.model == 'pf131' else 0.0))
                    q3 = q*q*q
                    term = (q3*(q3-2.0) if self.model == 'pf131' else 4.0*q3*(q3-1.0)) - comp
                    s = total + term
                    comp = (s-total)-term
                    total = s
            return total if math.isfinite(total) else math.inf
        except (ZeroDivisionError, OverflowError):
            return math.inf

    def metadata(self):
        return dict(model=self.model, energy_dtype='float64', gradient_dtype='float64',
                    scalar_backend='numpy-cpu', batch_backend=self.backend, device=self.device,
                    r2_regularization=1e-8 if self.model == 'pf131' else 0.0,
                    energy_formula='sum(q^3*(q^3-2)); q=1/(r^2+1e-8)' if self.model=='pf131'
                    else 'sum(4*(r^-12-r^-6))')

def lennard_jones_energy(x, n_atoms=None, model='pf131'):
    n = len(np.asarray(x).ravel())//3 if n_atoms is None else n_atoms
    return LJOracle(n, model).value(x)

def lennard_jones_gradient(x, n_atoms=None, model='pf131'):
    n = len(np.asarray(x).ravel())//3 if n_atoms is None else n_atoms
    return LJOracle(n, model).gradient(x)
