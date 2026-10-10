"""Bounded L-BFGS-B with numerical scaling and explicit stationarity diagnostics.

No new potential, target, domain, global move, seed or budget is introduced.
An optional projected-gradient Armijo recovery is used only after a nonstationary
local exit. Every recovery trial is a counted local energy evaluation.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize
from evaluation import EvaluationCounter, RunStop, NumericalFailure
from lj_oracle import LJOracle

POLICY_ID = 'scaled-lbfgsb-guarded-v3.1.0'
POLICY = dict(id=POLICY_ID, initial_max_component_step=0.1,
              initial_min_distance_fraction=0.1, recovery_restarts=8,
              recovery_residual_trigger=1e-3, armijo_c1=1e-4,
              armijo_max_backtracks=30, true_stationarity='unscaled gradient mapping <= local_gtol')


def projected_gradient_inf(x, g, bounds):
    """Unit-step gradient mapping for the actual box and UNSCALED gradient."""
    return float(np.max(np.abs(x - np.clip(x-g, -bounds, bounds))))


def _distance_and_step(x, n_atoms):
    r = x.reshape(n_atoms, 3)
    d = r[:, None, :] - r[None, :, :]
    dd = np.einsum('ijk,ijk->ij', d, d)
    dd[np.diag_indices(n_atoms)] = np.inf
    rmin = float(np.sqrt(np.min(dd)))
    # The initial max coordinate displacement is at most 10% of min pair
    # separation, capped at 0.1. The positive floor is numerical only.
    cap = max(1e-8, min(0.1, 0.1*rmin))
    return rmin, cap


def guarded_local_minimize(x0, n_atoms, bounds, maxiter, remove_com=False,
                           counter=None, ftol=1e-12, gtol=1e-10,
                           maxls=30, maxcor=10):
    """Same (best_x, best_energy, counted_objectives) API as v3.

    Positive energy/gradient scaling is fixed within each L-BFGS-B invocation.
    The objective itself and ledger energies remain unchanged. Both requested
    tolerances are scaled down, never relaxed. A function-reduction termination
    is labelled separately from verified projected-gradient stationarity.
    """
    if not (int(maxiter) == maxiter and maxiter >= 1 and bounds > 0):
        raise ValueError('Invalid local bounds/iteration limit')
    if not (math.isfinite(ftol) and ftol >= 0 and math.isfinite(gtol) and gtol >= 0):
        raise ValueError('Invalid local tolerances')
    if counter is None:
        counter = EvaluationCounter(LJOracle(n_atoms), 10000000, bounds=bounds)
    x0 = np.asarray(x0, dtype=float).ravel().copy()
    if x0.shape != (3*n_atoms,) or not np.isfinite(x0).all():
        raise NumericalFailure('Invalid local initial coordinates')
    if remove_com:
        r = x0.reshape(n_atoms,3)
        x0 = (r-r.mean(0)).ravel()
    x0 = np.clip(x0, -bounds, bounds)
    start_count = counter.count
    best_x = None
    best_e = math.inf
    status = 'exception'
    message = ''
    nit = 0
    details = dict(policy=POLICY_ID, stages=0, scaling_used=False, recovery_restarts=0,
                   recovery_evaluations=0, initial_energy=None, final_energy=None,
                   initial_gradient_inf=None, final_gradient_inf=None,
                   projected_gradient_inf=None, total_displacement_inf=None,
                   solver_success=None, stationary=False, diagnostics_complete=False,
                   unresolved_large_residual=False, stage_diagnostics=[])
    # Gradient memoization stores completed derivative calls only. All new
    # derivatives go through the shared ledger and remain separately counted.
    grad_x = grad_value = None
    def grad(x):
        nonlocal grad_x, grad_value
        counter.guard()
        if grad_x is not None and np.array_equal(x, grad_x):
            return grad_value
        g = np.asarray(counter.gradient(x), dtype=float)
        if g.shape != x0.shape or not np.isfinite(g).all():
            raise NumericalFailure('Nonfinite local gradient')
        grad_x, grad_value = np.asarray(x).copy(), g.copy()
        return g
    def energy(x):
        nonlocal best_x, best_e
        e = float(counter(x, origin='local'))
        if math.isfinite(e) and e < best_e:
            best_x, best_e = np.asarray(x).copy(), e
        if details['initial_energy'] is None:
            details['initial_energy'] = e if math.isfinite(e) else None
        return e
    def complete_diag(x, g):
        pg = projected_gradient_inf(x, g, bounds)
        details.update(final_energy=best_e, final_gradient_inf=float(np.max(np.abs(g))),
                       projected_gradient_inf=pg, stationary=bool(pg <= gtol),
                       total_displacement_inf=float(np.max(np.abs(x-x0))),
                       diagnostics_complete=True,
                       unresolved_large_residual=bool(pg > POLICY['recovery_residual_trigger']))
        return pg
    try:
        x = x0.copy()
        for stage in range(POLICY['recovery_restarts']+1):
            counter.guard()
            g0 = grad(x)
            ginf = float(np.max(np.abs(g0)))
            if details['initial_gradient_inf'] is None:
                details['initial_gradient_inf'] = ginf
            rmin, cap = _distance_and_step(x, n_atoms)
            scale = max(1.0, ginf/cap)
            if not math.isfinite(scale):
                raise NumericalFailure('Invalid local scale')
            details['scaling_used'] |= scale > 1.0
            remaining_it = int(maxiter)-nit
            if remaining_it <= 0:
                status = 'iteration_limit'; break
            before_x = x.copy()
            before_count = counter.count
            # ftol/scale is conservative relative to the unscaled ftol because
            # scale>=1. This prevents a large scale from weakening ftol.
            res = minimize(lambda a: energy(a)/scale, x,
                           jac=lambda a: grad(a)/scale, method='L-BFGS-B',
                           bounds=[(-bounds,bounds)]*(3*n_atoms),
                           options=dict(maxiter=remaining_it, maxfun=max(1,counter.remaining),
                                        ftol=float(ftol)/scale, gtol=float(gtol)/scale,
                                        maxls=int(maxls), maxcor=int(maxcor)))
            nit += int(res.nit)
            details['stages'] += 1
            details['solver_success'] = bool(res.success)
            message = str(res.message)
            if best_x is None:
                raise NumericalFailure('Local solve evaluated no finite candidate')
            # Check the exact point we return, not res.x if another evaluated
            # candidate was better. No post-solve clipping or uncounted energy.
            x = best_x.copy()
            g = grad(x)
            pg = complete_diag(x, g)
            details['stage_diagnostics'].append(dict(stage=stage+1, scale=scale,
                start_min_pair_distance=rmin, energy=best_e, nit=int(res.nit),
                counted_evaluations=counter.count-before_count, solver_message=message,
                solver_success=bool(res.success), projected_gradient_inf=pg,
                displacement_inf=float(np.max(np.abs(x-before_x)))))
            if pg <= gtol:
                status = 'stationary'; break
            if pg <= POLICY['recovery_residual_trigger']:
                status = 'function_tolerance' if res.success else 'nonstationary_exit'; break
            if nit >= int(maxiter):
                status = 'iteration_limit'; break
            if stage >= POLICY['recovery_restarts']:
                status = 'unresolved_nonstationary'; break
            # Counted, bounded Armijo safeguard. This is a local numerical
            # recovery, NOT a new random/global proposal or extra budget.
            rmin, cap = _distance_and_step(x, n_atoms)
            direction = -g.copy()
            direction[(x <= -bounds) & (direction < 0)] = 0
            direction[(x >= bounds) & (direction > 0)] = 0
            dn = float(np.max(np.abs(direction)))
            if dn == 0:
                status = 'unresolved_nonstationary'; break
            direction /= dn
            alpha = cap
            e0 = best_e
            accepted = False
            for _ in range(POLICY['armijo_max_backtracks']):
                trial = np.clip(x+alpha*direction, -bounds, bounds)
                step = trial-x
                if not np.any(step): break
                slope = float(np.dot(g,step))
                if not math.isfinite(slope) or slope >= 0: break
                trial_e = energy(trial)
                details['recovery_evaluations'] += 1
                if math.isfinite(trial_e) and trial_e < e0 and trial_e <= e0+POLICY['armijo_c1']*slope:
                    x = trial
                    accepted = True
                    nit += 1  # safeguard step also uses the local iteration allowance
                    break
                alpha *= 0.5
            if not accepted:
                status = 'unresolved_nonstationary'; break
            details['recovery_restarts'] += 1
        if best_x is None:
            raise NumericalFailure('Local solve returned no finite incumbent')
        # A last recovery step may change the returned incumbent; audit its
        # derivative for classification (no energy call outside the ledger).
        complete_diag(best_x, grad(best_x))
        if details['stationary']: status = 'stationary'
        if details['unresolved_large_residual']:
            status = 'unresolved_nonstationary'
        return best_x, best_e, counter.count-start_count
    except RunStop as exc:
        status, message = exc.reason, str(exc)
        details.update(diagnostics_complete=False, stationary=False, unresolved_large_residual=False)
        # The run-owned ledger has all evaluated energies, including its last
        # value if a budget/target/deadline interrupt occurred inside energy().
        raise
    finally:
        counter.record_local_status(status, nit, message, details=details)
