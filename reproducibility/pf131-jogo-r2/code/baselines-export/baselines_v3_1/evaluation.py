"""One run-owned counter and incumbent, shared by all attempts and local solvers."""
from __future__ import annotations
import math
import time
from collections import Counter
import numpy as np

class RunStop(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason

class BudgetExceeded(RunStop):
    def __init__(self, message='evaluation_limit'):
        super().__init__('evaluation_limit')

class NumericalFailure(RuntimeError):
    pass

def target_met(energy, target, tolerance):
    return target is not None and math.isfinite(energy) and abs(energy-target) <= tolerance

class EvaluationCounter:
    def __init__(self, oracle, max_evals, *, bounds=5., target=None, tolerance=1e-4,
                 job_seconds=0., chunk_size=32, event_callback=None, clock=time.monotonic):
        if isinstance(max_evals,bool) or int(max_evals) != max_evals or max_evals < 1:
            raise ValueError('max_evals must be a positive integer')
        if not math.isfinite(bounds) or bounds <= 0 or isinstance(chunk_size,bool) or int(chunk_size)!=chunk_size or chunk_size < 1:
            raise ValueError('Invalid bounds/chunk size')
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError('Invalid target tolerance')
        if not math.isfinite(job_seconds) or job_seconds<0:
            raise ValueError('Invalid job duration')
        if target is not None and not math.isfinite(target):
            raise ValueError('Target must be finite or None')
        self.oracle, self.max_evals = oracle, int(max_evals)
        self.bounds, self.target, self.tolerance = float(bounds), target, float(tolerance)
        self.chunk_size, self.clock = int(chunk_size), clock
        self.started = clock()
        self.deadline = self.started + job_seconds if job_seconds > 0 else math.inf
        self.attempt_deadline = math.inf
        self.count = self.gradient_calls = self.audit_calls = 0
        self.count_by_origin = Counter(global_search=0, local=0)
        self.nonfinite_energy_count = 0
        self.best_energy, self.best_x, self.best_eval = math.inf, None, None
        self.first_target_eval = self.first_target_seconds = None
        self.attempt = self.iteration = 0
        self.last_reason = None
        self.interrupted = False
        self.callback = event_callback
        self.local_status_counts, self.local_status_examples = Counter(), []
        self.local_calls = self.local_iterations = 0
        self.local_diagnostics_summary = Counter()
        self.local_anomaly_examples = []
        self.stopped = False

    @property
    def remaining(self):
        return self.max_evals-self.count

    def begin_attempt(self, index, seconds=0.):
        self.attempt, self.iteration = int(index), 0
        self.attempt_deadline = min(self.deadline, self.clock()+seconds) if seconds > 0 else self.deadline
        self.last_reason = None

    def guard(self, check_budget=True):
        reason = None
        if self.interrupted:
            reason = 'interrupted'
        elif self.first_target_eval is not None:
            reason = 'target_reached'
        elif check_budget and self.count >= self.max_evals:
            self.stopped = True
            reason = 'evaluation_limit'
        elif self.clock() >= self.deadline:
            reason = 'time_limit'
        elif self.clock() >= self.attempt_deadline:
            reason = 'attempt_time_limit'
        if reason:
            self.last_reason = reason
            if reason == 'evaluation_limit':
                raise BudgetExceeded()
            raise RunStop(reason)

    def _validate(self, x):
        a = np.asarray(x, dtype=np.float64)
        if a.shape != (3*self.oracle.n_atoms,) or not np.isfinite(a).all():
            raise NumericalFailure('Nonfinite or incorrectly shaped candidate')
        if np.any(np.abs(a)>self.bounds+1e-12):
            raise NumericalFailure('Candidate outside the declared box')
        return a

    def _record(self, x, e, index, origin):
        e = float(e)
        if not math.isfinite(e):
            self.nonfinite_energy_count += 1
            return
        if self.target is not None and e < self.target-self.tolerance:
            # Still retain the point; the independent audit will report inconsistency.
            pass
        if e < self.best_energy:
            self.best_energy, self.best_x, self.best_eval = e, x.copy(), int(index)
            if self.callback:
                self.callback(dict(event='incumbent', evaluation=index, energy=e,
                                   seconds=self.clock()-self.started, attempt=self.attempt,
                                   iteration=self.iteration, origin=origin))
        if self.first_target_eval is None and target_met(e, self.target, self.tolerance):
            self.first_target_eval, self.first_target_seconds = int(index), self.clock()-self.started

    def __call__(self, x, n_atoms=None, *, origin='global_search'):
        self.guard()
        if origin not in ('global_search','local'):
            raise ValueError('Invalid evaluation origin')
        a = self._validate(x)
        self.count += 1
        self.count_by_origin[origin] += 1
        value = self.oracle.value(a)
        self._record(a, value, self.count, origin)
        self.guard(check_budget=False)
        return value

    def batch(self, X, *, origin='global_search'):
        if origin not in ('global_search','local'):
            raise ValueError('Invalid evaluation origin')
        a = np.asarray(X, dtype=np.float64)
        if a.ndim!=2 or a.shape[1]!=3*self.oracle.n_atoms:
            raise NumericalFailure('Invalid batch shape')
        values = []
        for start in range(0,len(a),self.chunk_size):
            self.guard()
            stop = min(len(a),start+self.chunk_size, start+self.remaining)
            b = a[start:stop]
            if not np.isfinite(b).all() or np.any(np.abs(b)>self.bounds+1e-12):
                raise NumericalFailure('Invalid batch coordinates')
            old = self.count
            self.count += len(b)
            self.count_by_origin[origin] += len(b)
            es = np.asarray(self.oracle.batch(b),dtype=np.float64)
            if es.shape!=(len(b),):
                raise NumericalFailure('Incorrect oracle output shape')
            for j,(x,e) in enumerate(zip(b,es)):
                self._record(x,e,old+j+1,origin)
            values.extend(es.tolist())
            self.guard(check_budget=False)
            if stop<min(len(a), start+self.chunk_size):
                self.stopped=True
                raise BudgetExceeded()
        return np.asarray(values,dtype=np.float64)

    def gradient(self, x):
        self.guard()
        self._validate(x)
        self.gradient_calls += 1
        return self.oracle.gradient(x)

    def record_local_status(self, status, nit=0, message='', details=None):
        self.local_calls += 1
        self.local_iterations += int(nit)
        self.local_status_counts[str(status)] += 1
        if details is not None:
            for key in ('stages','recovery_restarts','recovery_evaluations'):
                self.local_diagnostics_summary[key] += int(details.get(key, 0))
            for key in ('scaling_used','stationary','diagnostics_complete','unresolved_large_residual'):
                self.local_diagnostics_summary[key] += int(bool(details.get(key, False)))
            if details.get('diagnostics_complete') and details.get('total_displacement_inf') == 0 and not details.get('stationary'):
                self.local_diagnostics_summary['zero_displacement_nonstationary'] += 1
                if details.get('unresolved_large_residual'):
                    self.local_diagnostics_summary['zero_displacement_large_residual'] += 1
            if details.get('diagnostics_complete') and details.get('unresolved_large_residual') and len(self.local_anomaly_examples)<30:
                self.local_anomaly_examples.append(dict(status=str(status),nit=int(nit),message=str(message),diagnostics=details))
        if len(self.local_status_examples)<30:
            entry = dict(status=str(status),nit=int(nit),message=str(message))
            if details is not None: entry['diagnostics'] = details
            self.local_status_examples.append(entry)

    def audit(self):
        if self.best_x is None:
            return dict(passed=False, reason='no_finite_incumbent', energy=None, search_energy=None)
        self.audit_calls += 1
        e = self.oracle.reference(self.best_x)
        err = abs(e-self.best_energy)
        passed = (math.isfinite(e) and np.isfinite(self.best_x).all() and
                  np.max(np.abs(self.best_x)) <= self.bounds+1e-12 and
                  err <= 1e-8 + 1e-12*abs(e))
        consistent = self.target is None or e >= self.target-self.tolerance
        return dict(passed=bool(passed and consistent), energy=float(e) if math.isfinite(e) else None,
                    search_energy=self.best_energy, absolute_difference=err if math.isfinite(err) else None,
                    target_consistent=bool(consistent), max_abs_coordinate=float(np.max(np.abs(self.best_x))),
                    audit_evaluations=self.audit_calls,
                    reason='ok' if passed and consistent else 'energy_or_target_inconsistency')
