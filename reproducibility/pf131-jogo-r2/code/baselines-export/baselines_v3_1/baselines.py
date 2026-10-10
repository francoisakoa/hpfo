"""Five audited baseline implementations. Shared float64 oracle and run ledger.

Algorithmic families from v2 are retained. Repairs and protocol changes are
listed in CHANGELOG.md, especially SMC resample/move and initialisation.
No monkey-patching or separate GPU random dynamics are used.
"""
from __future__ import annotations
import math
import time
import hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from lj_oracle import LJOracle, lennard_jones_energy, lennard_jones_gradient
from evaluation import (EvaluationCounter, RunStop, BudgetExceeded,
                        NumericalFailure, target_met)

VERSION='3.1.0'
METHODS=['BH-OJ','GA','PSO','CLPSO','SMC']
KNOWN_MINIMA={38:-173.928427,75:-397.492331,76:-402.894866,77:-409.083517,
              98:-543.665361,102:-569.363653,103:-575.766131,104:-582.086642}
DEFAULT_CONVERGENCE_TOL=1e-4

DEFAULTS={
 'BH-OJ':dict(n_iter=1000000,temperature=1.,stepsize=.5,stagnation_trigger=60,
              jump_len=40,jump_stepsize=3.5,local_maxiter=4000,postjump_local_maxiter=1200,
              jump_restart_after=2,restart_collapse=.75,postjump_collapse=.85,
              local_ftol=1e-12,local_gtol=1e-10,local_maxls=30,local_maxcor=10),
 'GA':dict(pop_size=300,n_generations=1000000,crossover_rate=.7,mutation_rate=.1,
           elite_frac=.05,refine_every=25,local_maxiter=5000,eta_c=15.,eta_m=20.,tournament_k=3,
           local_ftol=1e-12,local_gtol=1e-10,local_maxls=30,local_maxcor=10),
 'PSO':dict(n_particles=256,n_iterations=1000000,w=.7298,c1=1.49618,c2=1.49618,vmax=None),
 'CLPSO':dict(n_particles=256,n_iterations=1000000,w_max=.9,w_min=.4,c=1.49445,
              refresh_gap=7,inertia_schedule='evaluation_budget'),
 'SMC':dict(n_particles=256,n_iterations=1000000,beta_min=0.,beta_max=8.,
            ess_target_frac=.7,mcmc_steps=2,mcmc_sigma0=.6,mcmc_sigma_floor_frac=.15),
}

def systematic_resample(w,rng):
    w=np.asarray(w,dtype=np.float64)
    if w.ndim!=1 or not len(w) or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise NumericalFailure('Invalid resampling weights')
    w=w/w.sum()
    c=np.cumsum(w);c[-1]=1.
    positions=(rng.random()+np.arange(len(w)))/len(w)
    return np.searchsorted(c,positions,side='right')

def reflect_box(x,bounds):
    # Symmetric reflected Gaussian proposal on a box; unlike clipping it has
    # no boundary atoms. No centre-of-mass projection is composed with MH.
    width=2.*bounds
    t=np.mod(np.asarray(x)+bounds,2.*width)
    return -bounds+np.where(t<=width,t,2.*width-t)

def normalized_logweights(E,dbeta,logw=None):
    E=np.asarray(E,dtype=np.float64)
    if E.ndim!=1 or not len(E) or not np.isfinite(E).all() or dbeta<0 or not math.isfinite(dbeta):
        raise NumericalFailure('Invalid tempering energies or increment')
    base=np.full(len(E),-math.log(len(E))) if logw is None else np.asarray(logw,dtype=float)
    with np.errstate(over='ignore',invalid='raise'):
        s=base-dbeta*(E-E.min())
    z=logsumexp(s)
    if not math.isfinite(z):
        raise NumericalFailure('Invalid log-weight normalizer')
    return s-z

def effective_size_from_logweights(logw):
    return float(np.exp(-logsumexp(2.*np.asarray(logw))))

def adaptive_beta(beta,E,beta_max,ess_target_frac):
    """Uniform-weight resample/move SMC: choose beta from the same weights used.

    Bisection is done in a dimensionless variable, avoiding a hard 1e-6
    temperature increment on enormous initial energies. Return next beta and
    whether rounding forced an advance by one representable float.
    """
    E=np.asarray(E,dtype=float)
    if not np.isfinite(E).all():
        raise NumericalFailure('SMC needs finite current energies')
    if beta>=beta_max:
        return float(beta_max),False
    scale=float(np.max(E-E.min()))
    if scale==0:
        return float(beta_max),False
    scaled=(E-E.min())/scale
    cap=(beta_max-beta)*scale
    target=ess_target_frac*len(E)
    def ess(t):
        lw=-t*scaled;lw-=logsumexp(lw)
        return effective_size_from_logweights(lw)
    if ess(cap)>=target:
        return float(beta_max),False
    lo=0.;hi=min(1.,cap)
    while ess(hi)>target:
        lo=hi;hi=min(2.*hi,cap)
    for _ in range(65):
        mid=.5*(lo+hi)
        if ess(mid)>=target:lo=mid
        else:hi=mid
    candidate=min(beta_max,beta+lo/scale)
    rounded=candidate<=beta
    if rounded:candidate=float(np.nextafter(beta,beta_max))
    return candidate,rounded


from local_refinement import guarded_local_minimize as _local_minimize

class _Algorithm:
    method=''
    def __init__(self,n_atoms,*,counter=None,max_evals=1000000,bounds=5.,seed=None,
                 initialization='box',remove_com=False,model='pf131',backend='cpu',
                 max_time=0.,verbose=False,use_gpu=None,**params):
        allowed=DEFAULTS[self.method]
        unknown=set(params)-set(allowed)
        if unknown:raise TypeError(f'{self.method}: unknown parameters: {sorted(unknown)}')
        self.p={**allowed,**params}
        self.n_atoms=int(n_atoms);self.dim=3*self.n_atoms;self.bounds=float(bounds)
        if initialization not in ('box','sphere'):raise ValueError('Unknown initialization')
        self.initialization=initialization;self.remove_com=bool(remove_com)
        if self.method=='SMC' and (initialization!='box' or remove_com):
            raise ValueError('This SMC target is on a box: require box initialization and remove_com=False')
        if use_gpu is not None:
            desired='cupy' if use_gpu else 'cpu'
            if backend!='cpu' and backend!=desired:raise ValueError('Conflicting backend/use_gpu')
            backend=desired
        self.rng=np.random.Generator(np.random.PCG64(seed))
        self.counter=counter if counter is not None else EvaluationCounter(
            LJOracle(self.n_atoms,model,backend),max_evals,bounds=bounds,job_seconds=max_time)
        if self.counter.oracle.n_atoms!=self.n_atoms or self.counter.bounds!=self.bounds:
            raise ValueError('Counter and optimizer domains disagree')
        self.verbose=bool(verbose);self.iteration=0
        self.diagnostics={'initial_population_sha256':None}
        self._validate()

    def _validate(self):
        for k,v in self.p.items():
            if isinstance(v,bool):raise ValueError(f'Boolean is not a numeric parameter: {k}')
            if k not in ('inertia_schedule',) and v is not None and not isinstance(v,(float,int)):
                raise ValueError(f'Numeric parameter required: {k}')
            if isinstance(v,(float,int)) and not isinstance(v,bool) and (not math.isfinite(v) or v<0):
                raise ValueError(f'Invalid {self.method} parameter {k}={v}')
        for k in ('n_iter','n_generations','n_iterations','pop_size','n_particles','local_maxiter',
                  'postjump_local_maxiter','jump_len','stagnation_trigger','jump_restart_after',
                  'refresh_gap','mcmc_steps','tournament_k','local_maxls','local_maxcor'):
            if k in self.p and (int(self.p[k])!=self.p[k] or self.p[k]<1):
                raise ValueError(f'{k} must be a positive integer')
        for k in ('crossover_rate','mutation_rate','elite_frac','ess_target_frac','mcmc_sigma_floor_frac'):
            if k in self.p and not (0<=self.p[k]<=1):raise ValueError(f'Invalid fraction {k}')
        for k in ('pop_size','n_particles'):
            if k in self.p and self.p[k]<3:raise ValueError('Population must have at least 3 particles')
        if 'refine_every' in self.p and int(self.p['refine_every'])!=self.p['refine_every']:
            raise ValueError('refine_every must be a nonnegative integer')
        if self.method=='SMC':
            if self.p['mcmc_sigma_floor_frac']<=0:raise ValueError('Positive SMC sigma floor required')
            if not (0<self.p['ess_target_frac']<1) or self.p['beta_max']<=self.p['beta_min']:
                raise ValueError('Invalid SMC temperature/ESS settings')
            if self.p['mcmc_sigma0']<=0:raise ValueError('SMC sigma must be positive')
        if self.method=='CLPSO':
            if self.p['w_min']>self.p['w_max'] or self.p['inertia_schedule'] not in ('evaluation_budget','iterations'):
                raise ValueError('Invalid inertia schedule')

    def _project(self,x):
        a=np.asarray(x,dtype=float)
        if not np.isfinite(a).all():raise NumericalFailure('Nonfinite proposal coordinates')
        if self.remove_com:
            c=a.reshape(-1,self.n_atoms,3);a=(c-c.mean(axis=1,keepdims=True)).reshape(a.shape)
        return np.clip(a,-self.bounds,self.bounds)

    def _init(self,n):
        if self.initialization=='box':
            X=self.rng.uniform(-self.bounds,self.bounds,size=(n,self.dim))
        else:
            v=self.rng.normal(size=(n,self.n_atoms,3))
            v/=np.linalg.norm(v,axis=2,keepdims=True)
            radius=min(.9*self.bounds,2.2*self.n_atoms**(1/3))
            X=(v*(radius*self.rng.random((n,self.n_atoms,1))**(1/3))).reshape(n,self.dim)
        X=self._project(X)
        if self.diagnostics['initial_population_sha256'] is None:
            self.diagnostics['initial_population_sha256']=hashlib.sha256(np.asarray(X,dtype='<f8').tobytes()).hexdigest()
            self.diagnostics['initial_population_size']=n
        return X

    def _step(self,k):
        self.iteration=int(k);self.counter.iteration=int(k);self.counter.guard()

    def _E(self,x):return self.counter(x)
    def _E_batch(self,X):return self.counter.batch(X)
    def _local(self,x,maxiter=None):
        p=self.p
        return _local_minimize(x,self.n_atoms,self.bounds,maxiter or p['local_maxiter'],
                               self.remove_com,self.counter,p['local_ftol'],p['local_gtol'],
                               p['local_maxls'],p['local_maxcor'])[:2]

    def optimize(self,target_energy=None,target_tol=DEFAULT_CONVERGENCE_TOL):
        self.counter.target=target_energy;self.counter.tolerance=float(target_tol)
        start=time.monotonic();before=self.counter.count
        reason='iteration_limit'
        try:
            reason=self._run() or 'iteration_limit'
            self.counter.guard()
        except RunStop as exc:
            reason=exc.reason
        return dict(x=self.counter.best_x,energy=float(self.counter.best_energy),
                    n_func_evals=self.counter.count-before,n_iterations=self.iteration,
                    convergence_iter=self.counter.iteration if reason=='target_reached' else None,
                    wall_time=time.monotonic()-start,success=target_met(self.counter.best_energy,target_energy,target_tol),
                    stopped_by_budget=reason=='evaluation_limit',stopped_by_time=reason in ('time_limit','attempt_time_limit'),
                    stop_reason=reason,diagnostics=self.diagnostics)

class BasinHoppingOJ(_Algorithm):
    method='BH-OJ'
    def _run(self):
        p=self.p;x=self._init(1)[0];x,e=self._local(x)
        no_improve=failed=0
        for it in range(1,p['n_iter']+1):
            self._step(it);old_best=self.counter.best_energy
            prop=self._project(x+self.rng.normal(0,p['stepsize'],self.dim))
            y,ey=self._local(prop)
            de=ey-e
            if de<=0 or (p['temperature']>0 and math.log(max(self.rng.random(),np.finfo(float).tiny))<-de/p['temperature']):
                x,e=y,ey
            no_improve=0 if self.counter.best_energy<old_best-1e-12 else no_improve+1
            if no_improve>=p['stagnation_trigger']:
                before=self.counter.best_energy;x=self.counter.best_x.copy()
                for _ in range(p['jump_len']):
                    self.counter.guard();x=self._project(x+self.rng.normal(0,p['jump_stepsize'],self.dim));self._E(x)
                c=x.reshape(self.n_atoms,3);mean=c.mean(0)
                x=self._project((mean+p['postjump_collapse']*(c-mean)).ravel())
                x,e=self._local(x,min(p['local_maxiter'],p['postjump_local_maxiter']))
                failed=0 if self.counter.best_energy<before-1e-12 else failed+1
                if failed>=p['jump_restart_after']:
                    x=self._init(1)[0];c=x.reshape(self.n_atoms,3);mean=c.mean(0)
                    x=self._project((mean+p['restart_collapse']*(c-mean)).ravel())
                    x,e=self._local(x,min(p['local_maxiter'],p['postjump_local_maxiter']));failed=0
                    self.diagnostics['internal_restarts']=self.diagnostics.get('internal_restarts',0)+1
                self.diagnostics['jump_cycles']=self.diagnostics.get('jump_cycles',0)+1
                no_improve=0
        return 'iteration_limit'

class GeneticAlgorithm(_Algorithm):
    method='GA'
    def _tournament(self,F):
        ix=self.rng.integers(0,len(F),self.p['tournament_k']);return int(ix[np.argmin(F[ix])])
    def _sbx(self,a,b):
        u=self.rng.random(self.dim);eta=self.p['eta_c'];beta=np.empty_like(u);m=u<=.5
        beta[m]=(2*u[m])**(1/(eta+1));beta[~m]=(1/(2*(1-u[~m])))**(1/(eta+1))
        return self._project(.5*((1+beta)*a+(1-beta)*b)),self._project(.5*((1-beta)*a+(1+beta)*b))
    def _poly_mut(self,x):
        x=x.copy();mask=self.rng.random(self.dim)<self.p['mutation_rate']
        if not mask.any():return x
        y=(x[mask]+self.bounds)/(2*self.bounds);u=self.rng.random(len(y));eta=self.p['eta_m'];power=1/(eta+1)
        d=np.empty_like(y);m=u<.5
        d[m]=(2*u[m]+(1-2*u[m])*(1-y[m])**(eta+1))**power-1
        d[~m]=1-(2*(1-u[~m])+2*(u[~m]-.5)*y[~m]**(eta+1))**power
        x[mask]+=d*(2*self.bounds);return self._project(x)
    def _run(self):
        p=self.p;pop=self._init(p['pop_size']);F=self._E_batch(pop)
        nk=max(1,min(p['pop_size']-1,round(p['elite_frac']*p['pop_size'])))
        for gen in range(1,p['n_generations']+1):
            self._step(gen);idx=np.argsort(F,kind='stable')[:nk];elite=pop[idx].copy()
            if p['refine_every']>0 and gen%p['refine_every']==0:
                for k in range(len(elite)):elite[k],_=self._local(elite[k])
            new=list(elite)
            while len(new)<p['pop_size']:
                a=pop[self._tournament(F)];b=pop[self._tournament(F)]
                children=self._sbx(a,b) if self.rng.random()<p['crossover_rate'] else (a.copy(),b.copy())
                for c in children:
                    if len(new)<p['pop_size']:new.append(self._poly_mut(c))
            pop=np.asarray(new);F=self._E_batch(pop)
        return 'iteration_limit'

class ParticleSwarmOptimization(_Algorithm):
    method='PSO'
    def _run(self):
        p=self.p;X=self._init(p['n_particles']);V=self.rng.normal(0,.1,X.shape)
        F=self._E_batch(X);P=X.copy();PF=F.copy();g=P[np.argmin(PF)].copy()
        for it in range(1,p['n_iterations']+1):
            self._step(it)
            V=p['w']*V+p['c1']*self.rng.random(X.shape)*(P-X)+p['c2']*self.rng.random(X.shape)*(g-X)
            if p['vmax'] is not None:V=np.clip(V,-p['vmax'],p['vmax'])
            X=self._project(X+V);F=self._E_batch(X);m=F<PF;P[m]=X[m];PF[m]=F[m];g=P[np.argmin(PF)].copy()
        return 'iteration_limit'

class ComprehensiveLearningPSO(_Algorithm):
    method='CLPSO'
    def _select_exemplars(self,F):
        n=len(F);d=self.dim;indices=np.tile(np.arange(n)[:,None],(1,d))
        Pc=.05+.45*np.expm1(10*np.arange(n)/(n-1))/np.expm1(10)
        for i in range(n):
            other=np.delete(np.arange(n),i)
            a=self.rng.integers(0,n-1,d);b=(a+self.rng.integers(1,n-1,d))%(n-1)
            a,b=other[a],other[b];winner=np.where(F[a]<F[b],a,b)
            learn=self.rng.random(d)<Pc[i]
            if not learn.any():learn[self.rng.integers(0,d)]=True
            indices[i,learn]=winner[learn]
        return indices
    def _run(self):
        p=self.p;X=self._init(p['n_particles']);V=self.rng.normal(0,.1,X.shape)
        F=self._E_batch(X);P=X.copy();PF=F.copy();ex=self._select_exemplars(PF);stay=np.zeros(len(X),dtype=int)
        cols=np.arange(self.dim)[None,:]
        for it in range(1,p['n_iterations']+1):
            self._step(it)
            progress=self.counter.count/self.counter.max_evals if p['inertia_schedule']=='evaluation_budget' else (it-1)/max(1,p['n_iterations']-1)
            w=p['w_max']-(p['w_max']-p['w_min'])*min(1.,progress)
            V=w*V+p['c']*self.rng.random(X.shape)*(P[ex,cols]-X)
            X=self._project(X+V);F=self._E_batch(X);improved=F<PF;P[improved]=X[improved];PF[improved]=F[improved]
            stay[improved]=0;stay[~improved]+=1;refresh=stay>=p['refresh_gap']
            if refresh.any():ex[refresh]=self._select_exemplars(PF)[refresh];stay[refresh]=0
            self.diagnostics['last_inertia']=float(w)
        return 'iteration_limit'

class SMCOptimizer(_Algorithm):
    method='SMC'
    def _adapt_beta(self,beta,E,w=None):
        if w is not None and not np.allclose(w,np.full(len(w),1/len(w)),rtol=1e-12,atol=1e-14):
            raise ValueError('Resample/move SMC adapts from uniform weights only')
        return adaptive_beta(beta,E,self.p['beta_max'],self.p['ess_target_frac'])[0]
    def _systematic_resample(self,w):return systematic_resample(w,self.rng)
    def _move(self,X,E,beta):
        p=self.p;fraction=beta/p['beta_max'];sigma=p['mcmc_sigma0']*(p['mcmc_sigma_floor_frac']+(1-p['mcmc_sigma_floor_frac'])*(1-fraction))
        for _ in range(p['mcmc_steps']):
            self.counter.guard()
            Y=reflect_box(X+self.rng.normal(0,sigma,X.shape),self.bounds)
            before=self.counter.count
            try:EY=self._E_batch(Y)
            finally:self.diagnostics['mcmc_proposals']=self.diagnostics.get('mcmc_proposals',0)+self.counter.count-before
            with np.errstate(over='ignore',invalid='ignore'):
                ratio=-beta*(EY-E)
            accept=(EY<=E)|(np.log(np.maximum(self.rng.random(len(E)),np.finfo(float).tiny))<ratio)
            accept &= np.isfinite(EY)
            X[accept]=Y[accept];E[accept]=EY[accept]
            self.diagnostics['mcmc_complete_sweep_proposals']=self.diagnostics.get('mcmc_complete_sweep_proposals',0)+len(E)
            self.diagnostics['mcmc_accepts']=self.diagnostics.get('mcmc_accepts',0)+int(accept.sum())
        return X,E
    def _run(self):
        p=self.p;X=self._init(p['n_particles']);E=self._E_batch(X)
        if not np.isfinite(E).all():raise NumericalFailure('Initial SMC population contains nonfinite energies')
        beta=0.;self.diagnostics['resampling_policy']='every_tempering_stage'
        for it in range(1,p['n_iterations']+1):
            self._step(it)
            if it==1 and p['beta_min']>0:
                bnew=p['beta_min'];rounded=False
            else:bnew,rounded=adaptive_beta(beta,E,p['beta_max'],p['ess_target_frac'])
            lw=normalized_logweights(E,bnew-beta)
            ess=effective_size_from_logweights(lw)
            idx=self._systematic_resample(np.exp(lw));X=X[idx].copy();E=E[idx].copy();beta=bnew
            self.diagnostics.update(beta_final=float(beta),ess_pre_resample=float(ess),stages=it)
            self.diagnostics['roundoff_temperature_advances']=self.diagnostics.get('roundoff_temperature_advances',0)+int(rounded)
            # Always move, not only below a threshold lower than the target ESS.
            X,E=self._move(X,E,beta)
            if beta>=p['beta_max']:
                return 'annealing_complete'
        return 'iteration_limit'

MAP={'BH-OJ':BasinHoppingOJ,'GA':GeneticAlgorithm,'PSO':ParticleSwarmOptimization,
     'CLPSO':ComprehensiveLearningPSO,'SMC':SMCOptimizer}

def create_optimizer(method,n_atoms,**kwargs):
    m=method.strip().upper()
    if m not in MAP:raise ValueError(f'Unknown method {method}')
    return MAP[m](n_atoms,**kwargs)

def check_availability():
    print('Baselines v'+VERSION+': '+', '.join(METHODS))

if __name__=='__main__':check_availability()
