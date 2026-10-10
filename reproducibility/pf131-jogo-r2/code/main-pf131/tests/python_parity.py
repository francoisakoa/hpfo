"""Compare native operators against equations and CPU backend from the supplied release."""
import ast
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.integrate import solve_ivp

reference = Path(__file__).resolve().parents[1] / 'reference'
sys.path.insert(0, str(reference))
import backend_ops
source = reference / 'general_nlp_pf_optimized_gpu.py'
tree = ast.parse(source.read_text())
names = {'compute_sharing_factors', 'linear_mapping', 'residual_mapping', 'affine_coupling_mapping', 'compute_novelty'}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
solver = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PF_NLP_Solver')
nodes += [n for n in solver.body if isinstance(n, ast.FunctionDef) and n.name in {'_compute_weights','adjust_beta','_resample_particles_by_weights'}]
ns = {'np':np, 'Dict':dict, 'compute_sharing_factors_backend':backend_ops.compute_sharing_factors_backend}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
class State:
    sigma_share=3.1
    lambda_share=lambda_novelty=0.01
    backend_cfg={"use_gpu":False}
    num_particles=50
    diversity_threshold=1
    beta_min_init=1e-4
    beta_max_init=5
    beta_factor=1.3
x=np.array([.2,-.1,.3,1.3,.4,.7]);b=np.array([.7,.1,-.2,.9,.6,.3])
s=State();s.archive=[x+b]
rows=[np.fromstring(line,sep=' ') for line in subprocess.check_output([str(Path(sys.argv[1]).resolve())],text=True).splitlines()]
local_tree=ast.parse((reference/'solve_nlp_local_gpu.py').read_text())
local_nodes=[n for n in local_tree.body if isinstance(n, ast.FunctionDef) and n.name=='lj_energy_numpy']
exec(compile(ast.Module(body=local_nodes,type_ignores=[]),str(reference/'solve_nlp_local_gpu.py'),'exec'),ns)
expected=[np.array([ns['lj_energy_numpy'](x,2,3)])]
for method in ['linear_mapping','residual_mapping','affine_coupling_mapping']:
    expected.append(ns[method](x,b,.73))
expected.append(solve_ivp(lambda t,y:.73*np.tanh(b-y),(0,1),x,rtol=1e-12,atol=1e-13).y[:,-1])
w,ess=ns['_compute_weights'](s,np.array([-4.,-2.,1.,7.]),1.8,[x,b,2*x,2*b])
expected.extend([w,np.array([ess])])
# Any weights with ESS=17 produce this source beta schedule.
beta_weights=np.full(17,1/17)
expected.append(np.array([ns['adjust_beta'](s,t,5,beta_weights,.9) for t in range(5)]))
class Draws:
    def random(self,n): return np.array([.2,.5,.1,.9])
    def normal(self,scale,size): return np.zeros(size)
s.lb=np.full(1,-100.);s.ub=np.full(1,100.)
expected.append(np.array(ns['_resample_particles_by_weights'](s,[np.array([i],dtype=float) for i in range(4)],np.array([.1,.2,.3,.4]),Draws())).ravel())
expected.extend([backend_ops.compute_sharing_factors_backend(np.array([x,b,2*x,2*b]),3.1,cfg=s.backend_cfg), np.array([backend_ops.compute_diversity_backend(np.array([x,b,2*x,2*b]),cfg=s.backend_cfg)])])
assert len(rows)==len(expected)
for i,(a,b) in enumerate(zip(rows,expected)):
    np.testing.assert_allclose(a,b,rtol=2e-11,atol=2e-12,err_msg=f'operator row {i}')
print(f'PASS {len(rows)} source-equation comparisons (supplied CPU backend; CNF compared to high-accuracy ODE)')
