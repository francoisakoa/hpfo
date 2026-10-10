"""Validate native LJ derivatives against the supplied JAX and Pyomo models.

Requires numpy, jax, jaxlib, pyomo. The Pyomo SolverFactory is replaced solely
to capture its constructed model: this does NOT claim a Python Ipopt solve.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import jax
import pyomo.environ as pyo
from pyomo.core.expr.calculus.derivatives import differentiate, Modes

parser = argparse.ArgumentParser()
parser.add_argument('probe', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
reference = Path(__file__).resolve().parents[1] / 'reference'
sys.path.insert(0, str(reference))
import solve_nlp_local_gpu as local
# Load the active function verbatim without importing unrelated optional Numba helpers.
ipopt_path=reference/'solve_nlp_ipopt.py'
ipopt_node=next(node for node in ast.parse(ipopt_path.read_text()).body
                if isinstance(node,ast.FunctionDef) and node.name=='solver_lj_cyipopt')
ipopt_ns=dict(vars(pyo),np=np,pyo=pyo,Sequence=Sequence)
exec(compile(ast.Module(body=[ipopt_node],type_ignores=[]),str(ipopt_path),'exec'),ipopt_ns)
jax.config.update('jax_enable_x64', True)

fixtures = [
    ('pair', np.array([[.2,-.1,.3], [1.3,.4,.7]])),
    ('triangle', np.array([[0.,0.,0.], [1.1,0.,0.], [.4,.9,.1]])),
    ('five_atoms', np.array([[0.,0.,0.],[1.12,.1,.02],[.3,1.14,.2],[-.15,.15,1.2],[1.2,1.1,1.]])),
    ('near_collision', np.array([[0.,0.,0.], [.002,.001,-.001]])),
    ('collision', np.zeros((2,3))),
]
records=[]
def compare(actual, expected):
    actual=np.asarray(actual);expected=np.asarray(expected)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=2e-10)
    return float(np.max(np.abs(actual-expected)/(1+np.abs(expected))))

class Capture:
    def __init__(self): self.options={};self.model=None
    def solve(self, model, **kwargs):
        self.model=model
        return SimpleNamespace(solver=SimpleNamespace(termination_condition='optimal'))

for name, positions in fixtures:
    n,d=positions.shape;x=positions.ravel()
    native=subprocess.check_output([str(args.probe.resolve()),'--lj'],
        input=f'{n} {d}\n'+' '.join(format(v,'.17g') for v in x)+'\n',text=True)
    rows=[np.fromstring(line,sep=' ') for line in native.splitlines()]
    f=rows[0][0];g=rows[1];h=np.array(rows[2:])
    objective=local.make_lj_objective(n,d)
    capture=Capture();ipopt_ns['SolverFactory']=lambda name:capture
    ipopt_ns['solver_lj_cyipopt'](n,d,np.full(x.size,-5.),np.full(x.size,5.),x,radius=.35)
    model=capture.model
    variables=list(model.x.values())
    symbolic_gradient=differentiate(model.obj.expr,wrt_list=variables,mode=Modes.reverse_symbolic)
    pg=np.array([pyo.value(expr) for expr in symbolic_gradient])
    ph=np.array([differentiate(expr,wrt_list=variables,mode=Modes.reverse_numeric) for expr in symbolic_gradient])
    record={'fixture':name,'variables':x.size,
        'numpy_energy_error':compare(f,local.lj_energy_numpy(x,n,d)),
        'jax_energy_error':compare(f,objective(x)),
        'jax_gradient_error':compare(g,jax.grad(objective)(x)),
        'jax_hessian_error':compare(h,jax.hessian(objective)(x)),
        'pyomo_energy_error':compare(f,pyo.value(model.obj)),
        'pyomo_gradient_error':compare(g,pg),'pyomo_hessian_error':compare(h,ph)}
    assert len(model.ball)==n
    for atom,constraint in enumerate(model.ball.values()):
        compare(pyo.value(constraint.body),np.dot(positions[atom],positions[atom]))
        compare(pyo.value(constraint.upper),.35**2)
        expected=np.zeros(x.size);expected[atom*d:(atom+1)*d]=2*positions[atom]
        compare(differentiate(constraint.body,wrt_list=variables,mode=Modes.reverse_numeric),expected)
    records.append(record)
result={'jax_version':jax.__version__,'jax_x64':True,'devices':[str(d) for d in jax.devices()],
        'numpy_version':np.__version__,'fixtures':records,
        'scope':'Energy/gradient/Hessian and source constraint structure; no full Python PF trajectory replay.'}
if args.output:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
