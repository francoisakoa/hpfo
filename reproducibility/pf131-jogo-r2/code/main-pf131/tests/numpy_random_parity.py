"""Independent NumPy oracle, source AST methods and end-to-end CLI initialization.

No JAX, sklearn or Python runtime is required by the C++ executable. NumPy is
required by this test only. Run with --report PATH to retain validation evidence.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('probe', type=Path)
parser.add_argument('solver', type=Path)
parser.add_argument('--report', type=Path)
args = parser.parse_args()
probe, solver = str(args.probe.resolve()), str(args.solver.resolve())
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
source = Path(__file__).resolve().parents[1] / 'reference/general_nlp_pf_optimized_gpu.py'
tree = ast.parse(source.read_text())
make_seed_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'make_seed')
source_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PF_NLP_Solver')
names = {'_random_box_point', '_resample_particles', 'invertible_particle_flow'}
methods = [n for n in source_class.body if isinstance(n, ast.FunctionDef) and n.name in names]
mapping = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'linear_mapping', 'invertible_mapping'}]
namespace = {'np': np, 'hashlib': hashlib}
exec(compile(ast.Module(body=[make_seed_node, *methods, *mapping], type_ignores=[]), str(source), 'exec'), namespace)
make_seed = namespace['make_seed']
report = {'numpy_version': np.__version__, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
checks = 0
compared_values = 0


def query(command):
    return subprocess.check_output([probe], input=command+'\n', text=True, env=env, timeout=30).splitlines()


def exact(actual, expected, name):
    global checks, compared_values
    expected = np.asarray(expected)
    actual = np.asarray(actual, dtype=expected.dtype)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    # Integers compare as integers, never through float64 (raw bits can exceed 2**53).
    assert np.array_equal(actual, expected), (name, np.flatnonzero(actual != expected)[:5])
    if expected.dtype == np.float64:
        assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64)), name
    checks += 1
    compared_values += expected.size


seed_cases = [(2024, n, m, offset) for n in (38, 76) for m in range(1, 31) for offset in (0, 1000)]
seed_cases += [(base, n, m, offset) for base, n, m, offset in
               [(0, 2, 1, 0), (2**64-1, 76, 30, 1000), (2**64-1, -2**31, -2**31, -2**31)]]
commands = '\n'.join(f'seed {b} {n} {m} {o}' for b, n, m, o in seed_cases)
actual = subprocess.check_output([probe], input=commands+'\n', text=True, env=env).splitlines()
exact([int(x) for x in actual], [make_seed(*case) for case in seed_cases], 'MD5 source make_seed')
report['md5_seed_cases'] = len(seed_cases)

seeds = [0, 1, 42, 2024, 2**32-1, 2**32, 2**64-1, make_seed(2024, 76, 30, 1000)]
normal_samples = 0
for seed in seeds:
    for mode, count in [('raw', 2048), ('uniform', 2048), ('normal', 100000)]:
        rng = np.random.default_rng(seed)
        row = query(f'{mode} {seed} {count}')[0]
        if mode == 'raw':
            actual = np.array([int(x) for x in row.split()], dtype=np.uint64)
            expected = rng.bit_generator.random_raw(count)
        else:
            actual = np.fromstring(row, sep=' ')
            expected = rng.random(count) if mode == 'uniform' else rng.standard_normal(count)
        exact(actual, expected, f'{mode} seed={seed}')
        if mode == 'normal': normal_samples += count
    for size in [1, 2, 3, 170, 10000, 2**31+1, 2**32, 2**32+1, 2**63-1]:
        rng = np.random.default_rng(seed)
        row, next_raw = query(f'integers {seed} {size} 101')
        exact([int(x) for x in row.split()], rng.integers(0, size, 101), 'bounded integers')
        exact(int(next_raw), rng.bit_generator.random_raw(), 'integer continuation')
    for size, count, replace in [(1, 1, 0), (12, 0, 0), (12, 12, 0), (170, 169, 0),
                                 (284, 2, 0), (170, 210, 1), (10001, 300, 0), (10001, 20, 0)]:
        rng = np.random.default_rng(seed)
        rows = query(f'choice {seed} {size} {count} {replace}')
        exact([int(x) for x in rows[0].split()], rng.choice(size, count, replace=bool(replace)), 'choice')
        exact(int(rows[1]), rng.bit_generator.random_raw(), 'choice continuation')
    rng = np.random.default_rng(seed)
    expected = [[rng.integers(10000), rng.standard_normal(), rng.random()] for _ in range(2000)]
    actual = np.fromstring(query(f'mixed {seed} 2000')[0], sep=' ').reshape(-1, 3)
    exact(actual, expected, 'interleaved uint32/normal/uniform buffer')
    for count in (0, 1, 19380, 64752):
        rng = np.random.default_rng(seed)
        first = rng.integers(10000)
        rng.bit_generator.random_raw(count)
        second = rng.integers(10000)
        third = rng.bit_generator.random_raw()
        exact([int(x) for x in query(f'advance {seed} {count}')[0].split()],
              np.array([first, second, third], dtype=np.uint64), 'skip-ahead preserves buffered half')
report['normal_samples_bitwise_equal'] = normal_samples

for seed in (0, 42, 2024, 2**32-1):
    rng = np.random.RandomState(seed)
    expected = [[rng.random_sample(), rng.randint(170)] for _ in range(2000)]
    exact(np.fromstring(query(f'legacy {seed} 2000')[0], sep=' ').reshape(-1, 2), expected, 'sklearn MT19937 stream')

# Exercise the actual C++ Solver through 3 iterations. All DBSCAN points form
# one group, eliminating external clustering differences from this draw-order test.
namespace['perform_clustering'] = lambda points, *a: np.zeros(len(points), dtype=int)
for seed in (2024, make_seed(2024, 38, 1, 1000)):
    rng = np.random.default_rng(seed)
    oracle = SimpleNamespace(lb=np.full(6, -5.), ub=np.full(6, 5.), mapping_method='linear',
                             clustering_algo='dbscan', backend_cfg={},
                             objective_batch=lambda points: np.sum(np.asarray(points)**2, axis=1))
    particles = [namespace['_random_box_point'](oracle, rng) for _ in range(12)]
    rows = query(f'flow {seed}')
    for t, row in enumerate(rows):
        count = max(2, int(12*(1-t/3)))
        particles = namespace['_resample_particles'](oracle, particles, count, rng)
        particles = namespace['invertible_particle_flow'](oracle, particles, rng, t, 3)
        values = np.fromstring(row, sep=' ')
        assert int(values[0]) == count
        exact(values[1:].reshape(count, 6), np.array(particles), f'source PF iteration {t+1}')
report['controlled_source_pf_iterations_bitwise_equal'] = 6

with tempfile.TemporaryDirectory(prefix='pf_numpy_initialization_') as temp:
    output = Path(temp) / 'starts'
    subprocess.run([solver, '--atoms', '38', '76', '--runs', '30', '--seed', '2024',
                    '--local-refine', 'none', '--generate-only', '--quiet', '--output-dir', str(output)],
                   check=True, capture_output=True, env=env, timeout=90)
    coordinates = 0
    for n in (38, 76):
        for m in range(1, 31):
            folder = output / f'LJ{n}_run{m}'
            metadata = json.loads((folder / 'initialization.json').read_text())
            si, sp = make_seed(2024, n, m, 0), make_seed(2024, n, m, 1000)
            assert metadata['seed_init'] == si and metadata['seed_pf'] == sp
            assert metadata['base_seed'] == 2024 and not metadata['x0_used_by_pf']
            assert metadata['population_source'] == 'python-pf-box'
            lb, ub = np.full(3*n, -5.), np.full(3*n, 5.)
            margin = .1*(ub-lb)
            x0 = np.random.default_rng(si).uniform(lb+margin, ub-margin, size=3*n)
            exact(metadata['x0'], x0, f'x0 LJ{n} run {m}')
            rng = np.random.default_rng(sp)
            oracle = SimpleNamespace(lb=lb, ub=ub)
            expected = np.array([namespace['_random_box_point'](oracle, rng) for _ in range(max(3*n+56, 50))])
            actual = np.loadtxt(folder / 'initial.pfpop', skiprows=1)
            exact(actual, expected, f'PF population LJ{n} run {m}')
            assert metadata['pf_raw_draws_before_loop'] == expected.size
            coordinates += expected.size + x0.size
    # Frozen starts must reproduce the actual solved trajectory, including RNG continuation.
    common = ['--atoms', '4', '--particles', '12', '--min-particles', '2', '--max-iter', '3',
              '--local-refine', 'none', '--no-final-refine', '--no-target', '--quiet']
    serial, replay = Path(temp)/'serial', Path(temp)/'replay'
    for dest, extra in [(serial, []), (replay, ['--population', str(serial/'LJ4_run1/initial.pfpop')])]:
        subprocess.run([solver, *common, *extra, '--output-dir', str(dest)], check=True,
                       capture_output=True, env=env, timeout=30)
    a, b = [json.loads((d/'LJ4_run1/pf.json').read_text()) for d in (serial, replay)]
    for field in ('x', 'history', 'objective_evaluations'):
        assert a[field] == b[field], field
    assert a['options']['seed_init'] == make_seed(2024, 4, 1, 0)
    assert a['options']['seed_pf'] == a['options']['seed'] == make_seed(2024, 4, 1, 1000)
    assert json.loads((replay/'LJ4_run1/initialization.json').read_text())['population_source'] == 'external-pfpop'
report.update(status='PASS', checks=checks, compared_values=compared_values,
              initialization_runs=60, initial_coordinates_bitwise_equal=coordinates,
              frozen_replay=True, full_lj76_trajectory_validated=False)
if args.report:
    args.report.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
