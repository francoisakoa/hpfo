import importlib.util,json,math,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('study',ROOT/'study.py'); s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
P=dict(ablation_runs=30,sensitivity_runs=10,base_seed=2024,max_iter=1000,max_time=36000.0,max_evals=0,threads=10,workers=8,fixed_beta=1e-4)
class PlanTests(unittest.TestCase):
 def test_dimensions(self):
  p=s.make_plan(P);self.assertEqual((len(p['tasks']),len(p['table_rows'])),(820,860))
  self.assertEqual({t['atoms'] for t in p['tasks']},{75,102});self.assertEqual(len({t['id'] for t in p['tasks']}),820)
 def test_article_grids(self):
  c=s.configurations();self.assertEqual([v['changes']['particles'] for v in c if v['table']==8],[100,200,300,400,500])
  self.assertEqual([v['changes']['beta_initial'] for v in c if v['table']==9],[1e-5,1e-4,1e-3,1e-2])
  self.assertEqual([v['changes']['ipf_alpha_initial'] for v in c if v['table']==10],[.95,.98,.99,1.])
 def test_aliases(self):
  p=s.make_plan(P);ids={t['id'] for t in p['tasks']}
  for r in p['table_rows']:self.assertIn(r['task'],ids)
  rows=[r for r in p['table_rows'] if r['configuration'] in ('alpha_0.98','beta_1e-4')]
  self.assertEqual(len(rows),40);self.assertTrue(all(r['task'].startswith('full/') and r['run']<=10 for r in rows))
 def test_single_ablation_changes(self):
  p=s.make_plan(P);base=next(t['options'] for t in p['tasks'] if t['configuration']=='full' and t['atoms']==75)
  for name,key in [('no_ipf','ipf_enabled'),('no_sharing','lambda_share'),('no_novelty','lambda_novelty'),('no_kicks','kicks'),('no_island_exchange','island_exchange'),('no_adaptive_annealing','adaptive_annealing')]:
   d=next(t['options'] for t in p['tasks'] if t['configuration']==name and t['atoms']==75)
   self.assertEqual({k for k in base if d[k]!=base[k]},{key})
 def test_local_absent(self):
  p=s.make_plan(P)
  for t in p['tasks']:
   if t['configuration'] in ('no_local','pf_only'):
    self.assertEqual(t['options']['local_method'],'none');self.assertFalse(t['options']['final_refinement']);self.assertEqual(t['options']['refine_top_k'],0)
 def test_reduced(self):
  p=s.make_plan(P);o=next(t['options'] for t in p['tasks'] if t['configuration']=='pf_only')
  self.assertTrue(o['adaptive_annealing'])
  for k in ('ipf_enabled','lambda_share','lambda_novelty','kicks','islands','diversity_injection','ess_rejuvenation'):self.assertFalse(o[k])
 def test_same_paired_seeds(self):
  p=s.make_plan(P);groups={}
  for t in p['tasks']:groups.setdefault((t['atoms'],t['run']),set()).add((t['seed_pf'],t['seed_init']))
  self.assertTrue(all(len(x)==1 for x in groups.values()));self.assertEqual(len({next(iter(x))[0] for x in groups.values()}),60)
 def test_no_false_population_count(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'p';path.write_text('PFPOP1 2 6\n'+'0 '*12+'\n')
   with self.assertRaises(RuntimeError):s.population_shape(path,2,2)
 def test_plan_deterministic(self):self.assertEqual(s.make_plan(P),s.make_plan(P))
 def test_cli_true_ablations(self):
  p=s.make_plan(P)
  for t in p['tasks']:
   args=s.solver_args(t['options'])
   self.assertNotIn('--include-ipopt',args);self.assertEqual(args.count('--particles'),1)
   if t['configuration']=='no_ipf':self.assertIn('--no-ipf',args);self.assertNotIn('mutation',args)
 def test_partial_archive(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);p=s.make_plan(dict(P,ablation_runs=1,sensitivity_runs=1));s.write_json(out/'PLAN.json',p)
   self.assertFalse(s.summary(out,p)['complete'])
   with self.assertRaises(RuntimeError):s.make_bundle(out,p)
   path=s.make_bundle(out,p,True);self.assertIn('PARTIEL',path.name);self.assertTrue(path.with_name(path.name+'.sha256').exists())
 def test_done_tamper(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);t=s.make_plan(P)['tasks'][0];root=out/'runs'/t['id'];folder=root/'raw';folder.mkdir(parents=True)
   file=folder/'pf.json';file.write_text('{}');s.write_json(root/'DONE.json',dict(task_sha256=s.digest(t),folder=str(folder.relative_to(out)),hashes={'pf.json':s.sha(file)},summary={}))
   self.assertIsNotNone(s.done_result(out,t));file.write_text('{"fake":1}')
   with self.assertRaises(RuntimeError):s.done_result(out,t)
class RefineAllTests(unittest.TestCase):
 def test_campaign_identity(self):
  p=s.make_plan(P)
  self.assertEqual(p['schema'],'PF131_LJ75_LJ102_REFINE_ALL_R2')
  self.assertEqual(p['study_design']['atoms'],[75,102])
  self.assertEqual(p['solver_interface'],'PF131_ARTICLE_STUDY_R1')
 def test_all_active_local_configurations(self):
  for task in s.make_plan(P)['tasks']:
   o=task['options'];args=s.solver_args(o)
   if task['configuration'] not in ('no_local','pf_only'):
    self.assertTrue(o['refine_all']);self.assertEqual(args.count('--refine-all'),1)
    self.assertEqual(o['refine_top_k'],5) # terminal native refinement retained
   else:
    self.assertFalse(o['refine_all']);self.assertNotIn('--refine-all',args)
    self.assertIn('--no-final-refine',args);self.assertEqual(o['refine_top_k'],0)
 def test_lj102_target_and_population(self):
  o=s.default_options(102,P)
  self.assertEqual(o['particles'],362);self.assertEqual(o['target'],-569.363653)
 def test_runs_per_size(self):
  tasks=s.make_plan(P)['tasks']
  for n in (75,102):self.assertEqual(sum(t['atoms']==n for t in tasks),410)
  self.assertFalse(any(t['atoms'] in (77,98) for t in tasks))
 def test_no_local_loop_or_terminal_calls_in_spec(self):
  for t in s.make_plan(P)['tasks']:
   if t['configuration'] in ('no_local','pf_only'):
    o=t['options'];self.assertEqual(o['local_method'],'none')
    self.assertFalse(o['final_refinement']);self.assertFalse(o['refine_all'])
 def test_input_pairing_preserved(self):
  tasks=s.make_plan(P)['tasks']
  for n in (75,102):
   groups=[t for t in tasks if t['atoms']==n and t['run']==1 and not t['configuration'].startswith('np_')]
   self.assertEqual(len({(t['options']['particles'],t['seed_pf'],t['seed_init']) for t in groups}),1)
if __name__=='__main__':unittest.main()

