# PF C++ 1.3.1 — JOGO reproducibility package

Repository: https://github.com/francoisakoa/hpfo
Manuscript: A Hybrid Particle Filtering and Local Optimization Framework
for Solving Challenging Lennard-Jones Cluster Problems.
Manuscript identifier: JOGO-D-25-00267.

Start with REVIEWER_GUIDE.md. This package separates:
1. The main PF131 benchmark: 240 PF runs and 240 associated Ipopt multistart runs.
2. The instrumented LJ75/LJ102 component study: 820 distinct runs.
3. External baseline exports, whose imported status is recorded separately.

The component study is not pooled with the main benchmark.
Table 7 uses 30 component-study runs per configuration and instance.
Tables 8–10 use 10 runs; their default beta and alpha rows reuse full runs 1–10.

Source snapshots are preserved separately in code/main-pf131 and code/study-r2.
Newer HPFO versions are not substitutes for either snapshot.
Raw data are distributed as release assets from this same repository.

No optimization experiment is executed by the preparation or table-rebuild tools.
No claim of fresh compilation, successful replay, or public availability follows
from preparing this directory.
