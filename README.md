# HPFO — PF C++ 1.3.1

Reproducibility materials for:

**A Hybrid Particle Filtering and Local Optimization Framework for Solving
Challenging Lennard-Jones Cluster Problems**

Journal of Global Optimization — Manuscript **JOGO-D-25-00267**

**Author:** François Bertrand Akoa  
**Position and institution:** R&D Director, MAATAI-LAB  
**Contact:** francois.akoa@maatai-lab.com

## Start here

[Open the experimental files and source archives](https://github.com/francoisakoa/hpfo/releases/tag/pf131-jogo-r2)

The tables below identify the downloads for each experiment.
Large numerical datasets are distributed under **Assets** on the release
page, not inside GitHub's automatic **Source code** downloads.

## Main benchmark: 240 runs with independent initial populations

The benchmark covers LJ38, LJ75, LJ76, LJ77, LJ98, LJ102, LJ103 and LJ104,
with 30 separately generated initial populations per size.

Download the protocols/source archive and the data archives for the
cluster sizes to be examined.

| Material | Download |
|---|---|
| Dataset guide | [Independent-initialization README](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/hpfo-pf131-independent-README-20261018.md) |
| Source code, campaign scripts, protocols and analysis materials | [Protocols and sources](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/hpfo-pf131-independent-protocols-20261018.tar.gz) |
| LJ38 — 30 runs | [LJ38 data](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/hpfo-pf131-independent-LJ38-20261018.tar.gz) |
| LJ75 — 30 runs | [LJ75 data](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/hpfo-pf131-independent-LJ75-20261018.tar.gz) |
| LJ76 — 30 runs | [LJ76 data](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/hpfo-pf131-independent-LJ76-20261018.tar.gz) |
| LJ77 — 30 runs | [LJ77 data](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/hpfo-pf131-independent-LJ77-20261018.tar.gz) |
| LJ98 — 30 runs | [LJ98 data](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/hpfo-pf131-independent-LJ98-20261018.tar.gz) |
| LJ102 — 30 runs | [LJ102 data](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/hpfo-pf131-independent-LJ102-20261018.tar.gz) |
| LJ103 — 30 runs | [LJ103 data](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/hpfo-pf131-independent-LJ103-20261018.tar.gz) |
| LJ104 — 30 runs | [LJ104 data](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/hpfo-pf131-independent-LJ104-20261018.tar.gz) |
| Download checksums | [SHA256 manifest](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/hpfo-pf131-independent-SHA256SUMS-20261018.txt) |

The run materials include complete `pf.json` records, native
`initial.pfpop` populations, initialization metadata, final `pf.xyz`
coordinates and recorded iteration histories.

## Baselines, component studies and initialization comparison

These are separate experimental datasets. Do not pool their runs with
the main independent-initialization benchmark.

| Experiment | Download |
|---|---|
| Five baseline methods: BH-OJ, GA, PSO, CLPSO and SMC — 1,200 runs | [Baseline archive](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/BASELINES_COMPACT_001.zip) |
| Ablation and sensitivity on LJ75/LJ102 — 820 distinct runs | [Component-study archive](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/hpfo-pf131-study-data.tar.gz) |
| Fixed-population HPF comparison — 240 runs, with associated Ipopt multistart records | [Fixed-population archive](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/hpfo-pf131-main-data.tar.gz) |
| Checksums for these three archives | [SHA256 manifest](https://github.com/francoisakoa/hpfo/releases/download/pf131-jogo-r2/SHA256SUMS.txt) |

The component-study `full` configuration is the reference for its
ablation comparisons. It must not be replaced by results from the
main benchmark.

## Browse the implementation

- [PF C++ 1.3.1 source code](https://github.com/francoisakoa/hpfo/tree/pf131-jogo-r2/reproducibility/pf131-jogo-r2/code/main-pf131)
- [Component-study source code and driver](https://github.com/francoisakoa/hpfo/tree/pf131-jogo-r2/reproducibility/pf131-jogo-r2/code/study-r2)
- [Baseline source code](https://github.com/francoisakoa/hpfo/tree/pf131-jogo-r2/reproducibility/pf131-jogo-r2/code/baselines-export)
- [Build and replay instructions](https://github.com/francoisakoa/hpfo/blob/pf131-jogo-r2/docs/BUILD_AND_REPLAY.md)

For the independent-initialization campaign, use the campaign scripts,
configuration records and analysis materials in the **Protocols and
sources** archive linked above.

The C++ executable is named `pf_solver`.

## Verify downloaded files

After downloading all files listed in the corresponding manifest,
run the following commands from the download directory:

```bash
sha256sum -c hpfo-pf131-independent-SHA256SUMS-20261018.txt
sha256sum -c SHA256SUMS.txt
```

A missing-file message means that a file listed in that manifest has
not been downloaded. An integrity mismatch must be resolved before
using the affected archive.

Read the dataset guide and the instructions included with each
source/protocol archive before extracting or running analysis scripts.

## Interpretation of results

Initialization seeds and search seeds are recorded separately.
PF objective-evaluation counts include population and local-refinement
evaluations. Target attainment, solver stopping status, elapsed time
and evaluation counts are distinct measurements.

Rebuilding numerical summaries from saved records does not require
rerunning the optimization campaigns. A target-attainment result is
not a certificate of global optimality or stationarity.

## Contact

François Bertrand Akoa  
R&D Director, MAATAI-LAB  
francois.akoa@maatai-lab.com
