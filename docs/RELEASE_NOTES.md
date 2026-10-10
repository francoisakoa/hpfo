# PF C++ 1.3.1 — JOGO reviewer artifact

Reproducibility material for JOGO-D-25-00267. The main PF131 and instrumented
component-study source snapshots remain separate.

Data assets contain 240 main PF results, 240 companion IPOPT multistart results,
820 completed LJ75/LJ102 component-study results and 1,200 indexed external
baseline records. Keep `SHA256SUMS.txt` with these assets.

The component study comprises 600 ablation runs and 220 additional sensitivity
runs. Default sensitivity references reuse full runs 1–10; they are not extra
independent experiments. Table 7 uses all 30 component-study runs per configuration.

The repository includes verification, build/test, summary-reconstruction, diagnostic
plot and explicit replay tools. A replay is not started by verification or table
reconstruction. Archived scientific outputs are never overwritten by these tools.

See `docs/ARTIFACT_SCOPE.md` for provenance and scope limits, particularly historical
binary/run linkage, full baseline event logs, exact manuscript-figure regeneration,
the practical/theoretical distinction and the interpretation of comparable budgets.
