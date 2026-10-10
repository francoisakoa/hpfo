# Artifact scope and claims

## Included

- Exact files copied from the already verified PF131 preparation, with its old
  preparation and checksum list retained separately as upstream provenance.
- 240 main PF records and 240 associated IPOPT records in their original data asset.
- 820 completed component-study records, initial populations, coordinates, traces,
  commands and completion metadata in the study asset.
- The newly exported baseline ZIP(s), unchanged, and their original checksum list.
- Identified baseline source/configuration files exposed for reading in Git.
- Hash/grid verification, separate builds/tests, raw-summary reconstruction, optional
  recorded-command replays and explicit publication tools.

## Not asserted by assembly

- A new optimization result, bitwise replay, historical main binary-to-run certificate,
  global-optimality certificate, or stationarity certificate.
- Complete event-level baseline histories: the compact transport deliberately omits
  large event logs. The full backups and originals remain on the TRX.
- Exact regeneration of every submitted manuscript figure. The supplied plotting
  helper is diagnostic. The original figure-generation pipeline and historical input
  selections, where not included, must not be represented as reproduced.
- A new license for author or third-party code. Notices and provenance remain intact.
- Access by an Editor/reviewer account before the tag/Release is actually published
  and, for a private repository, the appropriate account is granted access.

## Interpretation

Use the recorded regularized LJ potential and targets. Do not silently change energy
normalization, domain, smoothing, seeds, loaded populations, local solvers or budgets.
The main source and component-study source are not identical, including IPF scheduling
controls. The latter uses its own `full` control. Sensitivity controls are runs 1–10,
not all 30 ablation repetitions.

Comparable-order baseline budgets are not identical budgets. Conclusions concern
these implementations and settings. Component removal is not a standalone algorithm
comparison; grouped removals do not identify one component's causal contribution.
The idealized uncapped theoretical setting does not provide a practical convergence
guarantee for the capped, adaptive, archive-dependent implementation.
