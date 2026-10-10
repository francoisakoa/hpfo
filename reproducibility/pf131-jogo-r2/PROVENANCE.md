# Provenance and boundaries

The main source comes from the archived local-source bundle, not from a later
working-tree copy. Its parent archive SHA256 is recorded in PREPARATION.json.

The main benchmark was collected retrospectively. The historical link between
each executable and each run is not established solely by this collection.
The historical source_sha256 field must not be described as a C++ tree hash:
the archived writer records a Python-reference-source hash in that field.

For the main benchmark, each size uses a frozen initial population across its
30 PF runs, with distinct stochastic search seeds. Distinct seed_init metadata
must not be interpreted as evidence of 30 distinct loaded populations.

The component study has its own source, BINARY.json, PLAN.json, task definitions,
DONE.json and EXECUTION.json records. The preparation checks these records
without executing archived binaries.

The two code trees and their IPF scheduling options are not silently merged.
Absolute historical paths are retained in provenance. This is not an anonymous
package, and private repository visibility does not anonymize its contents.

No figure is described as regenerated merely because an older image is copied.
Publication plotting scripts and their actual input traces must be identified
before claiming figure-level reproducibility.
