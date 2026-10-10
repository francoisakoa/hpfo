# Evaluation counting and timing

For PF results:
  objective_evaluations = objective_pf + objective_local

Counts include objective evaluations used by local optimization. Derivative
counters and the final reporting audit are separate fields.
The final reporting energy evaluation is not part of objective_evaluations.
Check the archived source for the precise counted callback paths.

seconds is elapsed solver time, not summed worker CPU time.
Do not multiply or divide counts or seconds by the number of workers.
Preparation and result-file export outside solve() have a different timing scope.
Concurrent system load and changed dependencies can affect elapsed times.

Target attainment uses the recorded target and target_tolerance.
It is not a certificate of global optimality or local stationarity.
The implemented regularized LJ energy must not be silently replaced by a
different unregularized or differently normalized potential.

Means, medians and sample standard deviations in the generated tables include
all completed runs unless a column explicitly says otherwise.
Unsuccessful runs are not omitted. Shared sensitivity controls are not counted
as additional independent experiments.

Baseline budgets come from the archived calibration protocol: comparable-order
budgets are not identical budgets. Preserve original limits, actual counts,
stopping statuses and all relevant local-search work.
