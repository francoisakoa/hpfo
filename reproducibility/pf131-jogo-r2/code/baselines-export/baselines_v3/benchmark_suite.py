#!/usr/bin/env python3
"""Compatibility entry point. A single campaign/runner/counting implementation."""
from evaluation import EvaluationCounter, BudgetExceeded
from protocol import method_params as _get_method_params, make_seed as _make_seed
if __name__=='__main__':
    from campaign import main
    raise SystemExit(main())
