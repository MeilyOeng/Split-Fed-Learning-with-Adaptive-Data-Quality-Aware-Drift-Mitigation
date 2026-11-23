"""Orchestration layer for the drift+fault+DQ+M4 final experiment matrix.

This package generates configs, prepares data, runs methods, evaluates,
builds condition-level mitigation reports, and aggregates summaries for the
frozen 90-scenario / 180-suite / 1260-method-run experiment matrix. It only
orchestrates the existing, already-validated
:mod:`drift.mitigation.prepared` pipeline and :mod:`drift.mitigation_report`
-- it never reimplements training, evaluation, or reporting semantics.
"""
