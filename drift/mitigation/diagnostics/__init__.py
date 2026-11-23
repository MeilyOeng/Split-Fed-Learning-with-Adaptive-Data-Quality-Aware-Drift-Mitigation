"""Standalone diagnostic/ablation toolkit for the prepared drift-mitigation
suite (R0/R1/M0/M1/M2/M3).

This package is completely additive: it never modifies production training
code (client/client.py, run_split.py, server/server.py,
server/frozen_gradient.py, drift/mitigation/prepared/ewc.py, or any other
production training module) and never writes into an existing suite root,
checkpoint H root, or preparation root. Every module either (a) reads
already-completed artifacts read-only, or (b) derives a config from an
existing one and invokes the existing canonical CLI
(``python -m drift.mitigation.prepared.cli``) as a subprocess, writing only
under a separate ``diagnostics/`` output root.

A few pure, side-effect-free production building blocks are imported
read-only (never modified, never coupled to for training behavior):
``models.split_lstm`` (the architecture definition itself), and
``server.frozen_gradient.frozen_server_activation_step`` (already a pure
inference function, verified side-effect-free by production's own
before/after state-hash assertion). The EWC penalty formula and the
MAE/RMSE/sMAPE metrics are deliberately re-implemented locally in
``common.py`` rather than imported/extracted from client.py or
evaluation.py, per an explicit "zero production file changes" requirement.
"""