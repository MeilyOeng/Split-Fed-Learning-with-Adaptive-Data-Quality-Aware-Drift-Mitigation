# Adaptive Drift Mitigation for Split Federated Learning

A research implementation of privacy-aware time-series forecasting with **SplitFed learning**, online drift detection, data-quality scoring, and targeted post-drift adaptation. The system splits an LSTM between clients and cluster servers, keeping raw client timelines local while coordinating federated updates.

## What this project demonstrates

- Split learning across 84 clients grouped into four federated clusters.
- Sudden, gradual, seasonal, measurement, and process-drift simulation.
- Page-Hinkley and ADWIN detector support with event-level evaluation.
- Data-quality-aware training, controlled fault injection, and quality-weighted aggregation.
- Reproducible adaptation comparisons: frozen baselines, continued SplitFed training, oracle client-only adaptation, and EWC.
- Offline-prepared experiments with immutable manifests, checkpoint validation, and regression tests.

## Architecture

```text
client timelines -> stream service -> client LSTM -> smashed activations
                                                      |
                                                      v
                                     cluster split server -> global FedAvg
                                                      |
                         drift/fault detection -> targeted mitigation
```

## Repository layout

```text
client/          client-side model and worker API
server/          split-server training and inference
fed_server/      cross-cluster federated aggregation
models/          split LSTM definition
data/            streaming and prepared-window services
data quality/    fault injection and data-quality pipeline
drift/           drift generation, detection, evaluation, and mitigation
plotting/        reusable analysis scripts (generated figures are ignored)
tests/           unit, integration, and regression coverage
run_split.py     main environment-driven SplitFed orchestrator
```

Datasets, checkpoints, logs, plots, diagnostics, and experiment outputs are intentionally excluded from Git. This keeps the repository portfolio-sized while preserving all local artifacts in the working directory.

## Saved forecasting result

The existing 84-client evaluation summary reports:

| Metric | Mean | Median | 90th percentile |
| --- | ---: | ---: | ---: |
| MAE | 4.582 | 1.782 | 10.912 |
| RMSE | 6.247 | 2.476 | 14.878 |

The best client MAE was **0.097**, and the mean client-level R² was **0.423**. Raw evaluation artifacts remain local and are not committed.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Place the prepared client data under `dataset/output/` (or update the data paths in `.env`), then run:

```bash
python run_split.py
```

The drift-mitigation pipeline exposes separate preparation, training, adaptation, and evaluation commands:

```bash
python -m drift.mitigation.prepared.cli --help
python -m drift.mitigation.prepared.cli run-full-suite --help
```

Run the automated checks with:

```bash
python -m unittest discover -s tests
```

## Tech stack

Python, PyTorch, Flask, NumPy, pandas, River, Prometheus, and YAML-based experiment configuration.
