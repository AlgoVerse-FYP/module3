# Module 3: Sequential Multi-Channel Fraud Detection

Bidirectional LSTM + merchant-network fusion for sequential fraud detection.
Every design choice is justified by an experiment (see `experiments/`).

## Folder structure

```
module3/
├── config.yaml                 → single source of truth for all hyperparameters
├── requirements.txt
├── data/
│   ├── raw/                    → fraudTrain.csv, fraudTest.csv (you add these)
│   ├── processed/              → cached feature tensors
│   └── synthetic/              → injected hard cases (Phase 3)
├── src/
│   ├── preprocessing/          → cleaning, temporal split, synthetic injection
│   ├── features/               → phase1_features.py (sequence + merchant features)
│   ├── models/                 → phase2_model.py (the Bi-LSTM + fusion model)
│   ├── evaluation/             → evaluate_model.py (metrics + figures)
│   └── explainability/         → SHAP + probability calibration (Phase 8)
├── experiments/                → THE EMPIRICAL JUSTIFICATION lives here
│   ├── window_size/            → why 5 transactions? the size sweep proves it
│   ├── ablation/               → baselines + bi-vs-uni + imbalance + learnability
│   └── hyperparameter_tuning/  → the search that justifies every hyperparameter
├── notebooks/                  → Colab notebooks
├── results/
│   ├── figures/                → PR curves, confusion matrices, window plot
│   ├── metrics/                → saved metric tables
│   └── models/                 → trained .keras / .onnx
├── configs/                    → config.yaml
├── tests/                      → leakage + feature sanity tests
└── docs/                       → workplan, Q&A guide, plan
```

## Principle
`config.yaml` holds every hyperparameter. Experiments read from it and write
results to `results/`. Nothing is hard-coded in two places — one change, one run,
reproducible.
