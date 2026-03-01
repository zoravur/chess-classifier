# Chess Classifier

Chess classifier is a classifier that predicts the likely outcome a chess game given a position and the 
elo rating of the players.

## Quickstart
TODO

## Project structure

```
.
├── configs/        # Hyperparameters, model configs (YAML/JSON)
├── data/           # Data processing code
├── evaluation/     # Metrics, evalution scripts
├── models/         # Model definitions
├── notebooks/      # Jupyter notebooks
├── scripts/        # Entry points
└── tests/          # Unit tests
```

## Roadmap

### Stage 1 -- Baseline
- [x] Project setup (ruff, pre-commit, Hydra)
- [ ] Preliminary exploration with duckdb
    - [x] Understand data format
    - [ ] Document findings in a Jupyter Notebook
        - [ ] Explain why duckdb and Aix
        - [ ] Some insights
- [ ] Dataset loading with HF Datasets
- [ ] Model architecture
- [ ] Training loop with accelerate

### Stage 2 -- Evalution and Analysis
TODO