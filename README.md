# Chess Classifier

Chess classifier is a classifier that predicts the likely outcome a chess game given a position and the 
elo rating of the players.

## Quickstart

Setup:
```
uv pip install -e .
```

To run a notebook: 
```
uv run --with jupyter jupyter lab
```

To run a script:
```
uv run scripts/<script_name>
```
See the relevant script in `scripts/` for more information.

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
- [x] Preliminary exploration with duckdb
- [ ] Train baselines on subset on small subset of data in Jupyter Notebook
    - [x] Understand data format
    - [ ] Document findings in a Jupyter Notebook
        - [ ] Explain why duckdb and Aix
        - [ ] Some insights
            - [ ] Number of games
            - [x] Elo histogram
- [ ] Productionize
    - [ ] Dataset loading with HF Datasets
    - [ ] Model architecture
    - [ ] Training loop with accelerate

### Stage 2 -- Evalution and Analysis
TODO