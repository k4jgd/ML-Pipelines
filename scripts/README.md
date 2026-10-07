# ML pipelines

Each pipeline owns its implementation:

```text
scripts/
├── common/
│   ├── mlflow_setup.py     # shared tracking configuration
│   └── registry.py         # champion comparison and promotion
├── cancellation/
├── customer_intelligence/
├── demand_operations/
├── machine_intelligence/
├── online_booking_forecast/
├── utilization/
└── walk_in_forecast/
```

Every pipeline folder contains:

- `config.py`: target, feature columns, model registry name, and metric settings.
- `preprocess.py`: loading, cleaning, feature engineering, and chronological splitting.
- `models.py`: candidate model definitions and fitting helpers.
- `validate.py`: evaluation metrics and prediction validation.
- `train.py`: training entry point, plots, MLflow logging, and model promotion.
- `__init__.py`: package marker.

Cancellation, online booking, and walk-in pipelines reuse the demand operations
preprocessing directly. Online booking and walk-in also reuse its regression
models and metrics. Imports name the owning module explicitly.

## Setup and training

From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m scripts.customer_intelligence.train
```

Replace `customer_intelligence` with any pipeline folder above. Direct execution
also works, for example `python scripts/customer_intelligence/train.py`.
Utilization accepts `--data-path` and `--min-improvement` options:

```sh
python -m scripts.utilization.train --help
```

Input CSV files live in `datasets/`; artifacts go to `outputs/<pipeline>/`.
MLflow uses `mlflow.db` at the repository root. Training and registration begin
only when `main()` is called or the training module is executed.

The former flat training files and `core/` compatibility modules have been
removed. Import helpers from their owning pipeline module, for example
`from scripts.customer_intelligence.preprocess import engineer_features`.
Shared infrastructure is available under `scripts.common`.
