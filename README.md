# Water Bottle Vision Pipeline

End-to-end **Machine Learning pipeline** that classifies the **fill level** of a water bottle
from an image: `full` · `half` · `overflow`.

It's not just a model — it's the full cycle working together: **EDA → preprocessing →
training → REST API deployment → PostgreSQL persistence → monitoring with drift detection →
retraining**, all containerized with Docker.

> **About the domain.** Classifying the fill level is conceptually a **visual quality-inspection**
> task, analogous to a bottling line (spotting under-filled or overflowing bottles).
> **Honest note:** the dataset is public stock/product imagery collected from the web, **not**
> real captures from an industrial line. The value of this repo is the **ML engineering of the
> pipeline**, not the dataset.

---

## Demo

**Classifier** — upload an image and get the predicted fill level + confidence:

![Classifier UI](docs/img/classifier.png)

**Monitoring dashboard** — prediction volume, class distribution, confidence, real accuracy (where labels exist) and PSI drift vs. the training baseline:

![Monitoring dashboard](docs/img/dashboard.png)

---

## End-to-end flow

```
                 ┌──────────────┐
   data/<class>/ │  notebooks/  │  EDA: counts, samples, stratified split
   images ─────► │   eda.ipynb  │  (exploration)
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │  src/train.py │  ResNet18 transfer learning + class weights
                 │  (retrain)    │  → artifacts/<version>.pt
                 └──────┬───────┘  → artifacts/metrics_<version>.json
                        │          → artifacts/reference_stats.json (drift baseline)
                        ▼
                 ┌──────────────┐        ┌──────────────────┐
   image ───────►│ src/app  API │───────►│   PostgreSQL     │  images / predictions / labels
   (POST /predict)│   FastAPI   │  stores│  (Alembic migr.) │
                 └──────┬───────┘  input/└────────┬─────────┘
                        │          pred           │
        POST /label ────┤          (+label)       │
                        ▼                         ▼
                 ┌──────────────┐        ┌──────────────────┐
                 │ /dashboard   │◄───────│  GET /monitoring │  volume, confidence,
                 │  (web)       │ consumes│  + drift (PSI)  │  real accuracy, drift
                 └──────────────┘        └──────────────────┘
                        │
                        └──► on drift / new labels → back to src/train.py
```

---

## Stack

| Layer | Technology |
|---|---|
| Model | PyTorch · torchvision (ResNet18, transfer learning) |
| Service | FastAPI · Uvicorn · REST API |
| Persistence | PostgreSQL 16 (psycopg v3) |
| Migrations | Alembic |
| Frontend | Vanilla HTML/CSS/JS (classifier + monitoring dashboard) |
| Packaging | Docker · Docker Compose |
| EDA | Jupyter · pandas · matplotlib · scikit-learn |

---

## Layout

```
.
├── src/
│   ├── app/
│   │   ├── main.py         # API: /predict, /label, /monitoring, /dashboard, /health
│   │   ├── monitoring.py   # aggregations + PSI (drift detection)
│   │   ├── config.py       # environment-based configuration
│   │   └── db.py           # PostgreSQL connection
│   └── train.py            # (re)training pipeline (CLI)
├── migrations/             # Alembic (versioned schema)
│   ├── env.py
│   └── versions/0001_initial_schema.py
├── sql/schema.sql          # reference snapshot of the schema
├── web/
│   ├── index.html          # classifier UI
│   ├── dashboard.html      # monitoring dashboard
│   └── styles.css
├── notebooks/eda.ipynb     # EDA + original training
├── artifacts/
│   ├── waterbottle_resnet18_v1.pt   # model served by the API
│   └── reference_stats.json         # drift baseline
├── samples/<class>/        # demo subset (15 images per class)
├── docker-compose.yml      # PostgreSQL + API
├── Dockerfile
├── entrypoint.sh           # applies migrations and starts the API
├── alembic.ini
├── requirements.txt
└── .env.example
```

---

## Run it

### Option A — Docker (recommended)

Brings up PostgreSQL + the API, applies migrations automatically, and leaves everything ready.

```bash
docker compose up -d --build
```

- Classifier: <http://localhost:8000/>
- Monitoring dashboard: <http://localhost:8000/dashboard>
- Health check: <http://localhost:8000/health>

Stop everything: `docker compose down` (add `-v` to also drop the database).

### Option B — Local (app not containerized)

```bash
# 1) Dependencies (a virtualenv is recommended)
pip install -r requirements.txt

# 2) Database (you can run just the Postgres container)
docker compose up -d db

# 3) Environment variables
cp .env.example .env        # adjust DATABASE_URL if needed
export DATABASE_URL="postgresql://app:app@localhost:5432/bottles"

# 4) Migrations
alembic upgrade head

# 5) API
uvicorn src.app.main:app --reload
```

---

## API

| Method | Path | Description |
|---|---|---|
| `GET`  | `/` | Classifier UI |
| `GET`  | `/dashboard` | Monitoring dashboard |
| `GET`  | `/health` | Service and model status |
| `POST` | `/predict` | Takes an image, returns class + probabilities, persists input + prediction |
| `POST` | `/label` | Stores the ground-truth label for an already-predicted image |
| `GET`  | `/monitoring` | Monitoring + drift report (JSON consumed by the dashboard) |

```bash
# Prediction (sample images live under samples/)
curl -F "file=@samples/full/example.jpeg;type=image/jpeg" http://localhost:8000/predict
# -> {"pred_label":"full","confidence":0.89,"probs":{...},"image_id":1,...}

# Store ground truth for that prediction
curl -X POST http://localhost:8000/label \
     -H "Content-Type: application/json" \
     -d '{"image_id": 1, "true_label": "full"}'
```

---

## Retraining

`src/train.py` reproduces the full pipeline as a reproducible CLI script:

```bash
# Quick demo on the subset shipped in the repo:
python -m src.train --epochs 5 --data-dir samples

# Full training (needs the complete dataset under data/<class>/, not versioned):
python -m src.train --epochs 15

# Available parameters
python -m src.train --help
#   --version, --epochs, --batch-size, --lr, --img-size,
#   --val-size, --test-size, --seed, --no-class-weights

# Regenerate only the drift baseline from an existing artifact
python -m src.train --reference-only --artifact artifacts/waterbottle_resnet18_v1.pt
```

Each training run produces:
- `artifacts/<version>.pt` — the model (with embedded metadata and version).
- `artifacts/metrics_<version>.json` — test accuracy, classification report and confusion matrix.
- `artifacts/reference_stats.json` — baseline (class distribution + confidence histogram) for drift.

To point the API at a new model, set `MODEL_ARTIFACT=artifacts/<version>.pt` and restart.

Because it's a single command, retraining can be triggered by **cron** or **CI/CD** when
monitoring flags drift or enough new labels accumulate (see *Possible improvements*).

---

## Monitoring and drift detection

`GET /monitoring` (and the dashboard) compute, reading from PostgreSQL:

- **Volume** of predictions and a daily time series.
- **Predicted-class distribution**.
- **Confidence**: mean and low-confidence rate (configurable threshold).
- **Real performance**: accuracy and per-class recall **wherever labels exist** (`predictions`↔`labels` join).
- **Drift** vs. the training baseline, using **PSI (Population Stability Index)** over:
  - the predicted-class distribution, and
  - the confidence histogram.

PSI interpretation: `< 0.10` no relevant drift · `0.10–0.25` moderate drift · `> 0.25`
significant drift (a signal to investigate/retrain). PSI is meaningful above a certain traffic
volume; with few predictions it's naturally unstable.

---

## Data and model

- **Dataset:** ~545 images across 3 classes — `full` (327), `half` (163), `overflow` (55).
  There is **real class imbalance**. The full dataset is **not versioned** (size); the repo
  ships a demo subset under `samples/` (15 images per class) plus the trained model.
- **Imbalance handling:** `src/train.py` uses **class weights** inversely proportional to
  frequency in the loss (on by default).
- **No data leakage:** **stratified** train/val/test split with a fixed seed; data augmentation
  is applied **to train only**.
- **Model:** ImageNet-pretrained ResNet18, frozen backbone, only the final layer (`fc`) is
  trained. `src/train.py` also keeps the **best model by validation accuracy**.

### Metrics (model v1, trained in the notebook)

| Metric | Value (test) |
|---|---|
| Accuracy | **0.80** |
| Macro F1 | **0.85** |
| F1 `full` / `half` / `overflow` | 0.81 / 0.73 / 1.00 |

The main confusion is `full ↔ half`; `overflow` separates well (on a small test set).

---

## MLOps practices implemented

- **Reproducible** pipeline (seeds, parameterized CLI script, versioned artifacts).
- **Environment-based configuration** (no secrets in code; see `.env.example`).
- **Database migrations** with Alembic (versioned, idempotent schema).
- **Model versioning** recorded on every prediction (`model_version`).
- **Persistence** of inputs, predictions and labels → enables auditing and retraining.
- **Monitoring + drift** on real production data.
- **Reproducible containerized deployment** (Docker Compose: DB + API + migrations).

## Possible improvements (roadmap)

Not implemented yet, ordered by impact:

- **Automatic retraining triggered by drift/labels** (orchestrated by cron, GitHub Actions or
  Airflow) — today retraining is a manual command.
- **Model registry / experiment tracking** (MLflow or similar) instead of `.pt` files.
- **Automated tests** (unit tests for the API and the PSI computation) + CI.
- **Explainability** (Grad-CAM / SHAP) to inspect what the model attends to.
- **More and better-curated data** to reduce the `full ↔ half` confusion; evaluate on a larger test set.
- **Authentication / rate limiting** on the API before exposing it publicly.
- **GPU** for faster training (CPU only for now, for simplicity).

> Note: this is a **computer vision** problem, not time series — so walk-forward validation and
> temporal features do not apply.

---

## License and data

Educational / portfolio project. The dataset images come from public sources on the web and
are included for demonstration purposes only.
