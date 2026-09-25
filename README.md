# CTD-Cloud-ETL-Capstone

Capstone project for CTD Python AI & Cloud Computing 26.3.

An end-to-end ETL pipeline that pulls daily weather data for Los Angeles from the [Open-Meteo](https://open-meteo.com/) API, stores it in [Supabase](https://supabase.com/), runs it through a trained ML classifier to predict whether a day is good for running, generates a natural-language recommendation with the OpenAI API, and writes the enriched results back to Supabase — orchestrated with [Prefect](https://www.prefect.io/).

## How it fits together

1. **`train_weather_classifier.py`** — pulls a year of historical LA weather from Open-Meteo, engineers a `good_for_running` label from temperature/precipitation/wind thresholds, tunes a `StandardScaler` + `LogisticRegression` pipeline with `GridSearchCV`, evaluates it (ROC/AUC), and saves the fitted model and its metadata to `models/`.
2. **`predict_weather.py`** — loads the saved model and metadata and runs it against a handful of hand-built test cases (clear/borderline good and bad days) to sanity-check predictions and probabilities.
3. **`populate_supabase.py`** — a standalone extract → transform → load → verify script that backfills the `weather_raw` Supabase table with a full year of historical daily weather.
4. **`etl_pipeline.py`** — the production Prefect flow:
   - **Extract**: fetches the current daily forecast from Open-Meteo.
   - **Load (raw)**: upserts it into the `weather_raw` Supabase table.
   - **Transform**: skips dates already enriched, runs the saved classifier to predict `good_for_running` + confidence, then calls the OpenAI API to generate a one-sentence running recommendation per day (with basic output validation).
   - **Load (enriched)**: upserts the results into the `weather_enriched` Supabase table.

## Project structure

```
.
├── etl_pipeline.py               # Prefect flow: extract → load raw → transform (ML + LLM) → load enriched
├── populate_supabase.py          # One-off backfill of historical weather into weather_raw
├── train_weather_classifier.py   # Trains and saves the "good for running" classifier
├── predict_weather.py            # Sanity-checks the saved model against sample days
├── models/                       # Saved model (weather_classifier.pkl) and metadata (weather_classifier_metadata.json)
├── outputs/                      # Generated artifacts (e.g. weather_roc.png)
└── .gitignore
```

## Setup

### 1. Install dependencies

The scripts rely on:

```
pandas
requests
scikit-learn
matplotlib
joblib
python-dotenv
supabase
openai
prefect
```

Install them with:

```bash
pip install pandas requests scikit-learn matplotlib joblib python-dotenv supabase openai prefect
```

or do

```bash
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root (already excluded via `.gitignore`) with:

```
SUPABASE_URL=your-supabase-project-url
SUPABASE_KEY=your-supabase-service-or-anon-key
OPENAI_API_KEY=your-openai-api-key
```

### 3. Supabase tables

The pipeline expects two tables in your Supabase project, each keyed on `date`:

- **`weather_raw`** — raw daily weather: `date`, `temperature_2m_max`, `temperature_2m_min`, `precipitation_sum`, `wind_speed_10m_max`.
- **`weather_enriched`** — model output: `date`, `good_for_running`, `confidence`, `llm_summary`.

Both are upserted on `date`, so the pipeline is safe to re-run.

### 4. `config.json`

`etl_pipeline.py` reads its Open-Meteo request configuration from a `config.json` file in the project root (not included in the repo). It should look like:

```json
{
  "OPEN_METEO_URL": "https://api.open-meteo.com/v1/forecast",
  "OPEN_METEO_PARAMS": {
    "latitude": 34.0522,
    "longitude": -118.2437,
    "daily": [
      "temperature_2m_max",
      "temperature_2m_min",
      "precipitation_sum",
      "wind_speed_10m_max"
    ],
    "timezone": "America/Los_Angeles"
  },
  "OPEN_METEO_FEATURES": [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "date"
  ],
  "MODEL_DIR": "models"
}
```

## Usage

Run these from the project root, in order, the first time you set things up:

```bash
# 1. Train the classifier (saves models/weather_classifier.pkl + metadata)
python train_weather_classifier.py

# 2. (Optional) sanity-check the model on sample days
python predict_weather.py

# 3. (Optional) backfill a year of historical data into weather_raw
python populate_supabase.py

# 4. Run the full ETL pipeline (forecast → predict → LLM summary → enriched table)
python etl_pipeline.py
```

`etl_pipeline.py` can be scheduled to run daily (e.g. via a Prefect deployment/schedule) to keep `weather_raw` and `weather_enriched` up to date automatically.

## Model notes

- The classifier is a `LogisticRegression` (with `StandardScaler`) tuned over a regularization grid via 5-fold cross-validated `GridSearchCV`, optimizing ROC AUC.
- The "good for running" label is a rule-based heuristic tuned for LA's climate: max temp 18–30°C, min temp ≥ 10°C, precipitation < 3.0 mm, and max wind speed < 30 km/h.
- Model artifacts (`weather_classifier.pkl`, `weather_classifier_metadata.json`) and the ROC curve plot (`outputs/weather_roc.png`) are produced by `train_weather_classifier.py`.