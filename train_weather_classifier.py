import requests
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys
import sklearn 
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, RocCurveDisplay

# Step 1: Fetch the Data
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 34.0522,
    "longitude": -118.2437,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)
print(df.info())
print()

# Step 2: Engineer Labels
def good_range(row):
    # In LA, temperature drops quick unless it is in heat wave, so even the highest temp is 30C, it is still comfortable in the morning or evening to run.
    # Minimum of 0C or Maximum of 7C is too cold to run in LA, 10C as lowest and 18C as minimum highest means morning and evening should have a temperature of 15C+, which is more comfortable.
    # Hence, the threshold are changed to fit my personal preference and local climate more.
    return int(
        (18 <= row["temperature_2m_max"] <= 30) and 
        (row["temperature_2m_min"] >= 10) and 
        (row["precipitation_sum"] < 3.0) and
        (row["wind_speed_10m_max"] < 30)
    )


df['good'] = df.apply(good_range, axis=1)
print('Good/Bad day count:')
print(df['good'].value_counts())
print()

# Around 56% (205/365) of the days are not good and around 44% (160/365) of the days are good for running.
# This is reasonable as Los Angeles gets around 45% of a year to be in temperature range, adding to the rare huge rain 
# during winter and high wind gust, that reduces another 1% of those time.

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

# Step 3: Train and Tune
X = df[FEATURES]
y = df['good']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"X_train size:   {X_train.shape}")
print(f"X_test size:    {X_test.shape}")
print(f"y_train size:   {y_train.shape}")
print(f"y_test size:    {y_test.shape}")


pipe = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, solver='liblinear'))
    ]
)

param_grid = {
    'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    pipe,
    param_grid,
    scoring='roc_auc',
    cv=5,
    error_score='raise'
)

grid_search.fit(X_train, y_train)
print(f'Best C value:   {grid_search.best_params_['clf__C']}')
print(f'Best CV AUC:    {grid_search.best_score_:.4f}')
y_pred = grid_search.predict(X_test)
print()
print(f'Classification report:')
print(classification_report(y_test, y_pred))
print()
test_auc = roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1])
print(f'Test AUC:   {test_auc:.4f}')

fig, ax = plt.subplots(figsize=(6, 5))
fpr, tpr, _ = roc_curve(y_test, grid_search.predict_proba(X_test)[:, 1])
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name="Actual Classifier")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_title("ROC of the Classifier")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/weather_roc.png")
plt.close()

# The test AUC score is 0.8529, which means the model's quality is high, it can differentiate the classes nicely.
# It is about what I would expect, as the weather of LA are in general warmer, which means for the most part the model is linear
# except for only around 50 days where the highest temperature is lower than 18C. Hence the Logistic Regression model should work good.
# In the report, the FP and FN are on average the same, in predicting non-good days, FP is less than FN, while in predicting good days,
# FP is more than FN. In practice, I would like the FP to be lower, as a FP means people will prepare to run even though it is not a good
# day, which causes inconvenience if they prepared the running shoes or schedule. A FN is merely losing out a good day that they can 
# potentially run. Hence, I would rather the app under-recommend it.
# If I were to set the threshold for the real app, I would use a value higher than 0.5, this pushes the FP rate lower as it is harder for
# the model to flag a good day, that makes the model less prone to over-recommendation.

# Step 5: Save the Model
joblib.dump(grid_search.best_estimator_, 'models/weather_classifier.pkl')
print('Model saved to models/.')
metadata = {
    'python_version': sys.version,
    'sklearn_version': sklearn.__version__,
    'features': FEATURES,
    'best_params': grid_search.best_params_,
    'test_auc': round(test_auc, 4),
    'city_name': "Los Angeles",
    "latitude": 34.0522, 
    "longitude": -118.2437,
    "label_thresholds": {
        "temperature_2m_max": "18–30°C",
        "temperature_2m_min": ">= 10°C",
        "precipitation_sum":  "< 3.0 mm",
        "wind_speed_10m_max": "< 30 km/h",
    },
    "label_threshold_description": "Adjusted from the starter thresholds to better match the warmer climate in Los Angeles and my comfort level for running."
}
with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Metadata saved to models/")