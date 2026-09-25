import joblib
import json
import pandas as pd

model = joblib.load("models/weather_classifier.pkl")
with open("models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)
print(f"City (lat, long):   {metadata['city']}")
print(f"Features:           {metadata['features']}")
print(f"Test AUC:           {metadata['test_auc']}")

test_cases = pd.DataFrame(
    [
        {"temperature_2m_max": 24.0, "temperature_2m_min": 15.0, "precipitation_sum": 0.0,  "wind_speed_10m_max": 12.0, "good": 1}, # Clearly good
        {"temperature_2m_max": 27.5, "temperature_2m_min": 18.0, "precipitation_sum": 1.2,  "wind_speed_10m_max": 22.0, "good": 1}, # Clearly good
        {"temperature_2m_max": 18.0, "temperature_2m_min": 10.0, "precipitation_sum": 0.0,  "wind_speed_10m_max": 0.0,  "good": 1}, # Borderline pass
        {"temperature_2m_max": 30.0, "temperature_2m_min": 12.0, "precipitation_sum": 2.99, "wind_speed_10m_max": 29.9, "good": 1}, # Borderline pass
        {"temperature_2m_max": 30.1, "temperature_2m_min": 15.0, "precipitation_sum": 0.0,  "wind_speed_10m_max": 10.0, "good": 0}, # Borderline bad
        {"temperature_2m_max": 22.0, "temperature_2m_min": 9.9,  "precipitation_sum": 0.0,  "wind_speed_10m_max": 10.0, "good": 0}, # Borderline bad
        {"temperature_2m_max": 25.0, "temperature_2m_min": 14.0, "precipitation_sum": 3.0,  "wind_speed_10m_max": 15.0, "good": 0}, # Borderline bad
        {"temperature_2m_max": 20.0, "temperature_2m_min": 11.0, "precipitation_sum": 0.5,  "wind_speed_10m_max": 30.0, "good": 0}, # Borderline bad
        {"temperature_2m_max": 22.0, "temperature_2m_min": 12.0, "precipitation_sum": 15.5, "wind_speed_10m_max": 45.0, "good": 0}, # Clearly bad
        {"temperature_2m_max": 12.0, "temperature_2m_min": 2.0,  "precipitation_sum": 8.0,  "wind_speed_10m_max": 50.0, "good": 0}, # Clearly bad
    ]
)

print('Test cases:')
print(test_cases)
print()

X_test = test_cases[metadata['features']]
y_test = test_cases['good']
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1].round(2)
test_cases['Prediction'] = y_pred
test_cases['Probability for good'] = y_proba
for index, row in test_cases.iterrows():
    print(f"Day {index + 1}: ")
    print_msg = ""
    for col in test_cases.columns:
        if col != 'good':
            print_msg += f"{col}: {row[col]}\t"
    print(print_msg)

# --- Reflection ---

# For the borderline cases, the model is really confident on id #3 (P = 0.0), but it is incorrect. It is not so confident on id #7 (P = 0.56), which is also incorrect.
# The model is correct on the other borderline cases, but not too sure on id #4, 5 as well (P = 0.27, 0.31). With a day where the model says 0.52, I will probably mark
# it as not good, as I want to lower the FPR to avoid frustration from users.

# If someone ran `predict_weather.py` before `train_weather_classifier.py`, it will throw a FileNotFound error as there will be no model and metadata file to load.
# We can do a try-except block and raise an error message of "Model file not found. Have you run the training script yet?".

# In a production system, this script may need to add an API call to fetch the next day weather forecast data, as well as some verification code to ensure the input stream
# is of the correct format. We also need to add scheduled automation script as well, or host it in some automation dashboard like Prefect. We may also want to hook the output
# to another app instead of just outputing as a numpy array or pandas dataframe.
