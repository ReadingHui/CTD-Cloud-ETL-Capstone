# Video link: https://youtu.be/h-jIH7v3CbQ

import requests
import pandas as pd
import os
from supabase import create_client
from dotenv import load_dotenv


# --- Step 1: Extract ---
def extract_weather(lat: float = 34.0522, 
                    lon: float = -118.2437, 
                    start: str = "2023-01-01", 
                    end: str = "2023-12-31", 
                    vars: list = [
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_sum",
                        "wind_speed_10m_max"
                    ]
                    ) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": vars,
        "timezone": "auto",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["daily"])
    df["date"] = pd.to_datetime(df["time"]).astype(str)
    df = df.drop("time", axis=1)
    print("=== Response from Open Mateo API ===")
    df.info()
    return df

# --- Step 2: Transform ---
def get_row_dict(df: pd.DataFrame) -> list:
    row_dict = df.to_dict('records')
    print(f"First record: {row_dict[0]}")
    print(f"Last record: {row_dict[-1]}")
    return row_dict

# I would expect 365 records for a full year, and I did get 365. If the numbers differs, there may be missing weather data from the API for a certain days.

# --- Step 3: Load ---
def get_client():
    load_dotenv()
    try:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
    except KeyError as e:
        raise RuntimeError(f"Missing required Supabase credential: {e}")
    supabase = create_client(url, key)
    return supabase

def safe_upsert(supabase, records):
    response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    print(f"{len(response.data)} rows upserted.")

# Idempotency is important, otherwise if I say, used insert() instead of upsert(), it will raise error in the second update and break the data pipeline.

# --- Step 4: Verify ---
def verify_upsert(supabase):
    total_rows = (
        supabase
        .table("weather_raw")
        .select("*", count="exact")
        .limit(0)     # Not actually fetching any data
        .execute()
    )
    print(f"Total number of rows in `weather_raw`: {total_rows.count}")
    first_date = (
        supabase
        .table("weather_raw")
        .select("date")
        .order("date", desc=False)
        .limit(1)    
        .execute()
    )
    last_date = (
        supabase
        .table("weather_raw")
        .select("date")
        .order("date", desc=True)
        .limit(1)    
        .execute()
    )
    print(f"Earliest date:  {first_date.data[0]["date"]}")
    print(f"Latest date:    {last_date.data[0]["date"]}")
    spec_row = (
        supabase
        .table("weather_raw")
        .select("*")
        .eq("date", "2023-07-04")
        .execute()
    )
    print(f"Row for 2023-07-04: {spec_row.data[0]}")
    



if __name__ == "__main__":
    if os.path.exists("assignments_09/temp/weather_data.csv"):
        df = pd.read_csv("assignments_09/temp/weather_data.csv") # Local testing so I don't call the API everytime
    else:
        df = extract_weather() # --- Step 1: Extract ---
        df.to_csv("assignments_09/temp/weather_data.csv", index=False)
    row_dict = get_row_dict(df) # --- Step 2: Transform ---
    supabase = get_client()
    safe_upsert(supabase, row_dict) # --- Step 3: Load ---
    verify_upsert(supabase) #--- Step 4: Verify ---
