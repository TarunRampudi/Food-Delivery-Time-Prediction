
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(
    title="Food Delivery Time Prediction",
    description="Delivery time prediction using Tuned XGBoost",
    version="1.0"
)

# Load trained pipeline
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "food_delivery_model.pkl")


@app.get("/")
def home():
    return {
        "message": "Food Delivery Time Prediction API is running!"
    }


# Input details
class DeliveryInput(BaseModel):
    Delivery_person_Age: float
    Delivery_person_Ratings: float

    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float

    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str

    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: int
    Festival: str
    City: str


@app.post("/predict")
def predict(data: DeliveryInput):

    # Create input DataFrame
    df = pd.DataFrame([data.model_dump()])

    # Date features
    order_date = pd.to_datetime(
        df["Order_Date"], format="%d-%m-%Y"
    )

    df["Order_Day"] = order_date.dt.day
    df["Order_Month"] = order_date.dt.month
    df["Order_Year"] = order_date.dt.year
    df["Order_Weekday"] = order_date.dt.dayofweek

    # Time features
    order_time = pd.to_datetime(
        df["Time_Orderd"], format="mixed"
    )

    pickup_time = pd.to_datetime(
        df["Time_Order_picked"], format="mixed"
    )

    df["Order_Hour"] = order_time.dt.hour
    df["Order_Minute"] = order_time.dt.minute
    df["Order_Second"] = order_time.dt.second

    df["Pickup_Hour"] = pickup_time.dt.hour
    df["Pickup_Minute"] = pickup_time.dt.minute
    df["Pickup_Second"] = pickup_time.dt.second

    # Distance calculation (Haversine)
    lat1 = np.radians(df["Restaurant_latitude"])
    lon1 = np.radians(df["Restaurant_longitude"])
    lat2 = np.radians(df["Delivery_location_latitude"])
    lon2 = np.radians(df["Delivery_location_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    df["Distance_km"] = (
        6371 * 2 * np.arcsin(np.sqrt(a))
    )

    # Select exactly the training features
    feature_columns = [
        'Delivery_person_Age',
        'Delivery_person_Ratings',
        'Restaurant_latitude',
        'Restaurant_longitude',
        'Delivery_location_latitude',
        'Delivery_location_longitude',
        'Weatherconditions',
        'Road_traffic_density',
        'Vehicle_condition',
        'Type_of_order',
        'Type_of_vehicle',
        'multiple_deliveries',
        'Festival',
        'City',
        'Distance_km',
        'Order_Day',
        'Order_Month',
        'Order_Year',
        'Order_Weekday',
        'Order_Hour',
        'Order_Minute',
        'Order_Second',
        'Pickup_Hour',
        'Pickup_Minute',
        'Pickup_Second'
    ]

    X = df[feature_columns]

    # Prediction
    prediction = model.predict(X)[0]

    return {
        "predicted_delivery_time_minutes": round(float(prediction), 2)
    }
