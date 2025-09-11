from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Inventory Forecasting API")

# Paths
MODEL_PATH = os.path.join("models", "model.pkl")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")

# Load model + preprocessor
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Pydantic model untuk request body
class InventoryData(BaseModel):
    store_id: str
    product_id: str
    category: str
    region: str
    inventory_level: float
    units_sold: float
    units_ordered: float
    price: float
    discount: float
    weather_condition: str
    holiday_promotion: int
    competitor_pricing: float
    seasonality: str

# Mapping kolom JSON -> nama kolom asli dataset
COLUMN_MAPPING = {
    "store_id": "Store ID",
    "product_id": "Product ID",
    "category": "Category",
    "region": "Region",
    "inventory_level": "Inventory Level",
    "units_sold": "Units Sold",
    "units_ordered": "Units Ordered",
    "price": "Price",
    "discount": "Discount",
    "weather_condition": "Weather Condition",
    "holiday_promotion": "Holiday/Promotion",
    "competitor_pricing": "Competitor Pricing",
    "seasonality": "Seasonality"
}

def preprocess_input(data: dict) -> pd.DataFrame:
    # Ubah snake_case ke nama kolom asli
    renamed_data = {COLUMN_MAPPING[k]: v for k, v in data.items() if k in COLUMN_MAPPING}
    return pd.DataFrame([renamed_data])

@app.post("/predict")
def predict(data: InventoryData):
    # Convert JSON -> DataFrame dengan nama kolom sesuai dataset
    df = preprocess_input(data.dict())

    # Transform pakai preprocessor
    X = preprocessor.transform(df)

    # Prediksi
    y_pred = model.predict(X)[0]

    return {"forecast": float(y_pred)}
