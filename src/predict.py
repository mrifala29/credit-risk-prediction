# predict.py
import joblib
import pandas as pd
from data_loader import DataLoader
from preprocessing import Preprocessor

class Predictor:
    def __init__(self, model_path="models/model.pkl", preprocessor_path="models/preprocessor.pkl"):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, df: pd.DataFrame, target_col="Units Sold"):
        # Simpan target kalau ada (untuk evaluasi)
        y_true = None
        if target_col in df.columns:
            y_true = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            X = df

        # Proses tanggal biar sama kaya training
        pre = Preprocessor(
            target_col=target_col,
            drop_cols=["Region", "Demand Forecast"],  # sama seperti training
            date_cols=["Date"]
        )
        X_processed = pre._process_dates(X)

        # Transform pakai preprocessor yang sudah disimpan
        X_ready = self.preprocessor.transform(X_processed)

        # Prediksi
        preds = self.model.predict(X_ready)

        return preds, y_true


if __name__ == "__main__":
    # --- Load data West ---
    loader = DataLoader()
    df_west = loader.get_data(source="local", filename="retail_store_inventory.csv", region="West")

    predictor = Predictor()

    preds, y_true = predictor.predict(df_west, target_col="Units Sold")

    # Kalau ada ground truth, tampilkan contoh evaluasi singkat
    if y_true is not None:
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_true, preds)
        print(f"MAE untuk Region=West: {mae:.2f}")

    # Cetak 5 hasil prediksi
    print("Contoh prediksi (5 baris):")
    print(preds[:5])
