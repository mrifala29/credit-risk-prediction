# evaluator.py
import joblib
import config
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_loader import DataLoader
from preprocessing import Preprocessor

class Evaluator:
    def __init__(self, model_path, preprocessor_path):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def evaluate(self, X_test, y_test):
        X_test_processed = self.preprocessor.transform(X_test)
        preds = self.model.predict(X_test_processed)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"R2 Score: {r2:.2f}")

        return {"mae": mae, "mse": mse, "r2": r2}


if __name__ == "__main__":
    # Load raw data
    loader = DataLoader()
    df = loader.get_data(source="local", filename="retail_store_inventory.csv", region="East")

    # Preprocessing
    pre = Preprocessor(
        target_col="Units Sold",
        drop_cols=["Region", "Demand Forecast"],
        date_cols=["Date"],
        test_size=0.2
    )
    _, X_test, _, y_test = pre.get_processed_data(df)

    # Evaluasi
    evaluator = Evaluator(
        model_path="models/model.pkl",
        preprocessor_path="models/preprocessor.pkl"
    )
    
    evaluator.evaluate(X_test, y_test)