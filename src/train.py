import joblib
import logging
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from preprocessing import Preprocessor
from data_loader import DataLoader

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Trainer:
    def __init__(self, model=None, n_estimators=100):
        # Pakai warm_start supaya bisa update model estimator per estimator
        self.model = model if model else RandomForestRegressor(
            n_estimators=1,  # mulai dari 1
            warm_start=True,
            random_state=42,
            n_jobs=-1
        )
        self.n_estimators = n_estimators

    def train(self, X_train, y_train):
        logging.info(f"Mulai training dengan {X_train.shape[0]} samples dan {X_train.shape[1]} features")

        # Train dengan progress bar
        for i in tqdm(range(1, self.n_estimators + 1), desc="Training Progress"):
            self.model.n_estimators = i
            self.model.fit(X_train, y_train)

        joblib.dump(self.model, "models/model.pkl")
        logging.info("Model tersimpan sebagai model.pkl")
        return self.model


if __name__ == "__main__":
    # Load raw data
    loader = DataLoader()
    df = loader.get_data(source="local", filename="retail_store_inventory.csv", region="East")

    # Preprocessing (dipisah dari Trainer)
    pre = Preprocessor(
        target_col="Units Sold",
        drop_cols=["Region", "Demand Forecast"],
        date_cols=["Date"],
        test_size=0.2
    )
    X_train, X_test, y_train, y_test = pre.get_processed_data(df)

    # Train model
    trainer = Trainer(n_estimators=100)  # bisa ubah jumlah trees disini
    model = trainer.train(X_train, y_train)
