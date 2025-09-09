import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Preprocessor:
    def __init__(self, target_col="target", drop_cols=None, date_cols=None, test_size=0.2):
        self.target_col = target_col
        self.drop_cols = drop_cols if drop_cols else []
        self.date_cols = date_cols if date_cols else []
        self.test_size = test_size
        self.preprocessor = None

    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df = df.sort_values(col)
                df = df.drop(columns=[col])
        return df

    def split_features_target(self, df: pd.DataFrame):
        df = df.drop(columns=self.drop_cols, errors="ignore")
        df = self._process_dates(df).reset_index(drop=True)
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        return X, y

    def create_preprocessor(self, X: pd.DataFrame):
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])
        return self.preprocessor

    def get_processed_data(self, df: pd.DataFrame):
        """
        Wrapper function:
        - split features & target
        - build preprocessing pipeline
        - transform train/test
        """
        X, y = self.split_features_target(df)

        # time series split
        split_index = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # buat pipeline jika belum ada
        if self.preprocessor is None:
            self.create_preprocessor(X_train)

        # transform data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader()
    df = loader.get_data(source="local", filename="retail_store_inventory.csv", region="East")

    pre = Preprocessor(
        target_col="Units Sold",
        drop_cols=["Region", "Demand Forecast"],
        date_cols=["Date"]
    )

    X_train, X_test, y_train, y_test = pre.get_processed_data(df)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
