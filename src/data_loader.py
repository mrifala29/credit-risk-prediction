import os
import pandas as pd

class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def load_csv(self, filename):
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} tidak ditemukan")
        
        df = pd.read_csv(file_path)
        print(f"[INFO] Data loaded: {file_path}, shape: {df.shape}")
        return df

    def filter_region(self, df: pd.DataFrame, region):
        """Filter data berdasarkan kolom Region.
           region bisa string (misal: 'East') atau list (misal: ['East', 'West'])."""
        if "Region" not in df.columns:
            raise ValueError("Kolom 'Region' tidak ada di dataset")

        if isinstance(region, str):
            df_filtered = df[df["Region"] == region].copy()
        elif isinstance(region, (list, tuple, set)):
            df_filtered = df[df["Region"].isin(region)].copy()
        else:
            raise ValueError("Parameter 'region' harus string atau list of string")

        print(f"[INFO] Data filtered by Region={region}, shape: {df_filtered.shape}")
        return df_filtered

    def get_data(self, source, **kwargs):
        if source == "local":
            filename = kwargs.get("filename")
            df = self.load_csv(filename)

            # cek apakah user ingin filter Region
            region = kwargs.get("region")
            if region:
                df = self.filter_region(df, region)

            return df
        else:
            raise ValueError(f"Unknown source: {source}")


if __name__ == "__main__":
    loader = DataLoader()

    # Contoh 1: tanpa filter region
    df_all = loader.get_data(source="local", filename="retail_store_inventory.csv")
    print(df_all)

    # Contoh 2: filter hanya East
    df_east = loader.get_data(source="local", filename="retail_store_inventory.csv", region="East")
    print(df_east.shape)

    # Contoh 3: filter multiple region
    df_multi = loader.get_data(source="local", filename="retail_store_inventory.csv", region=["East", "West"])
    print(df_multi.shape)
