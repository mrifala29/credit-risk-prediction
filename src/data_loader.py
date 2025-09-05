import os
import pandas as pd
class DataLoader:

    def __init__(self, data_dir = "data/"):
        self.data_dir = data_dir

    def load_csv(self, filename):
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} tidak ditemukan")
        
        df = pd.read_csv(file_path)
        print(f"[INFO] Data loaded: {file_path}, shape: {df.shape}")

        return df


    def get_data(self, source, **kwargs):

        if source == "local":
            filename = kwargs.get("filename")
            return self.load_csv(filename)
        else:
            raise ValueError(f"Unknown source: {source}")
        
    
if __name__ == "__main__":
    loader = DataLoader()

    # Load data lokal
    df_local = loader.get_data(source="local", filename="retail_store_inventory.csv")
    print(df_local.head())
    print(df_local.info())