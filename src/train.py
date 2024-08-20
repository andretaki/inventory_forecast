from data_loader import DataLoader
from model import InventoryForecastModel
from datetime import datetime, timedelta
import numpy as np

def train_model(start_date, end_date, sku):
    loader = DataLoader()
    raw_data = loader.load_order_history(start_date, end_date)
    processed_data = loader.preprocess_data(raw_data)
    
    if sku not in processed_data.columns:
        raise ValueError(f"SKU {sku} not found in the data")
    
    sku_data = processed_data[sku].dropna().values
    
    model = InventoryForecastModel(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    model.fit(sku_data)
    
    return model, sku_data

if __name__ == "__main__":
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    sku = "BX-GU9X-YHC9"  # Example SKU
    
    trained_model, _ = train_model(start_date, end_date, sku)
    print(f"Model trained for SKU: {sku}")
