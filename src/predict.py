import logging
import numpy as np
from datetime import datetime, timedelta
from data_loader import DataLoader
from model import InventoryForecastModel

logger = logging.getLogger(__name__)

def train_model(start_date, end_date, sku):
    logger.info(f"Training model for SKU: {sku}, start_date: {start_date}, end_date: {end_date}")
    loader = DataLoader()
    raw_data = loader.load_order_history(start_date, end_date)
    processed_data = loader.preprocess_data(raw_data)
    
    logger.info(f"Processed data shape: {processed_data.shape}")
    
    if sku not in processed_data.columns:
        logger.error(f"SKU {sku} not found in the data")
        raise ValueError(f"SKU {sku} not found in the data")
    
    sku_data = processed_data[sku].dropna().values
    logger.info(f"SKU data shape: {sku_data.shape}")
    
    # Prepare data for LSTM (sequence of 7 days to predict the next day)
    X, y = [], []
    for i in range(len(sku_data) - 7):
        X.append(sku_data[i:i+7])
        y.append(sku_data[i+7])
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to (samples, time steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], 1))
    
    logger.info(f"Training data shapes - X: {X.shape}, y: {y.shape}")
    
    model = InventoryForecastModel(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    model.fit(X, y)
    
    logger.info("Model training completed")
    return model, sku_data

def make_prediction(sku, days_to_predict=30):
    logger.info(f"Making prediction for SKU: {sku}, days: {days_to_predict}")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    model, historical_data = train_model(start_date, end_date, sku)
    
    # Prepare the input for prediction
    input_data = historical_data[-7:]  # Use the last 7 days of data
    input_data = input_data.reshape((7, 1))  # Reshape to (sequence_length, features)
    
    logger.info(f"Input data shape for prediction: {input_data.shape}")
    
    prediction = model.predict(input_data, days_to_predict)
    
    logger.info(f"Prediction completed. Result shape: {prediction.shape}")
    return prediction

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sku = "BX-GU9X-YHC9"  # Example SKU
    prediction = make_prediction(sku)
    print(f"30-day prediction for SKU {sku}:")
    print(prediction)
