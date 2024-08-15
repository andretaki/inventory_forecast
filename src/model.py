import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

class DataLoader:
    def __init__(self):
        host = os.getenv('DB_HOST')
        database = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        port = os.getenv('DB_PORT', 5432)

        encoded_password = quote_plus(password)
        self.connection_string = f"postgresql://{user}:{encoded_password}@{host}:{port}/{database}"
        self.engine = create_engine(self.connection_string)

    def load_sku_history(self, start_date, end_date):
        query = """
        SELECT o.order_date, oi.sku, oi.quantity, oi.unit_price, oi.name
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        WHERE o.order_date BETWEEN :start_date AND :end_date
        """
        df = pd.read_sql_query(text(query), self.engine, params={'start_date': start_date, 'end_date': end_date})
        return df

    def preprocess_data(self, df):
        # Convert order_date to datetime
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Calculate total price for each item
        df['total_price'] = df['quantity'] * df['unit_price']
        
        # Group by date and SKU, sum the quantities and total prices
        df_grouped = df.groupby(['order_date', 'sku']).agg({
            'quantity': 'sum',
            'total_price': 'sum',
            'name': 'first'  # Keep the name for reference
        }).reset_index()
        
        # Pivot the table to have SKUs as columns for quantity
        df_qty_pivoted = df_grouped.pivot(index='order_date', columns='sku', values='quantity').fillna(0)
        
        # Resample to daily frequency and forward fill missing values
        df_daily = df_qty_pivoted.resample('D').ffill()
        
        return df_daily

    def get_sku_info(self):
        query = """
        SELECT DISTINCT sku, name
        FROM order_items
        ORDER BY sku
        """
        return pd.read_sql_query(query, self.engine)

    def get_sku_list(self):
        return self.get_sku_info()['sku'].tolist()

if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load and preprocess data for the year 2023
    raw_data = loader.load_sku_history('2023-01-01', '2023-12-31')
    processed_data = loader.preprocess_data(raw_data)
    
    print("Processed data shape:", processed_data.shape)
    print("Number of SKUs:", len(processed_data.columns))
    
    # Display first few rows of processed data
    print(processed_data.head())
    
    # Get SKU information
    sku_info = loader.get_sku_info()
    print("\nSKU Information:")
    print(sku_info.head())

    # Example of how this data can inform purchasing decisions
    last_30_days = processed_data.last('30D')
    avg_daily_demand = last_30_days.mean()
    print("\nAverage daily demand for each SKU (last 30 days):")
    print(avg_daily_demand.sort_values(ascending=False).head())