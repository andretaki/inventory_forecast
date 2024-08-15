import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

class DataLoader:
    def __init__(self):
        # Load database credentials from environment variables
        host = os.getenv('DB_HOST')
        database = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        port = os.getenv('DB_PORT', 5432)

        # Safely encode the password
        encoded_password = quote_plus(password)
        self.connection_string = f"postgresql://{user}:{encoded_password}@{host}:{port}/{database}"
        self.engine = create_engine(self.connection_string)

    def load_order_history(self, start_date, end_date):
        query = """
        SELECT * FROM order_history
        WHERE order_date BETWEEN %(start_date)s AND %(end_date)s
        """
        df = pd.read_sql_query(query, self.engine, params={'start_date': start_date, 'end_date': end_date})
        return df

    def preprocess_data(self, df):
        # Convert order_date to datetime
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Sort by date
        df = df.sort_values('order_date')
        
        # Group by date and product, sum the quantities
        df = df.groupby(['order_date', 'product_id'])['quantity'].sum().reset_index()
        
        # Pivot the table to have products as columns
        df_pivoted = df.pivot(index='order_date', columns='product_id', values='quantity').fillna(0)
        
        # Resample to daily frequency and forward fill missing values
        df_daily = df_pivoted.resample('D').ffill()
        
        return df_daily

    def get_table_names(self):
        return self.engine.table_names()

    def get_product_list(self):
        query = "SELECT DISTINCT product_id FROM order_history ORDER BY product_id"
        return pd.read_sql_query(query, self.engine)['product_id'].tolist()

if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Print available tables
    print("Available tables:", loader.get_table_names())
    
    # Load and preprocess data
    raw_data = loader.load_order_history('2023-01-01', '2023-12-31')
    processed_data = loader.preprocess_data(raw_data)
    
    print("Processed data shape:", processed_data.shape)
    print("Processed data columns:", processed_data.columns)
    
    # Get list of products
    products = loader.get_product_list()
    print("Number of unique products:", len(products))