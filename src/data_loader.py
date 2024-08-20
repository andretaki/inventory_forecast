import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

class DataLoader:
    def __init__(self):
        try:
            self.api_key = os.getenv('SHIPSTATION_API_KEY')
            self.api_secret = os.getenv('SHIPSTATION_API_SECRET')
            self.base_url = "https://ssapi.shipstation.com"

            if not all([self.api_key, self.api_secret]):
                raise ValueError("Missing ShipStation API credentials in environment variables")

            self.headers = {
                "Authorization": f"Basic {self._get_auth_header()}",
                "Content-Type": "application/json"
            }
            logging.info("ShipStation API connection established successfully")
        except Exception as e:
            logging.error(f"Error initializing DataLoader: {str(e)}")
            raise

    def _get_auth_header(self):
        auth_string = f"{self.api_key}:{self.api_secret}"
        return base64.b64encode(auth_string.encode()).decode()

    def load_order_history(self, start_date, end_date):
        try:
            url = f"{self.base_url}/orders"
            params = {
                "createDateStart": start_date.isoformat(),
                "createDateEnd": end_date.isoformat(),
                "pageSize": 500  # Adjust as needed
            }
            
            all_orders = []
            page = 1
            
            while True:
                params['page'] = page
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data['orders']:
                    break
                
                all_orders.extend(data['orders'])
                page += 1

            df = pd.DataFrame(all_orders)
            logging.info(f"Loaded {len(df)} orders from ShipStation API")
            return df
        except Exception as e:
            logging.error(f"Error loading order history: {str(e)}")
            raise

    def preprocess_data(self, df):
        try:
            # Convert createDate to datetime
            df['createDate'] = pd.to_datetime(df['createDate'], errors='coerce')
            
            # Sort by date
            df = df.sort_values('createDate')
            
            # Function to safely extract SKU and quantity
            def extract_sku_quantity(item):
                if isinstance(item, dict):
                    return item.get('sku'), item.get('quantity', 0)
                elif isinstance(item, list) and item:
                    return item[0].get('sku'), item[0].get('quantity', 0)
                else:
                    return None, 0

            # Extract SKU and quantity from items
            df['items'] = df['items'].apply(lambda x: [x] if isinstance(x, dict) else x)
            df['items'] = df['items'].apply(lambda x: x if isinstance(x, list) else [])
            df_items = df.explode('items')
            df_items[['sku', 'quantity']] = df_items['items'].apply(extract_sku_quantity).tolist()
            
            # Convert quantity to numeric, replacing any non-numeric values with 0
            df_items['quantity'] = pd.to_numeric(df_items['quantity'], errors='coerce').fillna(0)
            
            # Group by date and SKU, sum the quantities
            df_daily = df_items.groupby(['createDate', 'sku'])['quantity'].sum().unstack(fill_value=0)
            
            # Resample to daily frequency and forward fill missing values
            df_daily = df_daily.resample('D').ffill()
            
            logging.info(f"Data preprocessed successfully. Shape: {df_daily.shape}")
            return df_daily
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            raise

    def get_product_list(self):
        try:
            url = f"{self.base_url}/products"
            params = {"pageSize": 500}
            
            all_products = []
            page = 1
            
            while True:
                params['page'] = page
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data['products']:
                    break
                
                all_products.extend(data['products'])
                page += 1

            products = [product['sku'] for product in all_products if product['sku']]
            logging.info(f"Retrieved {len(products)} unique products")
            return products
        except Exception as e:
            logging.error(f"Error getting product list: {str(e)}")
            raise

    def get_daily_order_totals(self, start_date, end_date):
        try:
            df = self.load_order_history(start_date, end_date)
            df['createDate'] = pd.to_datetime(df['createDate'], errors='coerce')
            daily_totals = df.groupby(df['createDate'].dt.date)['orderTotal'].sum().reset_index()
            daily_totals.columns = ['order_date', 'total_amount']
            logging.info(f"Retrieved daily order totals from {start_date} to {end_date}")
            return daily_totals
        except Exception as e:
            logging.error(f"Error getting daily order totals: {str(e)}")
            raise

    def get_top_selling_skus(self, start_date, end_date, top_n=10):
        try:
            df = self.load_order_history(start_date, end_date)
            df_items = df.explode('items')
            df_items['sku'] = df_items['items'].apply(lambda x: x.get('sku') if isinstance(x, dict) else None)
            df_items['quantity'] = df_items['items'].apply(lambda x: x.get('quantity', 0) if isinstance(x, dict) else 0)
            
            top_skus = df_items.groupby('sku')['quantity'].sum().nlargest(top_n).reset_index()
            logging.info(f"Retrieved top {top_n} selling SKUs from {start_date} to {end_date}")
            return top_skus
        except Exception as e:
            logging.error(f"Error getting top selling SKUs: {str(e)}")
            raise

def main():
    try:
        loader = DataLoader()
        
        # Example usage
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print(f"Querying data from {start_date} to {end_date}")
        
        raw_data = loader.load_order_history(start_date, end_date)
        
        if len(raw_data) > 0:
            print("Raw data sample:")
            print(raw_data.head())
            
            processed_data = loader.preprocess_data(raw_data)
            print("\nProcessed data shape:", processed_data.shape)
            print("Processed data columns:", processed_data.columns)
            print("\nProcessed data sample:")
            print(processed_data.head())

            # Get daily order totals
            daily_totals = loader.get_daily_order_totals(start_date, end_date)
            print("\nDaily Order Totals:")
            print(daily_totals)

            # Get top selling SKUs
            top_skus = loader.get_top_selling_skus(start_date, end_date)
            print("\nTop Selling SKUs:")
            print(top_skus)
        else:
            print(f"No data found between {start_date} and {end_date}")
        
        # Get list of products
        products = loader.get_product_list()
        print("\nNumber of unique products:", len(products))
        print("Sample of products:", products[:5] if len(products) > 5 else products)

    except Exception as e:
        logging.error(f"An error occurred in main execution: {str(e)}")

if __name__ == "__main__":
    main()
