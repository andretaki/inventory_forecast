import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from tqdm import tqdm

class SKUForecaster:
    def __init__(self, data, forecast_periods=30):
        self.data = data
        self.forecast_periods = forecast_periods
        self.models = {}
        self.forecasts = {}
        self.cv_results = {}
        self.performance_metrics = {}

    def prepare_data(self, sku):
        df = self.data[[sku]].reset_index()
        df.columns = ['ds', 'y']
        return df

    def train_models(self):
        for sku in tqdm(self.data.columns, desc="Training models"):
            df = self.prepare_data(sku)
            model = Prophet(daily_seasonality=True)
            model.fit(df)
            self.models[sku] = model

    def make_forecasts(self):
        for sku, model in tqdm(self.models.items(), desc="Making forecasts"):
            future = model.make_future_dataframe(periods=self.forecast_periods)
            forecast = model.predict(future)
            self.forecasts[sku] = forecast

    def get_forecast(self, sku):
        if sku in self.forecasts:
            return self.forecasts[sku][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        else:
            return None

    def get_purchase_recommendations(self, current_stock):
        recommendations = {}
        for sku, forecast in self.forecasts.items():
            last_30_days = forecast['yhat'].tail(30).sum()
            if sku in current_stock:
                needed = max(0, last_30_days - current_stock[sku])
                recommendations[sku] = needed
            else:
                recommendations[sku] = last_30_days
        return recommendations

    def get_seasonal_components(self, sku):
        if sku in self.models:
            model = self.models[sku]
            return model.component_modes()
        else:
            return None

    def perform_cross_validation(self, sku, initial='365 days', period='30 days', horizon='90 days'):
        if sku in self.models:
            df = self.prepare_data(sku)
            cv_results = cross_validation(self.models[sku], initial=initial, period=period, horizon=horizon)
            self.cv_results[sku] = cv_results
            metrics = performance_metrics(cv_results)
            self.performance_metrics[sku] = metrics
            return metrics
        else:
            return None

    def detect_anomalies(self, sku, threshold=0.05):
        if sku in self.forecasts:
            forecast = self.forecasts[sku]
            observed = self.prepare_data(sku)
            merged = forecast.merge(observed, on='ds', how='left')
            merged['error'] = merged['y'] - merged['yhat']
            merged['uncertainty'] = merged['yhat_upper'] - merged['yhat_lower']
            merged['normalized_error'] = merged['error'] / merged['uncertainty']
            anomalies = merged[abs(merged['normalized_error']) > threshold]
            return anomalies
        else:
            return None

    def simulate_scenario(self, sku, scenario_func):
        if sku in self.models:
            model = self.models[sku]
            future = model.make_future_dataframe(periods=self.forecast_periods)
            future = scenario_func(future)
            forecast = model.predict(future)
            return forecast
        else:
            return None

if __name__ == "__main__":
    from data_loader import DataLoader
    
    loader = DataLoader()
    data = loader.load_sku_history('2022-01-01', '2023-12-31')
    processed_data = loader.preprocess_data(data)
    
    forecaster = SKUForecaster(processed_data)
    forecaster.train_models()
    forecaster.make_forecasts()
    
    sku_example = processed_data.columns[0]
    
    # Cross-validation example
    cv_metrics = forecaster.perform_cross_validation(sku_example)
    print(f"Cross-validation metrics for {sku_example}:")
    print(cv_metrics)
    
    # Anomaly detection example
    anomalies = forecaster.detect_anomalies(sku_example)
    print(f"Detected anomalies for {sku_example}:")
    print(anomalies)
    
    # Scenario simulation example
    def increase_trend(future_df):
        future_df['trend_increase'] = (future_df['ds'] - future_df['ds'].min()).dt.days * 0.1
        return future_df
    
    scenario_forecast = forecaster.simulate_scenario(sku_example, increase_trend)
    print(f"Scenario forecast for {sku_example}:")
    print(scenario_forecast.tail())