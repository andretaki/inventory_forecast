import os
import sys
from dotenv import load_dotenv
from datetime import datetime, timedelta

def main():
    try:
        from data_loader import DataLoader
        from forecaster import SKUForecaster
        from visualizer import Visualizer
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please make sure all required modules are installed and in the correct location.")
        sys.exit(1)

    # Load environment variables
    load_dotenv()

    # Initialize DataLoader
    loader = DataLoader()

    # Set date range for data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data

    # Load and preprocess data
    print("Loading data...")
    raw_data = loader.load_order_history(start_date, end_date)  # Changed from load_sku_history to load_order_history
    processed_data = loader.preprocess_data(raw_data)

    # Initialize and train forecaster
    print("Training models...")
    forecaster = SKUForecaster(processed_data)
    forecaster.train_models()

    # Make forecasts
    print("Making forecasts...")
    forecaster.make_forecasts()

    # Get example SKU
    example_sku = processed_data.columns[0]

    # Get forecast for example SKU
    forecast = forecaster.get_forecast(example_sku)

    # Initialize visualizer
    visualizer = Visualizer()

    # Plot forecast
    print(f"Plotting forecast for {example_sku}...")
    fig = visualizer.plot_forecast(processed_data[example_sku], forecast)
    fig.savefig(f"{example_sku}_forecast.png")

    # Print some metrics
    print("\nForecast Summary:")
    print(f"SKU: {example_sku}")
    print(f"Last observed value: {processed_data[example_sku].iloc[-1]}")
    print(f"Forecasted value (next day): {forecast['yhat'].iloc[-1]}")

    # Optionally, if you have implemented cross-validation
    if hasattr(forecaster, 'perform_cross_validation'):
        print("\nPerforming cross-validation...")
        cv_results = forecaster.perform_cross_validation(example_sku)
        print("Cross-validation results:")
        print(cv_results)

if __name__ == "__main__":
    main()
