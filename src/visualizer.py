import pandas as pd
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        pass

    def plot_forecast(self, historical_data, forecast):
        plt.figure(figsize=(12, 6))
        plt.plot(historical_data.index, historical_data, label='Historical')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Forecast vs Historical Data')
        plt.legend()
        return plt.gcf()  # Return the current figure
