import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from data_loader import DataLoader
from forecaster import SKUForecaster
import pandas as pd

class Dashboard:
    def __init__(self):
        self.loader = DataLoader()
        self.forecaster = None
        self.data = None

    def load_data(self, start_date, end_date):
        data = self.loader.load_sku_history(start_date, end_date)
        self.data = self.loader.preprocess_data(data)
        self.forecaster = SKUForecaster(self.data)
        self.forecaster.train_models()
        self.forecaster.make_forecasts()

    def plot_sku_forecast(self, sku):
        forecast = self.forecaster.get_forecast(sku)
        history = self.forecaster.prepare_data(sku)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history['ds'], y=history['y'], mode='markers', name='Actual', marker=dict(color='red', size=4)))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(color='green', dash='dash')))

        fig.update_layout(title=f'Forecast for SKU: {sku}', xaxis_title='Date', yaxis_title='Quantity')
        return fig

    def plot_anomalies(self, sku):
        anomalies = self.forecaster.detect_anomalies(sku)
        forecast = self.forecaster.get_forecast(sku)
        history = self.forecaster.prepare_data(sku)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history['ds'], y=history['y'], mode='markers', name='Actual', marker=dict(color='blue', size=4)))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=anomalies['ds'], y=anomalies['y'], mode='markers', name='Anomalies', marker=dict(color='green', size=8, symbol='star')))

        fig.update_layout(title=f'Anomalies for SKU: {sku}', xaxis_title='Date', yaxis_title='Quantity')
        return fig

    def plot_scenario(self, sku, scenario_func):
        original_forecast = self.forecaster.get_forecast(sku)
        scenario_forecast = self.forecaster.simulate_scenario(sku, scenario_func)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=original_forecast['ds'], y=original_forecast['yhat'], mode='lines', name='Original Forecast'))
        fig.add_trace(go.Scatter(x=scenario_forecast['ds'], y=scenario_forecast['yhat'], mode='lines', name='Scenario Forecast'))

        fig.update_layout(title=f'Scenario Forecast for SKU: {sku}', xaxis_title='Date', yaxis_title='Quantity')
        return fig

def main():
    st.set_page_config(page_title="Inventory Forecast Dashboard", layout="wide")
    st.title("Inventory Forecast Dashboard")

    dashboard = Dashboard()

    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = col2.date_input("End Date", pd.to_datetime("2023-12-31"))

    if st.button("Load Data and Train Models"):
        with st.spinner("Loading data and training models..."):
            dashboard.load_data(start_date, end_date)
        st.success("Data loaded and models trained!")

    if dashboard.forecaster:
        sku_list = dashboard.data.columns.tolist()
        selected_sku = st.selectbox("Select SKU for analysis", sku_list)

        tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Anomalies", "Cross-Validation", "Scenario"])

        with tab1:
            st.plotly_chart(dashboard.plot_sku_forecast(selected_sku), use_container_width=True)

        with tab2:
            st.plotly_chart(dashboard.plot_anomalies(selected_sku), use_container_width=True)

        with tab3:
            if st.button("Perform Cross-Validation"):
                with st.spinner("Performing cross-validation..."):
                    cv_metrics = dashboard.forecaster.perform_cross_validation(selected_sku)
                st.write(cv_metrics)

        with tab4:
            trend_increase = st.slider("Trend Increase (% per day)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
            
            def increase_trend(future_df):
                future_df['trend_increase'] = (future_df['ds'] - future_df['ds'].min()).dt.days * trend_increase / 100
                return future_df
            
            st.plotly_chart(dashboard.plot_scenario(selected_sku, increase_trend), use_container_width=True)

if __name__ == "__main__":
    main()