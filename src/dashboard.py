import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from data_loader import DataLoader
from predict import make_prediction
from datetime import datetime, timedelta

@st.cache
def load_data():
    loader = DataLoader()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    raw_data = loader.load_order_history(start_date, end_date)
    return loader.preprocess_data(raw_data)

st.title('Inventory Forecast Dashboard')

data = load_data()

sku = st.selectbox('Select SKU', data.columns)

if st.button('Generate Forecast'):
    with st.spinner('Generating forecast...'):
        prediction = make_prediction(sku)
    
    sku_data = data[sku].dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sku_data.index, y=sku_data.values, name='Historical Data'))
    fig.add_trace(go.Scatter(x=pd.date_range(start=sku_data.index[-1], periods=31, freq='D')[1:], 
                             y=prediction, name='Prediction'))
    
    fig.update_layout(title=f'Sales History and Prediction for SKU: {sku}',
                      xaxis_title='Date',
                      yaxis_title='Quantity')
    
    st.plotly_chart(fig)

st.write('This dashboard uses a LSTM neural network to forecast inventory based on historical sales data.')
