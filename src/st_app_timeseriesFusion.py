import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- Title and Description ----
st.title('📈 Machine Learning & Analytics for Project Portfolio')
"""
The main objective is to create a comprehensive dashboard for the Strategic Planning area, specifically for the AGESIC portfolios, to have indicators available for decision-making.
In this case, we have developed a time series model that aims to predict the future budget over a period of time, both in pesos and in dollars.
"""
st.image('https://www.springboard.com/blog/wp-content/uploads/2022/02/data-scientist-without-a-degree-2048x1366.jpg')
st.sidebar.image('https://www.gub.uy/agencia-gobierno-electronico-sociedad-informacion-conocimiento/sites/agencia-gobierno-electronico-sociedad-informacion-conocimiento/files/catalogo/iso.png')
st.sidebar.subheader('What is Prophet?')
st.sidebar.write('Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and generally handles outliers well. With this tool, the Facebook data science team aimed to achieve quick and accurate forecasting models and obtain reasonably accurate forecasts automatically.')

# Updated data with correct values
data = {
    'Currency': ['Pesos', 'Dollars', 'Indexed Units', 'Euros', 'Adjustable Units'],
    'Count': [15337, 7999, 315, 7, 1],
    'Percentage': [64.83, 33.81, 1.33, 0.03, 0.00]
}

# Creating a bar plot
plt.figure(figsize=(10, 6))
bars = plt.barh(data['Currency'], data['Percentage'], color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#673AB7'])

# Adding labels and title
plt.xlabel('Percentage (%)')
plt.ylabel('Currency')
plt.title('Percentage Distribution of Different Currencies')
plt.xlim(0, 70)

# Adding data labels to each bar
for bar in bars:
    plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}%', va='center', ha='left', fontsize=12)

# Display the plot in Streamlit
st.pyplot(plt)

# ---- Load Data ----
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['pag_fecha_real', 'pag_fecha_planificada'])
    df = df[df['pag_pk'].notna()]
    df.dropna(subset=['pag_confirmar', 'pag_fecha_real', 'pag_importe_real'], inplace=True)
    df = df[df['pag_confirmar'] != 0]
    df.reset_index(drop=True, inplace=True)

    # Currency Conversion
    exchange_rate = {2024: 40.03, 2023: 38.84, 2022: 41.13, 2021: 43.56, 2020: 41.99, 2019: 35.22, 2018: 30.71, 2017: 28.66, 2016: 30.14, 2015: 27.31}
    
    def convert_to_dollars(row):
        return row['pag_importe_real'] / exchange_rate.get(row['pag_fecha_real'].year, 1) if row['mon_pk'] == 1 else row['pag_importe_real']
    
    df['pag_importe_real_convert'] = df.apply(convert_to_dollars, axis=1)
    df = df[['pag_fecha_real', 'pag_importe_real_convert']].rename(columns={'pag_fecha_real': 'ds', 'pag_importe_real_convert': 'y'})
    df = df[df['ds'].dt.year > 2014]
    
    return df

df_pagos = load_data("pagos_moneda_filtro_campos2024.csv")

# ---- Prophet Model Training ----
@st.cache_resource
def train_model(df):
    model = Prophet(growth='logistic', seasonality_prior_scale=1.0, changepoint_prior_scale=0.5, holidays_prior_scale=1.0, interval_width=0.95, weekly_seasonality=False)
    train_size = int(len(df) * 0.8)
    data_train = df.iloc[:train_size].copy()
    
    cap_value = data_train['y'].quantile(0.995)
    floor_value = data_train['y'].quantile(0.005)
    
    data_train['cap'] = cap_value
    data_train['floor'] = floor_value
    
    model.fit(data_train)
    return model

model = train_model(df_pagos)

# ---- User Input for Forecast Period ----
st.write("### Select Forecast Horizon")
periods_input = st.number_input('How many days would you like to forecast?', min_value=1, max_value=730, value=30)

future = model.make_future_dataframe(periods=periods_input)
future['cap'] = 0.995
future['floor'] = 0.005
forecast = model.predict(future)

# ---- Button to Calculate MAE and RMSE ----
if st.button('Calculate MAE and RMSE'):
    train_size = int(len(df_pagos) * 0.8)
    data_test = df_pagos.iloc[train_size:].copy()
    forecast_aligned = forecast.set_index('ds').join(data_test.set_index('ds'), how='inner').dropna()
    
    # Select only the last 'periods_input' days for evaluation
    forecast_aligned = forecast_aligned.tail(periods_input)
    
    st.write(f"✅ **Test Set MAE:** {mean_absolute_error(forecast_aligned['y'], forecast_aligned['yhat']):.4f}")
    st.write(f"✅ **MAE represents the average absolute difference between actual and predicted values**")                
    st.write(f"📉 **Test Set RMSE:** {np.sqrt(mean_squared_error(forecast_aligned['y'], forecast_aligned['yhat'])):.4f}")
    st.write(f"📉 **MSE gives a higher penalty to larger errors because it squares the differences.**")
# ---- Graph Selection ----
st.write("### Select a Graph to Display")
graph_option = st.selectbox("Choose a graph:", ["Forecast Trend", "Model Components", "Prediction vs. Actual Data"])

# ---- Graph 1: Forecast Trend ----
if graph_option == "Forecast Trend":
    fig_forecast, ax = plt.subplots(figsize=(10, 6))
    model.plot(forecast, ax=ax)
    ax.axhline(y=1_400_000, color='red', linestyle='dashed', linewidth=2, label="Threshold: 1.4M")
    ax.set_ylabel("Budget Value")
    ax.set_xlabel("Year")
    ax.legend()
    st.pyplot(fig_forecast)

# ---- Graph 2: Model Components ----
elif graph_option == "Model Components":
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)

# ---- Graph 3: Prediction vs. Actual Data ----
elif graph_option == "Prediction vs. Actual Data":
    fig_forecast_plotly = go.Figure()

    # Confidence interval filled with transparency
    fig_forecast_plotly.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        mode='lines', line=dict(color='rgba(0,0,255,0.3)'), name='Upper Bound', showlegend=False
    ))
    fig_forecast_plotly.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        mode='lines', fill='tonexty', line=dict(color='rgba(0,0,255,0.3)'), name='Lower Bound'
    ))

    # Predicted values
    fig_forecast_plotly.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='Prediction', line=dict(color='blue', width=3)
    ))

    # Actual data
    fig_forecast_plotly.add_trace(go.Scatter(
        x=df_pagos['ds'], y=df_pagos['y'],
        mode='markers', name='Actual Data', marker=dict(color='black', size=5, opacity=0.8)
    ))

    fig_forecast_plotly.add_shape(
        type="line", x0=min(forecast['ds']), x1=max(forecast['ds']),
        y0=1_400_000, y1=1_400_000, line=dict(color="red", width=2, dash="dash"), name="Threshold 1.4M"
    )

    st.plotly_chart(fig_forecast_plotly, use_container_width=True)

st.download_button(label="📥 Download Forecast Data", data=forecast.to_csv(index=False), file_name="forecast_results.csv", mime="text/csv")
st.write('📩 **For more information, contact us at:** canarysoftware@gmail.com')














