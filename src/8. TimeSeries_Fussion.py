import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Apply the imports first
# Linear algebra
import numpy as np 

# Data processing, CSV file I/O (e.g., pd.read_csv)
import pandas as pd 

# Data visualization
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns # Data visualization
import plotly.express as px # Data visualization
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, norm # Chi-square calculation

# Stocks-related missing information
import yfinance as yf

# Ignoring the warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Ranking the stocks
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import optuna
from wordcloud import WordCloud, STOPWORDS # For word cloud generation

# Evaluate if they should be removed
import statsmodels.api as sm

# Time series
import datetime

# Uncomment to install Prophet if not already installed
#!pip install fbprophet --quiet
import plotly.offline as py
py.init_notebook_mode()

# Save model
import pickle

# Apply the "from" statements after the imports

# Evaluate if they should be removed
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib.pyplot import figure
# Ranking the stocks
from plotly.subplots import make_subplots

# Prophet Model Stuff
# Uncomment to install Prophet if not already installed
#!pip install fbprophet --quiet

from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import plot_yearly
from prophet.plot import add_changepoints_to_plot


# Get names of indexes for which column Stock has value No
from pickle import FALSE


# connect with my sql %pip install mysql-connector-python pandas
import mysql.connector

# Load the CSV file into a DataFrame
df_pagos = pd.read_csv("C:\Users\perez\Documents\Tesis_USC\ProjectoFinal_IA_USC\src\pagos_moneda_filtro_campos2024.csv", engine="python", sep=',', quotechar='"', error_bad_lines=False)

# Filter out rows where pagos_pk is null
df_pagos = df_pagos[df_pagos['pag_pk'].notna()]
#Eliminan las filas de esos registros
indexNames = df_pagos[df_pagos['pag_confirmar'].isnull()].index
# Delete these row indexes from dataFrame

df_pagos.drop(indexNames,inplace=True)
df_pagos['pag_confirmar'].isnull().sum()
#Eliminan las filas de esos registros de los no confirmados.
indexNames = df_pagos[df_pagos['pag_confirmar']==0].index
# Delete these row indexes from dataFrame
df_pagos.drop(indexNames,inplace=True)
# Calcular el total de registros en df_pagos
total_count = df_pagos['mon_nombre'].count()

# Calcular el conteo de cada moneda
currency_counts = df_pagos['mon_nombre'].value_counts()

# Calcular el porcentaje de cada moneda
currency_percentages = (currency_counts / total_count) * 100

# Crear un DataFrame con los resultados
result_currency = pd.DataFrame({
    'Currency': currency_counts.index,
    'Count': currency_counts.values,
    'Percentage (%)': currency_percentages.values
})

# Filtrar para mostrar solo las monedas relevantes
filtered_result_currency = result_currency[result_currency['Currency'].isin(['Pesos', 'Dlares', 'Euros', 'Unidades Indexadas', 'Unidad Reajustable'])]

# Mostrar la tabla resultante
filtered_result_currency
# Updated data with correct values
data = {
    'Currency': ['Pesos', 'Dólares', 'Unidades Indexadas', 'Euros', 'Unidad Reajustable'],
    'Count': [15337, 7999, 315, 7, 1],
    'Percentage': [64.83, 33.81, 1.33, 0.03, 0.00]
}

# Creating a bar plot
plt.figure(figsize=(10, 6))
plt.barh(data['Currency'], data['Percentage'], color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#673AB7'])

# Adding labels and title
plt.xlabel('Percentage (%)')
plt.ylabel('Currency')
plt.title('Percentage Distribution of Different Currencies')
plt.xlim(0, 70)

# Adding data labels to each bar
for index, value in enumerate(data['Percentage']):
    plt.text(value + 0.2, index, f'{value:.2f}%', va='center', ha='left', fontsize=12)

# Display the plot
plt.show()
# Ensure the column is in datetime format
df_pagos['pag_fecha_real'] = pd.to_datetime(df_pagos['pag_fecha_real'])

# Calculate the minimum and maximum dates
min_date = df_pagos['pag_fecha_real'].min()
max_date = df_pagos['pag_fecha_real'].max()

# Calculate the date range
date_range = max_date - min_date

# Display the results
print(f"Minimum Date: {min_date}")
print(f"Maximum Date: {max_date}")
print(f"Date Range (Duration): {date_range}")
# Exchange rate table
exchange_rate = {
    2024: 40.03,
    2023: 38.84,
    2022: 41.13,
    2021: 43.56,
    2020: 41.99,
    2019: 35.22,
    2018: 30.71,
    2017: 28.66,
    2016: 30.14,
    2015: 27.31
}

# Ensure the 'pag_fecha_real' column is in datetime format
df_pagos['pag_fecha_real'] = pd.to_datetime(df_pagos['pag_fecha_real'])

# Function to calculate the converted amount
def convert_to_dollars(row, column_name):
    if row['mon_pk'] == 1:  # Check if currency is pesos
        year = row['pag_fecha_real'].year
        if year in exchange_rate:  # Ensure year exists in the exchange rate table
            tc = exchange_rate[year]  # Get the exchange rate for the year
            return row[column_name] / tc  # Convert pesos to dollars
        else:
            return row[column_name]  # Default to no conversion if year not found
    else:
        return row[column_name]  # If not pesos, keep the same value

# Apply the conversion logic to 'pag_importe_real' and 'pag_importe_planificado'
df_pagos['pag_importe_real_convert'] = df_pagos.apply(lambda row: convert_to_dollars(row, 'pag_importe_real'), axis=1)
df_pagos['pag_importe_planificado_convert'] = df_pagos.apply(lambda row: convert_to_dollars(row, 'pag_importe_planificado'), axis=1)

# Display the resulting DataFrame
print(df_pagos)
# Calculate delay between planned and real payment dates
df_pagos['delay_days'] = (pd.to_datetime(df_pagos['pag_fecha_real']) - pd.to_datetime(df_pagos['pag_fecha_planificada'])).dt.days

# Plot the distribution of delays
plt.figure(figsize=(12, 6))
sns.histplot(df_pagos['delay_days'], bins=50, kde=True, color='blue')
plt.title('Distribution of Payment Delays')
plt.xlabel('Delay (Days)')
plt.ylabel('Frequency')
plt.show()
#Cambio de los tipo objetos a tipo fecha 
df_pagos['pag_fecha_planificada'] = pd.to_datetime(df_pagos['pag_fecha_planificada'])
df_pagos['pag_fecha_real'] = pd.to_datetime(df_pagos['pag_fecha_real'])

df_pagos.head()
df_pagos_DP=df_pagos
dfi=df_pagos.set_index('pag_fecha_real', drop=True, append=False, inplace=False, verify_integrity=False)
dfi
pesos =dfi.query("mon_nombre == 'Pesos'")
dolares = dfi.query("mon_nombre == 'Dlares'")
euros = dfi.query("mon_nombre == 'Euro'")
df_pagos=dfi.reset_index()
# Keep only rows where 'mon_pk' is 1 or 2
df_pagos = df_pagos[df_pagos['mon_pk'].isin([1, 2])]

# Reset the index after filtering
df_pagos.reset_index(drop=True, inplace=True)

# Display the updated DataFrame
print(df_pagos)
pagos_modelo_pesos=df_pagos.drop(['pag_pk','pag_fecha_planificada','pag_importe_planificado','pag_confirmar','mon_pk','mon_nombre','pag_importe_real','pag_importe_planificado_convert','delay_days'],axis=1)
pagos_modelo_pesos.shape
pagos_modelo_pesos.sort_values(['pag_fecha_real', 'pag_importe_real_convert'],ascending=False) 
pagos_modelo_pesos
negative_values = pagos_modelo_pesos[pagos_modelo_pesos['pag_importe_real_convert'] < 0]
print(negative_values)
print(f"Total negative values: {negative_values.shape[0]}")
sp = pagos_modelo_pesos.rename(columns={'pag_fecha_real': 'ds','pag_importe_real_convert': 'y'})
sp_sample = sp[(sp.ds.dt.year>2014)]
# Load the data
sp_sample['ds'] = pd.to_datetime(sp_sample['ds'])

# Normalize the 'y' column
scaler = MinMaxScaler()
sp_sample['y'] = scaler.fit_transform(sp_sample[['y']])

# Set 'cap' and 'floor' for logistic growth
cap_value = sp_sample['y'].quantile(0.995)
floor_value = sp_sample['y'].quantile(0.005)

sp_sample['cap'] = cap_value
sp_sample['floor'] = floor_value

# Split the data into training and testing sets
train_size = int(len(sp_sample) * 0.8)
data_train = sp_sample.iloc[:train_size].copy()
data_test = sp_sample.iloc[train_size:].copy()

# Train the Prophet model with the best parameters
model = Prophet(
    growth='logistic',
    seasonality_prior_scale=1.0,
    changepoint_prior_scale=0.5,
    holidays_prior_scale=1.0,
    interval_width=0.95,
    weekly_seasonality=False
)
model.add_seasonality(name='yearly', period=365, fourier_order=8)
model.add_country_holidays(country_name='UY')

# Fit the model
model.fit(data_train)

# Perform cross-validation
df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
df_p = performance_metrics(df_cv)

# Reverse normalization for interpretation
df_cv[['yhat', 'yhat_lower', 'yhat_upper', 'y']] = scaler.inverse_transform(df_cv[['yhat', 'yhat_lower', 'yhat_upper', 'y']])
data_train['y'] = scaler.inverse_transform(data_train[['y']])
data_test['y'] = scaler.inverse_transform(data_test[['y']])

# Compute validation metrics
mae_cv = df_p['mae'].mean()
rmse_cv = df_p['rmse'].mean()

# Create a DataFrame for future predictions
future = model.make_future_dataframe(periods=len(data_test), freq='D')
future['cap'] = cap_value
future['floor'] = floor_value

# Make predictions
forecast = model.predict(future)

# Reverse normalization for interpretation
forecast[['yhat', 'yhat_lower', 'yhat_upper']] = scaler.inverse_transform(forecast[['yhat', 'yhat_lower', 'yhat_upper']])

# Custom Forecast Plot Function
def custom_forecast_plot(model, forecast, actual_data=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(forecast['ds'], forecast['yhat'], label="Predicted", color='blue')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2, label="Uncertainty Interval")
    if actual_data is not None:
        ax.scatter(actual_data['ds'], actual_data['y'], color='black', label="Actual Data", alpha=0.6)
    ax.axvspan(forecast['ds'].iloc[-len(data_test)], forecast['ds'].iloc[-1], color='orange', alpha=0.3, label="Test Period")
    add_changepoints_to_plot(ax, model, forecast)
    ax.set_title('Custom Prophet Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    plt.show()

# Call the custom forecast plot function
custom_forecast_plot(model, forecast, actual_data=data_test)

# Plot forecast components
fig2 = model.plot_components(forecast)
plt.show()

# Align and calculate errors on the test set
forecast_aligned = forecast.set_index('ds').join(data_test.set_index('ds'), how='inner', lsuffix='_forecast', rsuffix='_test')
forecast_aligned = forecast_aligned.dropna()
forecast_aligned['error'] = forecast_aligned['y'] - forecast_aligned['yhat']

# Compute test set metrics
coverage = ((forecast_aligned['y'] >= forecast_aligned['yhat_lower']) & 
            (forecast_aligned['y'] <= forecast_aligned['yhat_upper'])).mean()
mae_test = mean_absolute_error(forecast_aligned['y'], forecast_aligned['yhat'])
rmse_test = np.sqrt(mean_squared_error(forecast_aligned['y'], forecast_aligned['yhat']))

