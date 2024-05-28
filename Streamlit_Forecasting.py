import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
from prophet.plot import plot_plotly
import pickle

# Load data
df = pd.read_csv('EstateSolarWeather.csv')

# Drop rows where both 'Expected Value kWh' and 'PR %' are 0
df = df[~((df['Expected Value kWh'] == 0) & (df['PR %'] == 0))]

# Ensure df is a copy if it's a slice
df = df.copy()

# Safely create a new column
df.loc[:, 'Energy-kWh'] = df['Expected Value kWh'] * df['PR %'] / 100

# Ensure 'Date and Time' is of datetime type
df['Date and Time'] = pd.to_datetime(df['Date and Time'])

# Extract year and month name
df['Year'] = df['Date and Time'].dt.year
df['Month'] = df['Date and Time'].dt.strftime('%B')  # Get the month name

# Group by year and month name, then aggregate (sum) 'Energy-kWh'
monthly_data = df.groupby(['Year', 'Month'], sort=False)['Energy-kWh'].sum().reset_index()

# Pivot the table to make it suitable for bar plotting
pivot_table = monthly_data.pivot(index='Year', columns='Month', values='Energy-kWh')

# Ensure the months are in the correct order
months_order = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]
pivot_table = pivot_table[months_order]

# Reset index for plotting
pivot_table = pivot_table.reset_index()

# Ensure 'Date and Time' is of datetime type
df['Date and Time'] = pd.to_datetime(df['Date and Time'])

# Group by 'Date and Time' and sum 'Energy-kWh'
df = df.groupby('Date and Time')['Energy-kWh'].sum().reset_index()

# Prepare the data for Prophet
df.rename(columns={'Date and Time': 'ds', 'Energy-kWh': 'y'}, inplace=True)

# Fit the Prophet model
m = Prophet()
m.fit(df)

# Create future dataframe
future = m.make_future_dataframe(periods=365)

# Predict
forecast = m.predict(future)

# Save the forecast to a pickle file
with open('solar_forecast_prophet.pkl', 'wb') as f:
    pickle.dump(forecast, f)

# Create an input button on the Streamlit website to ask for input on how many days in advance the user wants to know the total energy produced
st.markdown("## Enter number of days in advance to forecast total energy produced:")
days_ahead = st.number_input('', min_value=1, max_value=365, value=30)

# Calculate total predicted energy produced based on the number of days the user selected
start_date_input = df['ds'].max() + pd.Timedelta(days=1)
end_date_input = start_date_input + pd.Timedelta(days=days_ahead - 1)
total_energy_predicted = forecast.loc[(forecast['ds'] >= start_date_input) & (forecast['ds'] <= end_date_input), 'yhat'].sum()

# Calculate the start date for the line chart's slider (previous 1/4 amount of the input)
start_date_slider = start_date_input - pd.Timedelta(days=days_ahead * 0.25)
# Ensure that start_date_slider is not before the minimum date in the dataset
start_date_slider = max(start_date_slider, df['ds'].min())

# Plot the forecast with Plotly
line_fig = plot_plotly(m, forecast)

# Update the layout to add a title and set the range for the slider
line_fig.update_layout(title='Energy Forecasts: Adjust the Slider Below to Select the Timeframe.',
                       xaxis=dict(rangeslider=dict(visible=True), range=[start_date_slider, end_date_input]))

# Show the plots in Streamlit
st.plotly_chart(line_fig)
