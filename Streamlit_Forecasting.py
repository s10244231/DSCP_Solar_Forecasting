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

# Plot using Plotly
fig = px.line(df, x='Date and Time', y='Energy-kWh', 
              labels={'Energy-kWh': 'Energy (kWh)', 'Date and Time': 'Date and Time'},
              title='Energy Consumption Over Time')

# Update layout for better readability
fig.update_layout(width=1200, height=400)

# Show the plot in Streamlit
st.plotly_chart(fig)

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

# Melt the DataFrame to long format for Plotly
melted_table = pivot_table.melt(id_vars='Year', value_vars=months_order, 
                                var_name='Month', value_name='Energy-kWh')

# Plot using Plotly
fig = px.bar(melted_table, x='Year', y='Energy-kWh', color='Month', 
             category_orders={'Month': months_order},
             labels={'Energy-kWh': 'Energy (kWh)', 'Year': 'Year', 'Month': 'Month'},
             title='This is Your Total Monthly Energy Production by Year')

# Update layout for better readability
fig.update_layout(barmode='group', xaxis_tickangle=-45)

# Show the plot in Streamlit
st.plotly_chart(fig)

# Drop Year and Month columns for Prophet model
df = df.drop(columns=['Year', 'Month'])

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

# Plot the forecast with Plotly
fig = plot_plotly(m, forecast)

# Update the layout to add a title
fig.update_layout(title='Energy Forecasts: Adjust the Slider Below to Select the Timeframe.')

# Show the plot in Streamlit
st.plotly_chart(fig)

# Create an input button on the Streamlit website to ask for input on how many days in advance the user wants to know the total energy produced
days_ahead = st.number_input('Enter number of days in advance to forecast total energy produced:', min_value=1, max_value=365, value=30)

# Calculate total predicted energy produced based on the number of days the user selected
total_energy_predicted = forecast.loc[forecast['ds'] <= (df['ds'].max() + pd.Timedelta(days=days_ahead)), 'yhat'].sum()

# Display the total predicted energy
st.write(f'Total predicted energy produced in the next {days_ahead} days: {total_energy_predicted:.2f} kWh')


    # Plot forecast components
    st.subheader("Forecast Components")
    fig_components = plot_components_plotly(m, forecast)
    st.plotly_chart(fig_components)
