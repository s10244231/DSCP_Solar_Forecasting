import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import joblib
import os

# Set the Streamlit page configuration
st.set_page_config(page_title="Energy Forecasting", layout="wide")

# Title of the Streamlit app
st.title("Energy Forecasting Visualization")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Drop rows where both 'Expected Value kWh' and 'PR %' are 0
    df = df[~((df['Expected Value kWh'] == 0) & (df['PR %'] == 0))]

    # Ensure df is a copy if it's a slice
    df = df.copy()

    # Safely create a new column
    df['Energy-kWh'] = df['Expected Value kWh'] * df['PR %'] / 100

    # Plot initial data
    st.subheader("Energy-kWh Over Time")
    fig, ax = plt.subplots(figsize=(18, 6))
    df.plot(x='Date and Time', y='Energy-kWh', ax=ax)
    st.pyplot(fig)

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

    # Plot monthly data
    st.subheader("Monthly Energy Consumption by Year")
    fig, ax = plt.subplots(figsize=(18, 6))
    pivot_table.plot(kind='bar', ax=ax)
    ax.set_title('Monthly Energy Consumption by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Energy-kWh')
    ax.legend(title='Month')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # Keep x-axis labels horizontal for better readability
    st.pyplot(fig)

    df = df.drop(columns=['Year', 'Month'])

    # Ensure 'Date and Time' is of datetime type
    df['Date and Time'] = pd.to_datetime(df['Date and Time'])

    # Group by 'Date and Time' and sum 'Energy-kWh'
    df = df.groupby('Date and Time')['Energy-kWh'].sum().reset_index()

    # Prepare the data for Prophet
    df.rename(columns={'Date and Time': 'ds', 'Energy-kWh': 'y'}, inplace=True)

    # Define model filename
    model_filename = 'solar_forecast_prophet.pkl'
    
    # Check if the model file exists
    if os.path.exists(model_filename):
        # Load the existing model
        with open(model_filename, 'rb') as f:
            m = joblib.load(f)
        st.write("Model loaded from disk.")
    else:
        # Fit a new Prophet model
        m = Prophet()
        m.fit(df)
        # Save the model to disk
        with open(model_filename, 'wb') as f:
            joblib.dump(m, f)
        st.write("Model trained and saved to disk.")

    # Create future dataframe
    future = m.make_future_dataframe(periods=365)

    # Make predictions
    forecast = m.predict(future)

    # Plot the forecast
    st.subheader("Forecasted Energy Consumption")
    fig_plotly = plot_plotly(m, forecast)
    st.plotly_chart(fig_plotly)

    # Plot forecast components
    st.subheader("Forecast Components")
    fig_components = plot_components_plotly(m, forecast)
    st.plotly_chart(fig_components)
