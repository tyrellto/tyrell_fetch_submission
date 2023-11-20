# Made by: Tyrell To
# Date: 11-20-2023
# Summary: This is the main file for the Streamlit app. 
#          It loads the pre-trained tensorflow model and makes forecasts depending on the user input.
#          The forecasted values are then plotted and displayed in the app.

import tensorflow as tf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

from pandas.tseries.offsets import Day
from datetime import datetime, timedelta

# Function to fill in the missing values in each aspect
def fill_nan(arr):
    # Forward fill
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = arr[idx]

    # Backward fill for the remaining NaNs
    mask = np.isnan(out)
    idx = np.where(~mask, np.arange(mask.shape[0]), mask.shape[0] - 1)
    idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
    out = out[idx]

    return out

# Function to create the dataset with look back
def create_dataset(dataset, look_back=1):
    
    # Initialize the lists
    dataX, dataY = [], []
    
    # Create the dataset by appending the previous 'look_back' values to the input array
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    
    return np.array(dataX), np.array(dataY)


# Load the dataset
# dataframe = pd.read_csv('data_daily.csv', usecols=[1], engine='python')
dataframe = pd.read_csv('data_daily.csv', engine='python')

dates = dataframe['# Date'].values
dataset = dataframe['Receipt_Count'].copy()

# Convert the dataframe to numpy array and scale the values by a factor of 1e5
# 1e5 was shown to be the optimal scaling factor for the model through various trials
dataset = dataset.astype('float32')
dataset = dataset.values
dataset = dataset/1e5

# Decompose the time series into trend, seasonality, and residuals
decomposition = sm.tsa.seasonal_decompose(dataset, model='additive', period = 12)

# Extract the trend, seasonality, and residuals
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Fill in the missing values in each aspect
trend = fill_nan(trend)
seasonal = fill_nan(seasonal)
residual = fill_nan(residual)

# Reshape each aspect to be a 2D array
trend = trend.reshape(-1,1)
seasonal = seasonal.reshape(-1,1)
residual = residual.reshape(-1,1)


# Instantiate the look back period (days)
look_back = 90

# Define inputs for each aspect
input_dataset = Input(shape=(look_back,))
input_residuals = Input(shape=(look_back,))
input_trend = Input(shape=(look_back,))
input_seasonal = Input(shape=(look_back,))

# Subnetwork for original dataset
net_dataset = Dense(12, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(input_dataset)
net_dataset = Dense(8, activation='swish', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(net_dataset)

# Subnetwork for residuals
net_residuals = Dense(12, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(input_residuals)
net_residuals = Dense(8, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(net_residuals)
net_residuals = Dense(8, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(net_residuals)

# Subnetwork for trend
net_trend = Dense(12, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(input_trend)
net_trend = Dense(8, activation='swish', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(net_trend)

# Subnetwork for seasonal
net_seasonal = Dense(12, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(input_seasonal)
net_seasonal = Dense(8, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(net_seasonal)
net_seasonal = Dense(8, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(net_seasonal)

# Combine all subnetworks
combined = concatenate([net_dataset, net_residuals, net_trend, net_seasonal])

# Separate output layers for each prediction target
output_dataset = Dense(1, name='output_dataset')(combined)
output_residuals = Dense(1, name='output_residuals')(combined)
output_trend = Dense(1, name='output_trend')(combined)
output_seasonal = Dense(1, name='output_seasonal')(combined)

# Create multi-output model
model = Model(inputs=[input_dataset, input_residuals, input_trend, input_seasonal], 
              outputs=[output_dataset, output_residuals, output_trend, output_seasonal])

# Load in pre-trained model weights
model.load_weights('ty_custom_model.h5')


# Streamlit app starts here
st.title('Time Series Forecasting App')

# A calendar to select the start date and end date, and then calculate the number of days between the two dates
start_date = st.date_input('Enter the start date:', 
                           value=datetime(2022, 1, 1), 
                           min_value=datetime(2022, 1, 1), 
                           max_value=datetime(2022, 12, 31))

end_date = st.date_input('Enter the end date:', 
                         value=datetime(2022, 12, 31), 
                         min_value=datetime(2022, 1, 2), 
                         max_value=datetime(2022, 12, 31))

# Calculate the number of days between the start date and end date
num_future_steps = (end_date - start_date).days + 1

# Button to trigger the forecast
if st.button('Predict'):
    
    # Extract the last 'look_back' values from each aspect
    last_values_dataset = dataset[-look_back:].reshape(1, look_back)
    last_values_residuals = residual[-look_back:].reshape(1, look_back)
    last_values_trend = trend[-look_back:].reshape(1, look_back)
    last_values_seasonal = seasonal[-look_back:].reshape(1, look_back)

    # Create empty lists to store the forecasted values for each aspect
    forecasted_dataset = []
    forecasted_residuals = []
    forecasted_trend = []
    forecasted_seasonal = []

    # Loop to predict the future values
    for _ in range(num_future_steps):
        
        # Predict the next set of values
        predictions = model.predict([last_values_dataset, last_values_residuals, last_values_trend, last_values_seasonal])
        predicted_dataset, predicted_residuals, predicted_trend, predicted_seasonal = predictions

        # Append the predicted values to the forecast lists
        forecasted_dataset.append(predicted_dataset[0, 0])
        forecasted_residuals.append(predicted_residuals[0, 0])
        forecasted_trend.append(predicted_trend[0, 0])
        forecasted_seasonal.append(predicted_seasonal[0, 0])

        # Update the last_values arrays with the newly predicted values
        # For each aspect, we roll the array to remove the first (oldest) value and append the new prediction
        last_values_dataset = np.roll(last_values_dataset, -1)
        last_values_dataset[0, -1] = predicted_dataset

        last_values_residuals = np.roll(last_values_residuals, -1)
        last_values_residuals[0, -1] = predicted_residuals

        last_values_trend = np.roll(last_values_trend, -1)
        last_values_trend[0, -1] = predicted_trend

        last_values_seasonal = np.roll(last_values_seasonal, -1)
        last_values_seasonal[0, -1] = predicted_seasonal

    # Convert the forecast lists to numpy arrays
    forecasted_dataset = np.array(forecasted_dataset).reshape(-1, 1)
    forecasted_residuals = np.array(forecasted_residuals).reshape(-1, 1)
    forecasted_trend = np.array(forecasted_trend).reshape(-1, 1)
    forecasted_seasonal = np.array(forecasted_seasonal).reshape(-1, 1)
    
    # Combine the forecasted values additively
    combined_forecast = forecasted_trend + forecasted_residuals + forecasted_seasonal
    combined_forecast = combined_forecast.flatten()/10 # scale to millions
    original_dataset = dataset.flatten()/10 # scale to millions
    
    # Access the last date in the original dataset
    last_date = dates[-1]
    date_object = datetime.strptime(last_date, '%Y-%m-%d')
    
    # Create a list of dates from the last date in the original dataset to the end date
    date_list = [date_object + timedelta(days=x) for x in range(num_future_steps+1)]
    date_list.pop(0)
    additional_dates = np.array(date_list)
    
    # Create a plotly figure to plot the original time series and the forecasted values
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=additional_dates, y=combined_forecast, name='Forecasted Receipt Amounts'))

    fig.add_trace(go.Scatter(x=dates, y=original_dataset, name='Prior Receipt Amounts'))

    fig.update_layout(title='Original Time Series and Forecast',
                    xaxis_title='Date',
                    yaxis_title='Receipt Amount (Millions)',
                    legend_title='Legend',
    )
    
    # Add the figure to the Streamlit app
    st.plotly_chart(fig)
    
    # Create a date range from the start date to the end date
    date_range = pd.date_range(start=date_list[0], end=date_list[-1], freq='D')
    
    # Initialize the counters
    start_counter = 0
    end_counter = 0

    # Initialize the list to store the start and end dates
    start_end_dates = []
    month_year = []
    
    # Iterate through the date range to mark the occurrence for the start and end of each month
    for i, date in enumerate(date_range):
        if date.is_month_start:
            start_counter += 1
            start_end_dates.append(i)
        if date.is_month_end:
            end_counter += 1
            start_end_dates.append(i)

    if end_counter == 0:
        # Show error message if there are no end dates
        st.error('There are no end dates for the forecasted period. Please enter a longer forecast period.')
    else:
        # Show success message if there are end dates
        st.success('There are monthly end dates for the forecasted period.')
                
        # Remove the last start date if there are more start dates than end dates
        if start_counter != end_counter:
            start_end_dates.pop()

        # Create a list to store the monthly forecasted values
        monthly_forecast = []
        for i in range(0, len(start_end_dates), 2):
            start = start_end_dates[i]
            end = start_end_dates[i+1]
            monthly_forecast.append(np.sum(combined_forecast[start:end+1]))

        # Create a list to store the month and year for each month
        month_year = []
        counter = 0
        for i, date in enumerate(date_range):
            if date.is_month_start and counter < len(monthly_forecast):
                month_year.append(date.strftime('%b %Y'))
                counter = counter + 1
        
        # Display a histogram of the monthly forecasted values
        fig2 = px.histogram(x=month_year, y=monthly_forecast, title='Monthly Forecasted Receipt Amounts')
        
        fig2.update_layout(title='Forecasted Receipt Amounts by Month',
                        xaxis_title='Month',
                        yaxis_title='Receipt Amount (Millions)',
                        legend_title='Legend',
        )
        
        st.plotly_chart(fig2)
        
        # Show the total receipt amount for each month with month_year as the index
        monthly_forecast_df = pd.DataFrame(monthly_forecast, index=month_year, columns=['Total Receipt Amount (Millions)'])
        st.write('The total receipt amount (Millions) for each month is: ', monthly_forecast_df)