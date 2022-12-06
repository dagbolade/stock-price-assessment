import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import load_model
import pandas_datareader as data
import streamlit as st
import tensorflow



#creating a start date and end date
start_input = st.text_input('Enter the start date of the stock you want to check in a format like this', '2015-01-01')
end =  datetime.today().strftime("%Y-%m-%d")
#end = '2019-12-31'

st.title('Solfintech Stock price prediction Trend')

user_input = st.text_input('Enter the stock ticker you wish to check', 'AAPL')
#creating the dataframe
df= data.DataReader(user_input, 'yahoo', start_input, end)
df.head()
df.shape

#describe the data
st.subheader('Data from', start_input)
st.write(df.describe())

#VISUALIZING THE Data
st.subheader('closing price(in usd) vs year chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

#VISUALIZING THE Data with the moving average
st.subheader('closing price(in usd) vs year chart with 50MA')
ma50 = df.Close.rolling(50).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma50,'r')
plt.plot(df.Close)
st.pyplot(fig)

#VISUALIZING THE Data with the moving average
st.subheader('closing price(in usd) vs year chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

#VISUALIZING THE Data with the moving average
st.subheader('closing price(in usd) vs year chart with 100MA & 200MA')
ma50 = df.Close.rolling(50).mean()
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(ma50)
plt.plot(df.Close)
st.pyplot(fig)