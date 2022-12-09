import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import load_model
import pandas_datareader as data
import streamlit as st
import tensorflow
from keras.models import load_model



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
st.subheader('Data Description from '+start_input+' to '+end+'')
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
st.subheader('closing price(in usd) vs year chart with 50ma, 100MA & 200MA')
ma50 = df.Close.rolling(50).mean()
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, label=f"ma100", color="blue")
plt.plot(ma200, label=f"ma200", color="green")
plt.plot(ma50, label=f"ma50", color="red")
plt.plot(df.Close)
st.pyplot(fig)

#Scaling the stock data so that they fit inbetween 0 and 1
from sklearn.preprocessing import MinMaxScaler
from numpy import set_printoptions
scaler = MinMaxScaler(feature_range=(0,1))
train_header_array = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

#train_header_array

lstm_model = load_model('new_lstm_model.h5')
#define how many days in the past we will look into
#the closing price of the particular day would depend on the previous days
historical_data = 50

#########################
# PREPARING TRAINING DATA
#########################
x_train = []
y_train = []

for x in range(historical_data, train_header_array.shape[0]):
    x_train.append(train_header_array[x - historical_data:x, 0])
    y_train.append((train_header_array[x, 0])) #because we are considering only one column

# turning x and y train into numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#%%
# now we are going to reshape the x_train so that it works with the neural network
# the 1 indicates one additional dimension
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#%%


# LOAD THE DATA - it has to be data the model has not seen before
# this is the time range of the data - we have the data but the model has never seen that data
# so this is a perfect way to see how well it performs
_start = start_input
_end = datetime.today().strftime("%Y-%m-%d")
_data = data.DataReader(user_input,  'yahoo', _start, _end)

#getting the actual prices
# we need to scale prices, we need to concatenate the full data set that we want to predict on
# this is NOT the predicted price, but the real price from the market
actual_prices = _data['Close'].values

# this will combine the training data and the test data
total_dataset = pd.concat((df['Close'], _data['Close']), axis=0)

# this is what our model is going to see as an input, so it can predict
model_inputs = total_dataset[len(total_dataset) - len(_data) - historical_data:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

###############################
# MAKE PREDICTIONS ON TEST DATA
###############################

# we don't use the y_test because we already have the stock prices
x_test = []
y_test = []
# if you remove the + 1 it will remove the newest data
for x in range(historical_data, len(model_inputs)):
    x_test.append(model_inputs[x - historical_data:x, 0])
    y_test.append(model_inputs[x, 0])

x_test,y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


#%%
# now let's predict on the x_test data,the predicted prices are now going to be scaled, so we need to reverse scale them or rather inverse_transform them
y_pred = lstm_model.predict(x_test)
# now we are back to the actual predicted price, non-scaled
y_pred = scaler.inverse_transform(y_pred)

# now let plot the predictions instead of just numbers
fig1=plt.figure(figsize=(12,6))
plt.plot(actual_prices, color="red", label=f"Actual {user_input} Price")
plt.plot(y_pred, color="blue", label=f"Predicted {user_input} Price")
plt.title(f"{user_input} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{user_input} Share Price")
plt.legend()
# you can also do .pdf for a vector like file (super zoomy)
plt.savefig('new_Prediction_test.svg')
# this acts as a 'end conn' situation
plt.show()
st.subheader("Actual Prices VS Predicted prices")
st.pyplot(fig1)

#########################
# PREDICTING OUR NEXT DAY
#########################

final_data = [model_inputs[len(model_inputs) + 1 - historical_data:len(model_inputs + 1), 0]]
final_data = np.array(final_data)
final_data = np.reshape(final_data, (final_data.shape[0], final_data.shape[1], 1))

#print(scaler.inverse_transform(final_data[-1]))
header = 'The next day prediction for '+user_input+' is '
st.subheader(header)
Next_day = lstm_model.predict(final_data)
Next_day  = scaler.inverse_transform(Next_day)
#print(f' Price Prediction for tomorrow would be: {Next_day}')

# Create a numpy.float32 object
result = np.float32(Next_day[0][0])

# Convert the numpy.float32 object to a string
x_str = str(result)
st.text('$' + x_str)

dfr = pd.DataFrame(df['Close'].tail(50))
dfr =pd.DataFrame(scaler.fit_transform(dfr))


prev_close=scaler.inverse_transform([[dfr.iloc[-1,0]]])
#st.text(prev_close)
temp=Next_day[0][0]-prev_close[0][0]
if temp > 0 :
    trend='rise'
else :
    trend='fall'
str1=Next_day[0][0]
te='There could be a '+trend+' in the stock price market for '+user_input+''
st.write(te)
