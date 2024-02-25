##Importing necessary libraries
import pandas as pd
from dataprep import *
import datetime
from stockdata import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,mean_squared_error


def Full_model(ticker):

    #Data acquisition
    stock_data=StockData(ticker)
    print("test0")
    stock_dates=stock_data.iloc[:,0]
    stock_data.reset_index(inplace=True)

    fig = go.Figure(data=[go.Candlestick(x=stock_data['Date'],
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close']),])

    fig.update_layout(xaxis_rangeslider_visible=False)
    plt.savefig('./main/web/static/tickerpic.png')

    
    X_train,X_test,y_train,y_test=train_test_split(stock_data.drop(columns=['Date','Close']),stock_data['Close'],test_size=0.08,shuffle=False)
    
    min_max = MinMaxScaler(feature_range=(0, 1))
    close_price=MinMaxScaler(feature_range=(0,1))

    X_train_scaled=min_max.fit_transform(X_train)
    X_test_scaled=min_max.transform(X_test)
    close_price.fit(stock_data['Close'].values.reshape(-1,1))
    y_train_scaled=close_price.transform(y_train.values.reshape(-1,1))
    y_test_scaled=close_price.transform(y_test.values.reshape(-1,1))
    v=X_test_scaled.shape[0]

    print("test1")

    # Define the number of time steps
    time_steps = 60

    X_train_copy=X_train.copy()
    y_train_copy=y_train.copy()

    X_train=[]
    y_train=[]
    X_test=[]

    # Loop through the data to create partitions
    for i in range(time_steps, X_train_scaled.shape[0]):
        # Create a partition of the previous 60 days' data
        X_train.append(X_train_scaled[i - time_steps:i,:])

        # Append the next day's Close price to the label array
    X_train, y_train = np.array(X_train), np.array(y_train)
    for i in range(time_steps, X_train_scaled.shape[0]-15):
        y_train=np.insert(y_train,i-time_steps,y_train_scaled[i+15])
    x=y_test_scaled[:16]
    for i in range(0,15):
        y_train=np.insert(y_train,i,x[i])
    X_test_scaled=X_test_scaled.tolist()
    a= X_train_scaled[-60:].tolist()
    X_test_scaled= a + X_test_scaled
    X_test_scaled=np.array(X_test_scaled)

    # Loop through the data to create partitions
    for i in range(time_steps, X_test_scaled.shape[0]):
        # Create a partition of the previous 60 days' data
        X_test.append(X_test_scaled[i - time_steps:i,:])

        # Append the next day's Close price to the label array

    X_test = np.array(X_test)
    X_train =np.array(X_train)
    y_train= np.array(y_train)
    y_train=y_train.astype(np.float32)


    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 33))


    print("test2")

    model1 = Sequential()

    model1.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 33)))
    model1.add(Dropout(0.2))

    model1.add(LSTM(units=50,return_sequences=True))
    model1.add(Dropout(0.2))

    model1.add(LSTM(units=50,return_sequences=True))
    model1.add(Dropout(0.2))

    model1.add(LSTM(units=50,return_sequences=True))
    model1.add(Dropout(0.2))

    model1.add(LSTM(units=50))
    model1.add(Dropout(0.2))


    model1.add(Dense(units=1))

    model2 = Sequential()
    model2.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 33)))
    model2.add(Dropout(0.1))
    model2.add(LSTM(units=50, return_sequences=True))
    model2.add(Dropout(0.1))
    model2.add(LSTM(units=50))
    model2.add(Dropout(0.1))
    model2.add(Dense(units=1))

    # Compile and train the model
    model1.compile(optimizer='adam', loss='mean_squared_error')
    history = model1.fit(X_train, y_train, epochs=10, batch_size=32)

    model2.compile(optimizer='adam', loss='mean_squared_error')
    history2 = model2.fit(X_train, y_train, epochs=10, batch_size=22)

    ensemble_model = concatenate([model1.output, model2.output])
    ensemble_outputs = Dense(units=1)(ensemble_model)

    final_model = Model(inputs=[model1.input, model2.input], outputs=ensemble_outputs)

    final_model.compile(optimizer='adam', loss='mean_squared_error')
    history_ensemble = final_model.fit([X_train, X_train], y_train, epochs=10, batch_size=32)



    close_price=MinMaxScaler(feature_range=(0,1))
    test_price = y_test_scaled


    print("test3")
    
    dataset_total = stock_data.drop(columns=['Date','Close'])

    inputs = dataset_total[len(dataset_total) - len(test_price) - 60:]

    inputs=np.array(inputs)
    inputs = inputs.reshape(-1, 33)
    inputs = min_max.transform(inputs)
    X_tes = []
    for i in range(60,76):
        X_tes.append(inputs[i - 60:i, :])
    X_tes = np.array(X_tes)

    predicted_stock_price = final_model.predict([X_test,X_test])



    print("test4")

    close_price.fit(stock_data['Close'].values.reshape(-1,1))

    predicted_stock_price = close_price.inverse_transform(predicted_stock_price.reshape(-1,1))

    history_df = pd.DataFrame(history_ensemble.history)
    p=history_df.loc[:, ['loss']].plot()
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('Loss')
    plt.savefig('./main/web/static/loss.png')

    m=final_model.predict([X_train,X_train])
    n=final_model.predict([X_test,X_test])
    a=[]
    b=[]
    c=[]
    for i in range(76,76+len(X_train)):
        a.append(i)
    for i in range(76+len(X_train),76+len(X_train)+len(X_test)):
        b.append(i)
    #stock_data.drop(columns=['Date','Close'])
    plt.figure(figsize=(12,6))
    plt.plot(a,close_price.inverse_transform(m.reshape(-1,1)))
    plt.plot(b,close_price.inverse_transform(n.reshape(-1,1)))
    l=stock_data.loc[: , 'Close']
    for i in range(0,len(l)):
        c.append(i)

    plt.plot(c,l)
    plt.ylabel('Price')
    plt.xlabel('Day Number')
    plt.title('Chart with Predictions')
    plt.legend(['Predicted Train_data', 'Predicted Test_data','Actual Price'])

    plt.savefig('./main/web/static/Chart_with_predictions.png')
    

    w=final_model.predict([X_test[-16:],X_test[-16:]])
    plt.figure(figsize=(6,6))
    w_t=close_price.inverse_transform(w.reshape(-1,1))
    plt.plot(w_t)
    plt.ylabel('Price')
    plt.xlabel('Next Day Number')
    plt.title('16 Day Price Prediction')
    plt.savefig('./main/web/static/16_Day_Price_Prediction.png')
    plt.legend(['price'])

    rmse_score= np.sqrt(mean_squared_error(y_test, predicted_stock_price))


    return w_t,rmse_score
