# Importling libraries 
import streamlit as st 
import datetime 
from pandas_datareader import data as pdr 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import tensorflow as tf 
from tensorflow.keras import regularizers 
from tensorflow.keras import datasets, layers, models 
from tensorflow.keras.models import Sequential , Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout , Input, Activation, SimpleRNN
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn import metrics 
import yfinance as yf
tf.random.set_seed(7)

# All libraries installed 

st.title("Stock Prediction using LSTM, ANN, RNN, MLP and Autoencoders")
st.write(""" #### Select the method of input:""")
option = st.radio("Select One", ["Get data from the net", "Upload the data (.csv format)"])
# st.sidebar.title("Created by")
# st.sidebar.subheader("Ritwik Raj")

# temp = st.text_input("Enter your name :")
# st.write("Your name is ",temp)
# flag = False

if option == "Get data from the net": 
    st.write(""" #### Query parameters""" )

    indian = st.radio("Stock listed on the NSE ? ",['Yes','No'] )
    stock_name = st.text_input("Enter company stock code :")

    def get_data(stock_name, indian):
        stock_name = stock_name.upper()
        if indian == 'Yes': 
            start = st.text_input("Enter start date (yyyy-mm-dd)")
            end = st.text_input("Enter end date (yyyy-mm-dd)")
            stock_name = stock_name + ".NS"
            stock_data = yf.download(stock_name, start=start, end=end)
            stock_data.reset_index(inplace=True)
            # flag = True
            # return stock_data, flag
            return stock_data
        elif indian == 'No': 
            start = st.text_input("Enter start date (yyyy-mm-dd)")
            end = st.text_input("Enter end date (yyyy-mm-dd)")
            stock_data = yf.download(stock_name, start=start, end=end)
            stock_data.reset_index(inplace=True)
            # flag = True
            # return stock_data, flag
            return stock_data
        else:
            return "Incorrect input"
    data = get_data(stock_name, indian)
    # flag = data[1]
    st.write(f"Displaying the first 5 rows of the '{(stock_name).upper()}' dataset.")
    # st.dataframe(data[0].head())
    st.dataframe(data.head())
    st.write("#### Closing Price")
    st.line_chart(data['Close'])

elif option == "Upload the data (.csv format)": 
    stock_name = st.text_input("Enter company stock code :")
    file = st.file_uploader("Dataset")
    if file is not None:
        data = pd.read_csv(file)
        data.reset_index(inplace=True)
        # flag = True
        st.write(f"Displaying the first 5 rows of the '{(stock_name).upper()}' dataset.")
        st.dataframe(data.head())
        st.write("#### Closing Price")
        st.line_chart(data['Close'])


class stock_predict_DL: 
    
    def __init__(self, company_dataframe):
        data = company_dataframe.filter(['Open'])
        dataset = data.values 
        st.write(""" ### How much percent of the data needs to be allocated for training ?""")
        st.text("(Default is set to 90)")
        perc_train = st.number_input('', step=1, min_value=40, value=90)
        training_data_len = int(np.ceil(len(dataset)*(perc_train/100)))
        self.scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = self.scaler.fit_transform(dataset)
        train_data = scaled_data[0:int(training_data_len)]
        self.X_train = []
        self.y_train = []
        for i in range(k,len(train_data)):
            self.X_train.append(train_data[i-k:i])
            self.y_train.append(train_data[i])
        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)
        # st.write(self.X_train)
        test_data = scaled_data[training_data_len - k :]
        self.X_test = []
        self.y_test = dataset[training_data_len:]
        for i in range(k, len(test_data)):
            self.X_test.append(test_data[i-k:i])
        self.X_test = np.array(self.X_test)
        test_dates = company_dataframe['Date'].values 
        self.test_d = test_dates[training_data_len:] # stores the test dates 
        # st.write(self.X_test)

    def LSTM_model(self):

        st.write(""" ### Long Short-Term Memory (LSTM)""")
        # Reshape the data
        Xtrain = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        # Reshape the data
        Xtest = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1 ))
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape= (Xtrain.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        # We are adding dropout to reduce overfitting 
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Train the model
        model.fit(Xtrain, self.y_train, batch_size=1, epochs= 1)
         # Get the models predicted price values 
        predictions = model.predict(Xtest)
        # We need to inverse transform the scaled data to compare it with our unscaled y_test data
        predictions = self.scaler.inverse_transform(predictions)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, predictions))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, predictions))
        plt.plot(predictions)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        # plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title("LSTM")
        plt.grid(True)
        st.pyplot(plt)

        # Forecasting on the last 'k' days
        X_forecast = model.predict(Xtest[-k :])
        X_forecast = self.scaler.inverse_transform(X_forecast)

        prediction_series = pd.Series(predictions[:,0])
        y_test_series = pd.Series(self.y_test[:,0])
        X_forecast = pd.Series(X_forecast[:, 0])

        df1 = pd.concat([prediction_series, y_test_series] , axis=1)
        df2 = pd.concat([df1, X_forecast], axis=0, ignore_index=True)
        df2 = df2.fillna(0)

        st.write(""" #### Forecasted values """)
        st.dataframe(df2[0][-k:])

        limit = len(df2) - k

        plt.plot(df2[1][:limit], color='green', label='Actual Price Movement')
        plt.plot(df2[0][:limit], color='orange', label='Predicted Price Movement')
        plt.plot(df2[0][limit:], label='Forecast', color='red')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)



    def autoen_model(self):
        
        st.write(""" ### Autoencoder""")
        # No of encoding dimensions
        encoding_dim = 32
        input_dim = self.X_train.shape[1]
        input_layer = Input(shape=(input_dim, ))
        # Encoder
        encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(1e-5))(input_layer)
        encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
        # Decoder
        decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
        decoder = Dense(1, activation='relu')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        nb_epoch = 10
        b_size = 32
        # Fitting and compiling the train data using adam (stochastic gradient) optimiser and mse loss
        autoencoder.compile(optimizer='adam',loss='mean_squared_error')
        autoencoder.fit(self.X_train, self.y_train,epochs=nb_epoch,batch_size = b_size,shuffle=True)
        predictions = autoencoder.predict(self.X_test)
        predictions = self.scaler.inverse_transform(predictions)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, predictions))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, predictions))
        plt.plot(predictions)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        # plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title("AUTOENCODER")
        plt.grid(True)
        st.pyplot(plt)

        # Forecasting on the last 'k' days
        X_forecast = autoencoder.predict(self.X_test[-k :])
        X_forecast = self.scaler.inverse_transform(X_forecast)

        prediction_series = pd.Series(predictions[:,0])
        y_test_series = pd.Series(self.y_test[:,0])
        X_forecast = pd.Series(X_forecast[:, 0])

        df1 = pd.concat([prediction_series, y_test_series] , axis=1)
        df2 = pd.concat([df1, X_forecast], axis=0, ignore_index=True)
        df2 = df2.fillna(0)

        st.write(""" #### Forecasted values """)
        st.dataframe(df2[0][-k:])

        limit = len(df2) - k

        plt.plot(df2[1][:limit], color='green', label='Actual Price Movement')
        plt.plot(df2[0][:limit], color='orange', label='Predicted Price Movement')
        plt.plot(df2[0][limit:], label='Forecast', color='red')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    def Mlp_model(self):
        
        st.write(""" ### Multilayer perceptron (MLP)""")
        # We are using MLPRegressor as the problem at hand is a regression problem
        regr = MLPRegressor(hidden_layer_sizes = 100, alpha = 0.01,solver = 'lbfgs',shuffle=True)
        
        # https://stackoverflow.com/questions/34972142/sklearn-logistic-regression-valueerror-found-array-with-dim-3-estimator-expec
        # Link to the solution solved by the code below
        nsamples, nx, ny = self.X_train.shape 
        temp = self.X_train.reshape((nsamples, nx*ny))
        
        regr.fit(temp, self.y_train)
        # predicting the price

        nsamples, nx, ny = self.X_test.shape
        temp_test = self.X_test.reshape((nsamples, nx*ny))

        y_pred = regr.predict(temp_test)
        y_pred = y_pred.reshape(len(y_pred),1)
        y_pred = self.scaler.inverse_transform(y_pred)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, y_pred))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, y_pred))
        plt.plot(y_pred)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        # plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title("MLP")
        plt.grid(True)
        st.pyplot(plt)

        st.write("Sorry! Working on forecasting for the Multi Layer Perceptron model. In the meantime try a different model.")
        # Forecasting on the last 'k' days
        # X_forecast = regr.predict(temp_test[-k :])
        # a = self.X_test.shape[0]-k+1
        # X_forecast = X_forecast.reshape(a,self.X_test.shape[1], self.X_test.shape[2])
        # temp_test.reshape(a,X_test.shape[1], X_test.shape[2]).shape

        # X_forecast = self.scaler.inverse_transform(X_forecast)

        # prediction_series = pd.Series(y_pred[:,0])
        # y_test_series = pd.Series(self.y_test[:,0])
        # X_forecast = pd.Series(X_forecast[:, 0])

        # df1 = pd.concat([prediction_series, y_test_series] , axis=1)
        # df2 = pd.concat([df1, X_forecast], axis=0, ignore_index=True)
        # df2 = df2.fillna(0)

        # st.write(""" #### Forecasted values """)
        # st.dataframe(df2[0][-k:])

        # limit = len(df2) - k

        # plt.plot(df2[1][:limit], color='green', label='Actual Price Movement')
        # plt.plot(df2[0][:limit], color='orange', label='Predicted Price Movement')
        # plt.plot(df2[0][limit:], label='Forecast', color='red')
        # plt.legend()
        # st.pyplot(plt)

    def basic_ann_model(self):
        
        st.write("""### Basic Artificial Neural Network (ANN)""")
        classifier = Sequential()
        classifier.add(Dense(units = 128, activation = 'relu', input_dim = self.X_train.shape[1]))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(units = 64))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(units = 1))
        # We are adding dropout to reduce overfitting
        # adam is one of the best optimzier for DL as it uses stochastic gradient method
        # Mean Square Error (MSE) is the most commonly used regression loss function.
        # MSE is the sum of squared distances between our target variable and predicted values.
        classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')
        classifier.fit(self.X_train, self.y_train, batch_size = 32, epochs = 10)
        # Predicting the prices
        prediction = classifier.predict(self.X_test)
        y_pred = self.scaler.inverse_transform(prediction)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, y_pred))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, y_pred))
        plt.plot(y_pred)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        # plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title("ANN")
        plt.grid(True)
        st.pyplot(plt)

        # Forecasting on the last 'k' days
        X_forecast = classifier.predict(self.X_test[-k :])
        X_forecast = self.scaler.inverse_transform(X_forecast)

        prediction_series = pd.Series(y_pred[:,0])
        y_test_series = pd.Series(self.y_test[:,0])
        X_forecast = pd.Series(X_forecast[:, 0])

        df1 = pd.concat([prediction_series, y_test_series] , axis=1)
        df2 = pd.concat([df1, X_forecast], axis=0, ignore_index=True)
        df2 = df2.fillna(0)

        st.write(""" #### Forecasted values """)
        st.dataframe(df2[0][-k:])

        limit = len(df2) - k

        plt.plot(df2[1][:limit], color='green', label='Actual Price Movement')
        plt.plot(df2[0][:limit], color='orange', label='Predicted Price Movement')
        plt.plot(df2[0][limit:], color='red', label='Forecast')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    def rnn_model(self):
        
        st.write("### Recurrent neural network (RNN)")
        # Reshape the data
        Xtrain = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        # Reshape the data
        Xtest = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1 ))
        model = Sequential()
        model.add(SimpleRNN(units=4, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(Xtrain, self.y_train, epochs=10, batch_size=1)
        # predicting the opening prices
        prediction = model.predict(Xtest)
        y_pred = self.scaler.inverse_transform(prediction)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, y_pred))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, y_pred))
        plt.plot(y_pred)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        # plt.xticks(range(0,len(self.y_test),50),self.testd,rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title("RNN")
        plt.grid(True)
        st.pyplot(plt)

        # Forecasting on the last 'k' days
        X_forecast = model.predict(Xtest[-k :])
        X_forecast = self.scaler.inverse_transform(X_forecast)

        prediction_series = pd.Series(y_pred[:,0])
        y_test_series = pd.Series(self.y_test[:,0])
        X_forecast = pd.Series(X_forecast[:, 0])

        df1 = pd.concat([prediction_series, y_test_series] , axis=1)
        df2 = pd.concat([df1, X_forecast], axis=0, ignore_index=True)
        df2 = df2.fillna(0)

        st.write(""" #### Forecasted values """)
        st.dataframe(df2[0][-k:])
        
        limit = len(df2) - k

        plt.plot(df2[1][:limit], color='green', label='Actual Price Movement')
        plt.plot(df2[0][:limit], color='orange', label='Predicted Price Movement')
        plt.plot(df2[0][limit:], label='Forecast', color='red')
        plt.legend()
        plt.grid(True)        
        st.pyplot(plt)

# if (flag == True):
# st.write(""" #### Define time window length :""")
# st.text("Default set to 60")
# k = st.number_input('',step=1,min_value=1,value=60)
# st.write(""" ### Which deep learning model would you like to train ? """)

# st.write("TESTING DL MODELS (delete later), (also add a link to the webpage regarding the latest news on the stock) ")
# ABOVE TASK DONE 

st.write("""### Input Time frame""")
k = st.number_input('', step=1, min_value=1, value=60)
company_stock = stock_predict_DL(data)

st.subheader("Which Deep Learning model would you like to train on ?")
model_select = st.selectbox('Select model', ["Click to select","LSTM", "MLP", "RNN", "Basic ANN", "Autoencoder"])

if model_select=="LSTM":
    company_stock.LSTM_model()

if model_select=="MLP":
    company_stock.Mlp_model()

if model_select == "RNN":
    company_stock.rnn_model()

if model_select=="Autoencoder":
    company_stock.autoen_model()

if model_select == "Basic ANN":
    company_stock.basic_ann_model()

st.write(f""" #### For further info on the {stock_name.upper()} stock visit : [link](https://www.google.com/search?q={stock_name}+stock+news&sca_esv=fc4b81c41d7ac4c9&sca_upv=1&sxsrf=ACQVn0-wHQwaWJHhtPSqaIP_hQfaRcty2g%3A1714154290689&ei=MusrZoXXKeSrvr0PuL258A8&ved=0ahUKEwiFuNisuuCFAxXkla8BHbheDv4Q4dUDCBE&uact=5&oq=tcs+stock+news&gs_lp=Egxnd3Mtd2l6LXNlcnAiDnRjcyBzdG9jayBuZXdzMgsQABiABBiRAhiKBTIGEAAYFhgeMgYQABgWGB4yBhAAGBYYHjILEAAYgAQYhgMYigUyCxAAGIAEGIYDGIoFMgsQABiABBiGAxiKBTILEAAYgAQYhgMYigUyCxAAGIAEGIYDGIoFMggQABiABBiiBEiwCVCMA1i_CHABeAGQAQCYAZwBoAGQBKoBAzAuNLgBA8gBAPgBAZgCBaACpwTCAgoQABiwAxjWBBhHwgIOEAAYgAQYkQIYsQMYigXCAgUQABiABJgDAIgGAZAGCJIHAzEuNKAHrRk&sclient=gws-wiz-serp)""")

## THINGS LEFT TO DO 

# Add datetime values to the x-index 
# Refine code structure and possibilities of bugs 