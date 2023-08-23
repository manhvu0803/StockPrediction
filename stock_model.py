from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import xgboost as xgb

scaler = MinMaxScaler(feature_range=(0,1))

def predict(data, modelFileName, timeColumn="Date", timeDelta=pd.Timedelta(minutes=1), shape=60, isXgb=False):
    new_data = data.loc[:, (timeColumn, 'close')]

    newestDate = new_data.iloc[len(new_data) - 1][timeColumn]
    newestPrice = new_data.iloc[len(new_data) - 1]["close"]

    for i in range(1, 10):
        new_data.loc[len(new_data)] = [newestDate + i * timeDelta, newestPrice]

    limit = len(new_data) // 8 * 7
    new_data.set_index(timeColumn, inplace=True)
    new_data.sort_index(axis=0)

    # Drop date since it's already the index
    try:
        new_data.drop(timeColumn, axis=1, inplace=True)
    except:
        pass

    dataset=new_data.values

    train = dataset[0:limit, :]
    valid = dataset[limit:, :]

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)

    x_train,y_train=[],[]

    for i in range(shape,len(train)):
        x_train.append(scaled_data[i-shape:i,0])
        y_train.append(scaled_data[i,0])

    x_train,y_train=np.array(x_train),np.array(y_train)

    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    inputs = new_data[len(new_data) - len(valid) - shape:].values
    # inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(shape, inputs.shape[0]):
        X_test.append(inputs[i - shape : i, 0])
    X_test = np.array(X_test)

    if (isXgb):
        model = xgb.XGBRegressor()
        model.load_model("./model/" + modelFileName)
    else:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        model=load_model("./model/" + modelFileName)

    closing_price = model.predict(X_test)

    if (isXgb):
        closing_price = np.reshape(closing_price, (closing_price.shape[0], 1))

    closing_price = scaler.inverse_transform(closing_price)

    train=new_data[:limit]
    valid=new_data[limit:]
    valid['predictions'] = closing_price

    return (train, valid)

def readAndPredict(filename):
    df_nse = pd.read_csv("./data/" + filename)
    df_nse["Date"]=pd.to_datetime(df_nse.Date, format="mixed")
    df_nse.index=df_nse['Date']
    data=df_nse.sort_index(ascending=True,axis=0)
    new_data = data.loc["2022-02-28":, ('Date', 'Close')]
    return predict(new_data, filename)
