import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path,test_size=0.2,random_state=42):
    df=pd.read_csv(path,parse_dates=['date'])
    df=df.sort_values('date').reset_index(drop=True)
    df['month']=df['date'].dt.month
    df['day_of_week']=df['date'].dt.dayofweek
    df['day']=df['date'].dt.day
    df['year']=df['date'].dt.year
    df['shipment_lag1']=df['shipment_volume'].shift(1).fillna(method='bfill')
    df['shipment_lag7']=df['shipment_volume'].shift(7).fillna(method='bfill')
    feats=['backlog','month','day_of_week','day','year','shipment_lag1','shipment_lag7']
    X=df[feats].astype(float); y=df['shipment_volume'].astype(float)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,shuffle=False)
    sc=StandardScaler()
    X_train_s=sc.fit_transform(X_train); X_test_s=sc.transform(X_test)
    return X_train_s,X_test_s,y_train.values,y_test.values,df,sc
