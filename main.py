from src.preprocess import load_and_preprocess
from src.model_training import train_random_forest, train_xgboost
from src.forecasting_arima import arima_forecast
from src.visualization import visualize_data, plot_predictions
import os

os.makedirs('models', exist_ok=True)

if __name__=='__main__':
    X_train,X_test,y_train,y_test,df,scaler=load_and_preprocess('data/historical_shipments.csv')
    visualize_data(df)
    rf,rfp=train_random_forest(X_train,X_test,y_train,y_test)
    xgb,xp=train_xgboost(X_train,X_test,y_train,y_test)
    total=len(df); ts=int(total*0.8); dates=df['date'].iloc[ts:].values
    plot_predictions(dates,y_test,rfp,'RF Predictions')
    plot_predictions(dates,y_test,xp,'XGB Predictions')
    f,m=arima_forecast(df,steps=30)
    print(f)
