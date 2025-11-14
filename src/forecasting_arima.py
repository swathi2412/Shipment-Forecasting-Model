import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(df,steps=12,order=(5,1,0),plot=True):
    ts=df.set_index('date')['shipment_volume'].asfreq('D')
    model=ARIMA(ts,order=order).fit()
    f=model.forecast(steps=steps)
    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(ts[-200:],label='Historical')
        plt.plot(f.index,f.values,label='Forecast')
        plt.legend(); plt.show()
    return f,model
