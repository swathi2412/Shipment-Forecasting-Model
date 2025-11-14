import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    plt.figure(figsize=(12,5))
    sns.lineplot(data=df,x='date',y='shipment_volume',label='Shipment')
    sns.lineplot(data=df,x='date',y='backlog',label='Backlog')
    plt.legend(); plt.show()

def plot_predictions(dates,actual,preds,title='Pred vs Actual'):
    plt.figure(figsize=(12,4))
    plt.plot(dates,actual,label='Actual')
    plt.plot(dates,preds,label='Predicted')
    plt.title(title); plt.legend(); plt.show()
