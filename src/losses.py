import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def rmse(pred, label):
    mse = mean_squared_error(label, pred)
    rmse = np.sqrt(mse)
    return rmse

def mspe(pred, label):
    percentage_error = (label - pred) / label
    mspe = np.mean(percentage_error ** 2)
    return mspe * 100

def mse(pred, label):
    return mean_squared_error(label, pred)

def criterion(pred, label, plot = False):
    if plot:
        plot_predictions(pred, label)
    return mse(pred, label), rmse(pred, label), mspe(pred, label)

def plot_predictions(pred, label):
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 8))
    plt.scatter(pred, label, alpha=0.5) 
    plt.plot([min(label), max(label)], [min(label), max(label)], color='red')
    plt.xlabel('Predictions', fontsize=14)
    plt.ylabel('Labels', fontsize=14)
    plt.title('Predictions vs Labels', fontsize=16)
    plt.show()