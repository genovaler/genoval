import matplotlib.pyplot as plt
import numpy as np


def priceshow(train):
    print(train.price.describe())    # price的分布情况

    plt.subplot(1, 2, 1)
    (train['price']).plot.hist(bins=50, figsize=(20, 10), edgecolor='white', range=[0, 250])
    plt.xlabel('price+', fontsize=17)
    plt.ylabel('frequency', fontsize=17)
    plt.tick_params(labelsize=15)
    plt.title('Price Distribution - Training Set', fontsize=17)

    plt.subplot(1, 2, 2)
    np.log(train['price'] + 1).plot.hist(bins=50, figsize=(20, 10), edgecolor='white')
    plt.xlabel('log(price+1)', fontsize=17)
    plt.ylabel('frequency', fontsize=17)
    plt.tick_params(labelsize=15)
    plt.title('Log(Price) Distribution - Training Set', fontsize=17)
    plt.show()