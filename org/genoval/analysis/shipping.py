import matplotlib.pyplot as plt
import numpy as np



def shippingShow(train):
    print(train.shipping.value_counts() / len(train))    # 0为卖家出邮费，1为买方出邮储

    prc_shipBySeller = train.loc[train.shipping == 0, 'price']
    prc_shipByBuyer = train.loc[train.shipping == 1, 'price']

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.hist(np.log(prc_shipBySeller + 1), color='#8CB4E1', alpha=1.0, bins=50,          #
            label='Price when Seller pays Shipping')
    ax.hist(np.log(prc_shipByBuyer + 1), color='#007D00', alpha=0.7, bins=50,
            label='Price when Buyer pays Shipping')
    ax.set(title='Histogram Comparison', ylabel='% of Dataset in Bin')
    plt.xlabel('log(price+1)', fontsize=17)
    plt.ylabel('frequency', fontsize=17)
    plt.title('Price Distribution by Shipping Type', fontsize=17)
    plt.tick_params(labelsize=15)
    plt.show()