import plotly.graph_objs as go
import plotly.offline as py
import plotly.plotly as pp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def brandShow(train):
    print("There are %d unique brand names in the training dataset." % train['brand_name'].nunique())
    brandVolueCount = train['brand_name'].value_counts()
    x = np.append(brandVolueCount.index.values.astype('str')[:10],"others")
    y = np.append(brandVolueCount.values[:10],brandVolueCount.values[10:].sum())
    print("brand_y的类型：%s" % type(y))
    pct = [100. * v for v in (y / train['brand_name'].value_counts().sum())]

    # trace1 = go.Bar(x=x, y=y,
    #                 marker=dict(
    #                     color=y, colorscale='Portland', showscale=True,
    #                     reversescale=False
    #                 ))
    # layout = dict(title='Top 10 Brand by Number of Items',
    #               yaxis=dict(title='Brand Name'),
    #               xaxis=dict(title='Count'))
    # fig = dict(data=[trace1], layout=layout)
    # py.iplot(fig)

    plt.axes(aspect=1)  # set this , Figure is round, otherwise it is an ellipse
    patches, texts = plt.pie(y, startangle=90, radius=1.2)
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(x, pct)]
    sort_legend = True
    if sort_legend:
        patches, labels, dummy = zip(*sorted(zip(patches, labels, y), key=lambda x: x[2], reverse=True))
    plt.legend(patches, labels, loc='left center', bbox_to_anchor=(-0.1, 1.), fontsize=8)
    plt.title("Top 10 Brand by Number of Items")
    plt.savefig('Top10Brand.png', bbox_inches='tight')
    plt.show()
    plt.close()






