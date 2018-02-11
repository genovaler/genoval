import plotly.graph_objs as go
import plotly.offline as py
import matplotlib.pyplot as plt
import numpy as np



def categoryShow(train, test):
    print("There are %d unique values in the category column." % train['category_name'].nunique())  # 分类有多少种
    print("# TOP 5 RAW CATEGORIES:")  # 商品数量最多的前5个分类
    print(train['category_name'].value_counts()[:5])
    print("There are %d items that do not have a label." % train['category_name'].isnull().sum())  # 没有商品分类的数量
    # 下面将子级分类增加到新的字段里，例如分类Men/Tops/T-shirts会被分隔成三个子级分类：Men、Tops、T-shirts
    train['general_cat'], train['subcat_1'], train['subcat_2'] = zip(*train['category_name'].apply(lambda x: split_cat(x)))
    test['general_cat'], test['subcat_1'], test['subcat_2'] = zip(*test['category_name'].apply(lambda x: split_cat(x)))
    print("There are %d unique first sub-categories." % train['subcat_1'].nunique())
    print("There are %d unique second sub-categories." % train['subcat_2'].nunique())
    """
    Overall, we have 7 main categories(114 in the first sub-categories and 871 second sub-categories):
    women's and beauty items as the two most popular categories (more than 50% of the observations),
    followed by kids and electronics.
    """
    x = train['general_cat'].value_counts().index.values.astype('str')
    y = train['general_cat'].value_counts().values
    pct = [("%.2f" % (v * 100)) + "%" for v in (y / len(train))]


    # 方式1：plotly绘制web图，用以和IPyhotn notebook交互，这里不用IPython notebook，改用matplotlib
    """
    trace1 = go.Bar(x=x, y=y, text=pct)
    layout = dict(title='Number of Items by Main Category',
                  yaxis=dict(title='Count'),
                  xaxis=dict(title='Category'))
    fig = dict(data=[trace1], layout=layout)
    py.iplot(fig)
    """

    # matplotlib绘制图
    explode = [0, 0.1, 0, 0]  # 0.1 凸出这部分，
    plt.axes(aspect=1)  # set this , Figure is round, otherwise it is an ellipse
    # autopct ，show percet
    plt.pie(x=y, labels=x, explode=explode, autopct='%3.1f %%',
            shadow=True, labeldistance=1.1, startangle=90, pctdistance=0.6)
    plt.show()




    x = train['subcat_1'].value_counts().index.values.astype('str')[:15]
    y = train['subcat_1'].value_counts().values[:15]
    pct = [("%.2f" % (v * 100)) + "%" for v in (y / len(train))][:15]

    trace1 = go.Bar(x=x, y=y, text=pct,
                    marker=dict(
                        color=y, colorscale='Portland', showscale=True,
                        reversescale=False
                    ))
    layout = dict(title='Number of Items by Sub Category (Top 15)',
                  yaxis=dict(title='Count'),
                  xaxis=dict(title='SubCategory'))
    fig = dict(data=[trace1], layout=layout)
    py.iplot(fig)


    """
    From the pricing (log of price) point of view, all the categories are pretty well distributed,
    with no category with an extraordinary pricing point
    """

    general_cats = train['general_cat'].unique()
    x = [train.loc[train['general_cat'] == cat, 'price'] for cat in general_cats]

    data = [go.Box(x=np.log(x[i] + 1), name=general_cats[i]) for i in range(len(general_cats))]

    layout = dict(title="Price Distribution by General Category",
                  yaxis=dict(title='Frequency'),
                  xaxis=dict(title='Category'))
    fig = dict(data=data, layout=layout)
    py.iplot(fig)


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")
