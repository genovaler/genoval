import plotly.graph_objs as go
import plotly.offline as py
import numpy as np

"""
Item Category：
There are about **1,287** unique categories but among each of them,
we will always see a main/general category firstly, followed by two
more particular subcategories (e.g. Beauty/Makeup/Face or Lips).
In adidition, there are about 6,327 items that do not have a category
labels. Let's split the categories into three different columns. We will
see later that this information is actually quite important from the seller's
point of view and how we handle the missing information in the `brand_name`
column will impact the model's prediction.
"""


def categoryShow(train):
    print("There are %d unique values in the category column." % train['category_name'].nunique())
    # TOP 5 RAW CATEGORIES
    train['category_name'].value_counts()[:5]
    # missing categories
    print("There are %d items that do not have a label." % train['category_name'].isnull().sum())
    # reference: BuryBuryZymon at https://www.kaggle.com/maheshdadhich/i-will-sell-everything-for-free-0-55
    train['general_cat'], train['subcat_1'], train['subcat_2'] = zip(
        *train['category_name'].apply(lambda x: split_cat(x)))
    train.head()
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

    trace1 = go.Bar(x=x, y=y, text=pct)
    layout = dict(title='Number of Items by Main Category',
                  yaxis=dict(title='Count'),
                  xaxis=dict(title='Category'))
    fig = dict(data=[trace1], layout=layout)
    py.iplot(fig)

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
