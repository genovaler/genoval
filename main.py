import csv

import nltk
import string
import re
import numpy as np
import pandas as pd
import pickle
# import lda

import matplotlib.pyplot as plt
import seaborn as sns

from brand import brandShow
from category import categoryShow
from itemDescription import descriptionShow
from price import priceshow
from shipping import shippingShow

sns.set(style="white")

from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words

from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
# %matplotlib inline

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook
# from bokeh.transform import factor_cmap

import warnings

warnings.filterwarnings('ignore')
import logging

# logging.getLogger("lda").setLevel(logging.WARNING)

PATH = "./input/"
train = pd.read_csv(f'{PATH}train.tsv', quoting=csv.QUOTE_NONE, sep='\t', encoding='utf-8')
test = pd.read_table(f'{PATH}test.tsv', sep='\t', encoding='utf-8')

# size of training and dataset
# print(train.shape)                      # 训练数据共1482535行，8个字段
# print(test.shape)                       # 测试数据共693359行，7个字段（无price字段）

# different data types in the dataset
print(train.dtypes)
# print(train.head())

"""
1. Target Variable : Price 价格分析

The next standard check is with our response or target variables, which in this case is the `price` we are suggesting to
the Mercari's marketplace sellers. The median price of all the items in the training is about $267 but given the exis-
tence of some extreme values of over $100 and the maximum at $2,009, the distribution of the variables is heavily skewed
to the left. So let's make log-transformation on the price (we added +1 to the value before the transformation to avoid
zero and negative values).
"""

# priceshow(train)


"""
2. Shipping:发货方式分析
The shipping cost burden is decently splitted between sellers and buyers with more than half of the items' shipping fees 
are paid by the sellers (55%). In addition, the average price paid by users who have to pay for shipping fees is lower 
than those that don't require additional shipping cost. This matches with our perception that the sellers need a lower 
price to compensate for the additional shipping.
"""

# shippingShow(train)


"""
3. Item Category：商品分类分析
There are about **1,287** unique categories but among each of them,we will always see a main/general category firstly, 
followed by two more particular subcategories (e.g. Beauty/Makeup/Face or Lips).In adidition, there are about 6,327 
items that do not have a category labels. Let's split the categories into three different columns. We will see later 
that this information is actually quite important from the seller's point of view and how we handle the missing 
information in the `brand_name` column will impact the model's prediction.
"""
# categoryShow(train,test)



"""
4.  Brand name :品牌分析 
"""
# brandShow(train)



"""
5. Item Description:商品描述信息

It will be more challenging to parse through this particular item since it's unstructured data.
Does it mean a more detailed and lengthy description will result in a higher bidding price?
We will strip out all punctuations, remove some english stop words (i.e. redundant words such
as "a", "the", etc.) and any other words with a length less than 3.

If we look at the most common words by category, we could also see that, size, free
and shipping is very commonly used by the sellers, probably with the intention to attract
customers, which is contradictory to what  we have shown previously that there is little correlation
between the two variables `price` and `shipping` (or shipping fees do not account for a differentiation
in prices). Brand names also played quite an important role - it's one of the most popular in
all four categories.

Text Processing - Item Description
The following section is based on the tutorial at
https://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html
"""

descriptionShow(train, test)



# """
# # K-Means Clustering
# K-means clustering obejctive is to minimize the average squared Euclidean distance of the document / description from
# their cluster centroids.
# """
# from sklearn.cluster import MiniBatchKMeans
#
# num_clusters = 30 # need to be selected wisely
# kmeans_model = MiniBatchKMeans(n_clusters=num_clusters,
#                                init='k-means++',
#                                n_init=1,
#                                init_size=1000, batch_size=1000, verbose=0, max_iter=1000)
# kmeans = kmeans_model.fit(vz)
# kmeans_clusters = kmeans.predict(vz)
# kmeans_distances = kmeans.transform(vz)
#
# sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names()
#
# for i in range(num_clusters):
#     print("Cluster %d:" % i)
#     aux = ''
#     for j in sorted_centroids[i, :10]:
#         aux += terms[j] + ' | '
#     print(aux)
#     print()
#
#
#
# """
# In order to plot these clusters, first we will need to reduce the dimension of the distances to 2 using tsne:
# """
#
# # repeat the same steps for the sample
# kmeans = kmeans_model.fit(vz_sample)
# kmeans_clusters = kmeans.predict(vz_sample)
# kmeans_distances = kmeans.transform(vz_sample)
# # reduce dimension to 2 using tsne
# tsne_kmeans = tsne_model.fit_transform(kmeans_distances)
#
#
#
# colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
# "#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
# "#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",
# "#52697d", "#194196", "#d27c88", "#36422b", "#b68f79"])
#
#
#
# #combined_sample.reset_index(drop=True, inplace=True)
# kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])
# kmeans_df['cluster'] = kmeans_clusters
# kmeans_df['description'] = combined_sample['item_description']
# kmeans_df['category'] = combined_sample['general_cat']
# #kmeans_df['cluster']=kmeans_df.cluster.astype(str).astype('category')
#
#
#
# #combined_sample.reset_index(drop=True, inplace=True)
# kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])
# kmeans_df['cluster'] = kmeans_clusters
# kmeans_df['description'] = combined_sample['item_description']
# kmeans_df['category'] = combined_sample['general_cat']
# #kmeans_df['cluster']=kmeans_df.cluster.astype(str).astype('category')
#
#
#
# source = ColumnDataSource(data=dict(x=kmeans_df['x'], y=kmeans_df['y'],
#                                     color=colormap[kmeans_clusters],
#                                     description=kmeans_df['description'],
#                                     category=kmeans_df['category'],
#                                     cluster=kmeans_df['cluster']))
#
# plot_kmeans.scatter(x='x', y='y', color='color', source=source)
# hover = plot_kmeans.select(dict(type=HoverTool))
# hover.tooltips={"description": "@description", "category": "@category", "cluster":"@cluster" }
# show(plot_kmeans)
#
#
# """
# #Latent Dirichlet Allocation
#
# Latent Dirichlet Allocation (LDA) is an algorithms used to discover the topics that are present in a corpus.
#
# >  LDA starts from a fixed number of topics. Each topic is represented as a distribution over words, and each document
# is then represented as a distribution over topics. Although the tokens themselves are meaningless, the probability
# distributions over words provided by the topics provide a sense of the different ideas contained in the documents.
# > Reference:
# https://medium.com/intuitionmachine/the-two-paths-from-natural-language-processing-to-artificial-intelligence-d5384ddbfc18
#
# Its input is a **bag of words**, i.e. each document represented as a row, with each columns containing the count of
# words in the corpus. We are going to use a powerful tool called pyLDAvis that gives us an interactive visualization for
# LDA.
# """
#
#
# cvectorizer = CountVectorizer(min_df=4,
#                               max_features=180000,
#                               tokenizer=tokenize,
#                               ngram_range=(1,2))
#
# cvz = cvectorizer.fit_transform(combined_sample['item_description'])
#
# lda_model = LatentDirichletAllocation(n_components=20,
#                                       learning_method='online',
#                                       max_iter=20,
#                                       random_state=42)
#
#
# X_topics = lda_model.fit_transform(cvz)
#
# n_top_words = 10
# topic_summaries = []
#
# topic_word = lda_model.components_  # get the topic words
# vocab = cvectorizer.get_feature_names()
#
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#     topic_summaries.append(' '.join(topic_words))
#     print('Topic {}: {}'.format(i, ' | '.join(topic_words)))
#
# # reduce dimension to 2 using tsne
# tsne_lda = tsne_model.fit_transform(X_topics)
#
# unnormalized = np.matrix(X_topics)
# doc_topic = unnormalized/unnormalized.sum(axis=1)
#
# lda_keys = []
# for i, tweet in enumerate(combined_sample['item_description']):
#     lda_keys += [doc_topic[i].argmax()]
#
# lda_df = pd.DataFrame(tsne_lda, columns=['x','y'])
# lda_df['description'] = combined_sample['item_description']
# lda_df['category'] = combined_sample['general_cat']
# lda_df['topic'] = lda_keys
# lda_df['topic'] = lda_df['topic'].map(int)
#
#
# plot_lda = bp.figure(plot_width=700,
#                      plot_height=600,
#                      title="LDA topic visualization",
#     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
#     x_axis_type=None, y_axis_type=None, min_border=1)
# source = ColumnDataSource(data=dict(x=lda_df['x'], y=lda_df['y'],
#                                     color=colormap[lda_keys],
#                                     description=lda_df['description'],
#                                     topic=lda_df['topic'],
#                                     category=lda_df['category']))
#
# plot_lda.scatter(source=source, x='x', y='y', color='color')
# hover = plot_kmeans.select(dict(type=HoverTool))
# hover = plot_lda.select(dict(type=HoverTool))
# hover.tooltips={"description":"@description",
#                 "topic":"@topic", "category":"@category"}
# show(plot_lda)
#
# def prepareLDAData():
#     data = {
#         'vocab': vocab,
#         'doc_topic_dists': doc_topic,
#         'doc_lengths': list(lda_df['len_docs']),
#         'term_frequency':cvectorizer.vocabulary_,
#         'topic_term_dists': lda_model.components_
#     }
#     return data
#
#
# def prepareLDAData():
#     data = {
#         'vocab': vocab,
#         'doc_topic_dists': doc_topic,
#         'doc_lengths': list(lda_df['len_docs']),
#         'term_frequency':cvectorizer.vocabulary_,
#         'topic_term_dists': lda_model.components_
#     }
#     return data
#
#
# import IPython.display
# from IPython.core.display import display, HTML, Javascript
#
# #h = IPython.display.display(HTML(html_string))
# #IPython.display.display_HTML(h)
#
#
