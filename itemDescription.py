import re
import string
import plotly.graph_objs as go
import plotly.offline as py
import plotly.plotly as pp
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import HoverTool
from wordcloud import WordCloud
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from category import split_cat
from sklearn.feature_extraction.text import TfidfVectorizer
import bokeh.plotting as bp
from bokeh.plotting import figure, show, output_notebook


def descriptionShow(train, test):
    """
    # add a column of word counts to both the training and test set
    train['desc_len'] = train['item_description'].apply(lambda x: wordCount(x))
    test['desc_len'] = test['item_description'].apply(lambda x: wordCount(x))

    train['general_cat'], train['subcat_1'], train['subcat_2'] = zip(
        *train['category_name'].apply(lambda x: split_cat(x)))
    print(train.head())
    """

    # # apply the tokenizer into the item descriptipn column
    # train['tokens'] = train['item_description'].map(tokenize)
    # test['tokens'] = test['item_description'].map(tokenize)
    # train.reset_index(drop=True, inplace=True)
    # test.reset_index(drop=True, inplace=True)

    # Let's look at the examples of if the tokenizer did a good job in cleaning up our descriptions
    # for description, tokens in zip(train['item_description'].head(), train['tokens'].head()):
    #     print('description:', description)
    #     print('tokens:', tokens)
    #     print()

    # df = train.groupby('desc_len')['price'].mean().reset_index()
    # x = df['desc_len']
    # y = np.log(df['price'] + 1)
    # plt.xlabel("Description Length")
    # plt.ylabel("Log(price)")
    # plt.scatter(x, y)
    # plt.plot(x, y)
    # plt.savefig('PriceDistributionByDescriptionLength.png', bbox_inches='tight')
    # plt.show()


    # trace1 = go.Scatter(
    #     x=df['desc_len'],
    #     y=np.log(df['price'] + 1),
    #     mode='lines+markers',
    #     name='lines+markers'
    # )
    # layout = dict(title='Average Log(Price) by Description Length',
    #               yaxis=dict(title='Average Log(Price)'),
    #               xaxis=dict(title='Description Length'))
    # fig = dict(data=[trace1], layout=layout)
    # py.iplot(fig)

    """
    We also need to check if there are any missing values in the item description (4 observations
    don't have a description) andl remove those observations from our training set.
    """

    # train.item_description.isnull().sum()
    # remove missing values in item description
    train = train[pd.notnull(train['item_description'])]

    # # build dictionary with key=category and values as all the descriptions related.
    # cat_desc = dict()
    # # create a dictionary of words for each category
    # general_cats = train['general_cat'].unique()
    # for cat in general_cats:
    #     text = " ".join(train.loc[train['general_cat'] == cat, 'item_description'].values)
    #     cat_desc[cat] = tokenize(text)
    #





    """
    the Top 20 Word Count Show:
    """

    """
    # flat list of all words combined
    flat_lst = [item for sublist in list(cat_desc.values()) for item in sublist]
    allWordsCount = Counter(flat_lst)
    allwordsum = sum(allWordsCount.values())
    all_top10 = allWordsCount.most_common(20)
    print("最常见的20个单词：\n")
    print(all_top10)

    for i in all_top10:
        allWordsCount.pop(i[0])
    others = allWordsCount

    x = [w[0] for w in all_top10]
    x.append("others")
    y = [w[1] for w in all_top10]
    y.append(sum(others.values()))
    pct = [100. * v / allwordsum for v in y]

    # trace1 = go.Bar(x=x, y=y, text=pct)
    # layout = dict(title='Word Frequency',
    #               yaxis=dict(title='Count'),
    #               xaxis=dict(title='Word'))
    # fig = dict(data=[trace1], layout=layout)
    # py.iplot(fig)

    plt.axes(aspect=1)  # set this , Figure is round, otherwise it is an ellipse
    patches, texts = plt.pie(y, startangle=90, radius=1.2)
    labels = ['{0} - {1:1.2f} % (count:{2})'.format(i, j, z) for i, j, z in zip(x, pct, y)]
    sort_legend = True
    if sort_legend:
        patches, labels, dummy = zip(*sorted(zip(patches, labels, y), key=lambda x: x[2], reverse=True))
    plt.legend(patches, labels, loc='left center', bbox_to_anchor=(-0.1, 1.), fontsize=8)
    plt.title("Top 20 Word Count In All Item Description")
    plt.savefig('Top20WordCount.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    """

    """
    The Top4 Categary Word Cloud Show:
    We could also use the package `WordCloud` to easily visualize which words has the highest frequencies within each 
    category:
    """

    """
    # find the most common words for the top 4 categories
    women100 = Counter(cat_desc['Women']).most_common(100)
    beauty100 = Counter(cat_desc['Beauty']).most_common(100)
    kids100 = Counter(cat_desc['Kids']).most_common(100)
    electronics100 = Counter(cat_desc['Electronics']).most_common(100)

    fig, axes = plt.subplots(2, 2, figsize=(30, 15))

    ax = axes[0, 0]
    ax.imshow(generate_wordcloud(women100), interpolation="bilinear")
    ax.axis('off')
    ax.set_title("Women Top 100", fontsize=30)

    ax = axes[0, 1]
    ax.imshow(generate_wordcloud(beauty100))
    ax.axis('off')
    ax.set_title("Beauty Top 100", fontsize=30)

    ax = axes[1, 0]
    ax.imshow(generate_wordcloud(kids100))
    ax.axis('off')
    ax.set_title("Kids Top 100", fontsize=30)

    ax = axes[1, 1]
    ax.imshow(generate_wordcloud(electronics100))
    ax.axis('off')
    ax.set_title("Electronic Top 100", fontsize=30)
    plt.savefig('Top4CategaryWordCloudShow', bbox_inches='tight')
    plt.show()
    plt.close()
    """

    """
    Pre-processing: tf-idf
    
    tf-idf is the acronym for Term Frequency–inverse Document Frequency. It quantifies the importance of a particular 
    word in relative to the vocabulary of a collection of documents or corpus. The metric depends on two factors:
    Term Frequency: the occurences of a word in a given document (i.e. bag of words)
    Inverse Document Frequency: the reciprocal number of times a word occurs in a corpus of documents
    
    Think about of it this way: If the word is used extensively in all documents, its existence within a specific 
    document will not be able to provide us much specific information about the document itself. So the second term 
    could be seen as a penalty term that penalizes common words such as "a", "the", "and", etc. tf-idf can therefore, 
    be seen as a weighting scheme for words relevancy in a specific document.
    """
    vectorizer = TfidfVectorizer(min_df=10,
                                 max_features=180000,
                                 tokenizer=tokenize,
                                 ngram_range=(1, 2))
    # all_desc = np.append(train['item_description'].values, test['item_description'].values)
    # """"
    # vz is a tfidf matrix where:
    # * the number of rows is the total number of descriptions
    # * the number of columns is the total number of unique tokens across the descriptions
    # """
    # vz = vectorizer.fit_transform(all_desc.tolist())
    # print(vz.shape[0])
    # print(vz.shape[1])
    # #  create a dictionary mapping the tokens to their tfidf values
    # tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    # tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
    #                     dict(tfidf), orient='index')
    # tfidf.columns = ['tfidf']
    # """
    # Below is the 10 tokens with the lowest tfidf score, which is unsurprisingly, very generic words that we could not
    # use to distinguish one description from another.
    # """
    # print(tfidf.sort_values(by=['tfidf'], ascending=True).head(10))
    # """
    # Below is the 10 tokens with the highest tfidf score, which includes words that are a lot specific that by looking at
    # them, we could guess the categories that they belong to:
    # """
    # print(tfidf.sort_values(by=['tfidf'], ascending=False).head(10))
    """
    Given the high dimension of our tfidf matrix, we need to reduce their dimension using the Singular Value Decomposit
    -ion (SVD) technique. And to visualize our vocabulary, we could next use t-SNE to reduce the dimension from 50 to 2. 
    t-SNE is more suitable for dimensionality reduction to 2 or 3.
    
    t-Distributed Stochastic Neighbor Embedding (t-SNE)
    t-SNE is a technique for dimensionality reduction that is particularly well suited for the visualization of high-
    dimensional datasets. The goal is to take a set of points in a high-dimensional space and find a representation of 
    those points in a lower-dimensional space, typically the 2D plane. It is based on probability distributions with 
    random walk on neighborhood graphs to find the structure within the data. But since t-SNE complexity is significant
    -ly high, usually we'd use other high-dimension reduction techniques before applying t-SNE.
    First, let's take a sample from the both training and testing item's description since t-SNE can take a very long time
    to execute. We can then reduce the dimension of each vector from to n_components (50) using SVD.
    """
    trn = train.copy()
    tst = test.copy()
    trn['is_train'] = 1
    tst['is_train'] = 0

    sample_sz = 15000

    combined_df = pd.concat([trn, tst])
    combined_sample = combined_df.sample(n=sample_sz)
    vz_sample = vectorizer.fit_transform(list(combined_sample['item_description']))

    from sklearn.decomposition import TruncatedSVD

    n_comp = 30
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd_tfidf = svd.fit_transform(vz_sample)

    # Now we can reduce the dimension from 50 to 2 using t-SNE!
    from sklearn.manifold import TSNE
    tsne_model = TSNE(n_components=2, verbose=1, random_state=42, n_iter=500)

    tsne_tfidf = tsne_model.fit_transform(svd_tfidf)

    """
    It's now possible to visualize our data points. Note that the deviation as well as the size of the clusters imply little
    information  in t-SNE.
    """
    output_notebook()
    plot_tfidf = bp.figure(plot_width=700, plot_height=600,
                           title="tf-idf clustering of the item description",
                           tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                           x_axis_type=None, y_axis_type=None, min_border=1)
    combined_sample.reset_index(inplace=True, drop=True)
    tfidf_df = pd.DataFrame(tsne_tfidf, columns=['x', 'y'])
    tfidf_df['description'] = combined_sample['item_description']
    tfidf_df['tokens'] = combined_sample['tokens']
    tfidf_df['category'] = combined_sample['general_cat']

    plot_tfidf.scatter(x='x', y='y', source=tfidf_df, alpha=0.7)
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips = {"description": "@description", "tokens": "@tokens", "category": "@category"}
    show(plot_tfidf)


def wordCount(text):
    # convert to lower case and strip regex
    try:
        # convert to lower case and strip regex
        text = text.lower()
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        txt = regex.sub(" ", text)
        # tokenize
        # words = nltk.word_tokenize(clean_txt)
        # remove words in stop words
        words = [w for w in txt.split(" ") \
                 if not w in stop_words.ENGLISH_STOP_WORDS and len(w) > 3]
        return len(words)
    except:
        return 0


"""
Pre-processing:  tokenization
Most of the time, the first steps of an NLP project is to "tokenize" your documents, which main purpose is to normalize
our texts. The three fundamental stages will usually include:
    * break the descriptions into sentences and then break the sentences into tokens
    * remove punctuation and stop words
    * lowercase the tokens
    * herein, I will also only consider words that have length equal to or greater than 3 characters
"""


def tokenize(text):
    stop = set(stopwords.words('english'))
    """
    sent_tokenize(): segment text into sentences
    word_tokenize(): break sentences into words
    """
    try:
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex.sub(" ", text)  # remove punctuation
        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w) >= 3]
        return filtered_tokens
    except TypeError as e:
        print(text, e)


def generate_wordcloud(tup):
    wordcloud = WordCloud(background_color='white',
                          max_words=50, max_font_size=40,
                          random_state=42
                          ).generate(str(tup))
    return wordcloud
