import re
import string
import csv
import numpy as np
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


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


PATH = "../input/"
train2 = pd.read_csv(f'{PATH}train2.tsv', quoting=csv.QUOTE_NONE, sep='\t', encoding='utf-8')
test2 = pd.read_table(f'{PATH}test2.tsv', sep='\t', encoding='utf-8')
print(train2.dtypes)
print(test2.dtypes)

train2 = train2[pd.notnull(train2['item_description'])]

vectorizer = TfidfVectorizer(min_df=10,
                             max_features=180000,
                             tokenizer=tokenize,
                             ngram_range=(1, 2))
all_desc = np.append(train2['item_description'].values, test2['item_description'].values)

vz = vectorizer.fit_transform(all_desc.tolist())
print(vz.shape[0])
print(vz.shape[1])
#  create a dictionary mapping the tokens to their tfidf values
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
                    dict(tfidf), orient='index')
tfidf.columns = ['tfidf']

print(tfidf.sort_values(by=['tfidf'], ascending=True).head(10))
print(tfidf.sort_values(by=['tfidf'], ascending=False).head(10))