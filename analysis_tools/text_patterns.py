def get_imp(bow,mf,ngram):
    import pandas as pd
    import numpy as np
    # creates BOW
    import sklearn.feature_extraction.text as text

    tfidf=text.CountVectorizer(ngram_range=(ngram,ngram), max_features=mf, stop_words='english')
    matrix=tfidf.fit_transform(bow)
    return pd.Series(np.array(matrix.sum(axis=0))[0],index=tfidf.get_feature_names()).sort_values(ascending=False).head(1000)
