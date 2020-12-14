from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from scipy.sparse.linalg import norm
from tqdm.notebook import tqdm
tqdm.pandas()

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, preprocess, stopwords):
        self.preprocess = preprocess
        self.stopwords = stopwords
          
    def fit(self, X: pd.DataFrame, y=None):
        return self


    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        print('Preprocessing...')
        X = X.progress_applymap(self.preprocess)
        X = X.progress_applymap(lambda x: self.del_stopwords(x, self.stopwords))
        return X

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.transform(X)
    
    @staticmethod
    def del_stopwords(text: str, stopwords):
        return ' '.join([i for i in text.split() if i not in stopwords if i != ''])
    
    
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, funcs):
        self.funcs = funcs

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        print('Applying functions...')
        for name, func in tqdm(self.funcs):
            X[name] = X.apply(lambda x: func(x.name_1, x.name_2), axis=1)
        return X

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.transform(X)
    
    
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, to_drop: list):
        self.to_drop = to_drop
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        return X.drop(columns=self.to_drop)

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.transform(X)
    
    

class NgramVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, ngram=3, stride=1):
        self.ngram = ngram
        self.stride = stride
        self.vector = CountVectorizer()
        
    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        X[['name_1', 'name_2']] = X[['name_1', 'name_2']].applymap(lambda x: self.get_ngram(x, self.ngram, self.stride))
        self.vector.fit(X.name_1 + ' ' + X.name_2)
        return self
        
    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        X_ngram = X[['name_1', 'name_2']].applymap(lambda x: self.get_ngram(x, self.ngram, self.stride))
        name_1 = self.vector.transform(X_ngram.name_1)
        name_2 = self.vector.transform(X_ngram.name_2)
        X['cos_ngram'] = np.array(name_1.multiply(name_2).sum(1)).ravel() / (norm(name_1, axis=1) * norm(name_2, axis=1) + 1e-8)
        return X

    def fit_transform(self, X: pd.DataFrame, y=None):
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def get_ngram(text, ngram, stride) -> str:
        a = set()
        for i in range(0, len(text), stride):
            spam = text[i:i+ngram].strip()
            if spam and len(spam) == ngram:
                a.add(spam)
        return ' '.join(a)