from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

class BowEmbedding():
    def __init__(self, lowercase=True, stop_words=None, 
                 ngram_range=(1,1), max_features=2000,
                 use_idf=True, token_pattern='(?u)\\b\\w\\w+\\b',
                 **kwargs):
        self._model = TfidfVectorizer(
            lowercase = lowercase,
            stop_words = stop_words,
            ngram_range = ngram_range,
            max_features = max_features,
            use_idf = use_idf,
            token_pattern = token_pattern,
            **kwargs
        )

    def fit(self, sentences):
        self._model.fit(sentences)

    def fit_transform(self, sentences):
        return self._model.fit_transform(sentences)

    def transform(self, sentences):
        return self._model.transform(sentences)

    def save(self, path):
        pass

    def load(self, path):
        pass