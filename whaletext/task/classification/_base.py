class MLBasicModel():
    def __init__(self, embedding_model, ml_model):
        self.embedding_model = embedding_model
        self.ml_model = ml_model

    def fit(self, sentences, labels):
        feats = self.embedding_model.fit_transform(sentences)
        self.ml_model.fit(feats, labels)

    def predict(self, sentences):
        feats = self.embedding_model.transform(sentences)
        return self.ml_model.predict(feats)

    def predict_proba(self, sentences):
        feats = self.embedding_model.transform(sentences)
        return self.ml_model.predict_proba(feats)

    def dump(self, path):
        dump((self.embedding_model, self.ml_model), path)
        
    def load(self, path):
        self.embedding_model, self.ml_model = load(path)