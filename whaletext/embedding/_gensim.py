from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

class GensimEmbedding():
    def train(self, sentences):
        self._model.build_vocab(sentences, update=True)
        self._model.train(
            sentences, 
            total_examples=len(sentences),
            epochs=self._model.epochs
        )

        self.key_to_index = self._model.wv.key_to_index

    def transform_word(self, word):
        if word not in self._model.wv:
            return None
        else:
            return self._model.wv[word]
        
    def transform_sentence(self, sentence):
        if len(sentence) == 0:
            return []
        
        encode_result = []
        for word in sentence:
            if word not in self._model.wv:
                encode_result.append(None)
            else:
                encode_result.append(self._model.wv[word])
        
        return encode_result

    def save(self, path):
        self._model.save(path)
    
    def load(self, path):
        pass
    
    def __getitem__(self, word):
        return self.transform_word(word)
    
    def similar_by_word(self, word):
        return self._model.wv.similar_by_word(word)

    def similar_by_vector(self, v):
        return self._model.wv.similar_by_vector(v)
    
class Word2VecEmbedding(GensimEmbedding):
    def __init__(self, sentences, vector_size=50, window=5, 
                 alpha=0.05, epochs=5, min_count=1, 
                 workers=1, **kwargs):
        self._model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size, 
            alpha=alpha,
            window=window,
            epochs=epochs,
            min_count=min_count,
            workers=workers,
            **kwargs)

        self.key_to_index = self._model.wv.key_to_index
        
    def load(self, path):
        self._model = Word2Vec.load(path)
        
class FastTextEmbedding(GensimEmbedding):
    def __init__(self, sentences, vector_size=50, window=5, 
                 alpha=0.05, epochs=5, min_count=1, 
                 workers=1, **kwargs):
        self._model = FastText(
            sentences=sentences,
            vector_size=vector_size, 
            alpha=alpha,
            window=window,
            epochs=epochs,
            min_count=min_count,
            workers=workers,
            **kwargs)

        self.key_to_index = self._model.wv.key_to_index

        
    def load(self, path):
        self._model = FastText.load(path)
        
class Doc2VecEmbedding(GensimEmbedding):
    def __init__(self, sentences, vector_size=50, window=5, 
                 alpha=0.05, epochs=5, min_count=1, 
                 workers=1, **kwargs):
        
        documents = [TaggedDocument(doc, [1]) for i, doc in enumerate(sentences)]
        self._model = Doc2Vec(
            documents=documents,
            vector_size=vector_size, 
            alpha=alpha,
            window=window,
            epochs=epochs,
            min_count=min_count,
            workers=workers,
            **kwargs)

        self.key_to_index = self._model.wv.key_to_index
    
    def train(self, sentences):
        documents = [TaggedDocument(doc, [1]) for i, doc in enumerate(sentences)]
        self._model.build_vocab(documents, update=True)
        self._model.train(
            documents, 
            total_examples=len(documents),
            epochs=self._model.epochs
        )
    
    def load(self, path):
        self._model = Doc2Vec.load(path)
        
    def infer_sentence(self, sentence):
        return self._model.infer_vector(sentence)