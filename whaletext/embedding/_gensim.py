from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

class GensimEmbedding():
    def train(self, sentences):
        self.model.build_vocab(sentences, update=True)
        self.model.train(
            sentences, 
            total_examples=len(sentences),
            epochs=self.model.epochs
        )

    def encode_word(self, word):
        if word not in self.model.wv:
            return None
        else:
            return self.model.wv[word]
        
    def encode_sentences(self, sentences):
        if len(sentences) == 0:
            return []
        
        encode_result = []
        for sentence in sentences:
            encode_result.append([])
            for word in sentence:
                if word not in self.model.wv:
                    encode_result[-1].append(None)
                else:
                    encode_result[-1].append(self.model.wv[word])
        
        return encode_result

    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        pass
    
    def __getitem__(self, word):
        return self.similar_by_word(word)
    
    def similar_by_word(self, word):
        return self.model.wv.similar_by_word(word)

    def similar_by_vector(self, v):
        return self.model.wv.similar_by_vector(v)
    
class Word2VecEmbedding(GensimEmbedding):
    def __init__(self, sentences, vector_size=50, window=5, 
                 alpha=0.05, epochs=5, min_count=1, 
                 workers=1, **kwargs):
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size, 
            alpha=alpha,
            window=window,
            epochs=epochs,
            min_count=min_count,
            workers=workers,
            **kwargs)
        
    def load(self, path):
        self.model = Word2Vec.load(path)
        
class FastTextEmbedding(GensimEmbedding):
    def __init__(self, sentences, vector_size=50, window=5, 
                 alpha=0.05, epochs=5, min_count=1, 
                 workers=1, **kwargs):
        self.model = FastText(
            sentences=sentences,
            vector_size=vector_size, 
            alpha=alpha,
            window=window,
            epochs=epochs,
            min_count=min_count,
            workers=workers,
            **kwargs)
        
    def load(self, path):
        self.model = FastText.load(path)
        
class Doc2VecEmbedding(GensimEmbedding):
    def __init__(self, sentences, vector_size=50, window=5, 
                 alpha=0.05, epochs=5, min_count=1, 
                 workers=1, **kwargs):
        
        documents = [TaggedDocument(doc, [1]) for i, doc in enumerate(sentences)]
        self.model = Doc2Vec(
            documents=documents,
            vector_size=vector_size, 
            alpha=alpha,
            window=window,
            epochs=epochs,
            min_count=min_count,
            workers=workers,
            **kwargs)
    
    def train(self, sentences):
        documents = [TaggedDocument(doc, [1]) for i, doc in enumerate(sentences)]
        self.model.build_vocab(documents, update=True)
        self.model.train(
            documents, 
            total_examples=len(documents),
            epochs=self.model.epochs
        )
    
    def load(self, path):
        self.model = Doc2Vec.load(path)
        
    def infer_sentence(self, sentence):
        return self.model.infer_vector(sentence)