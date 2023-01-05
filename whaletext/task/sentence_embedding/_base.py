import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class Word2VecSentenceEmbedding():
    def __init__(self, embedding_model):
        self._embedding_model = embedding_model
        
    def fit(self, sentences):
        pass
    
    def fit_transform(self, sentences):
        pass
    
    def transform(self, sentences):
        pass
    
    def cosine_sim(self, emb1, emb2):
        inn = (emb1 * emb2).sum()
        emb1norm = np.sqrt((emb1 * emb1).sum())
        emb2norm = np.sqrt((emb2 * emb2).sum())
        scores = inn / emb1norm / emb2norm
        return scores
    
class W2vMaxPoolingEmbedding(Word2VecSentenceEmbedding):
    def transform(self, sentences):
        result = []
        for sentence in sentences:
            sentence_feat = self._embedding_model.transform_sentence(sentence)
            sentence_feat = [x for x in sentence_feat if x is not None]
            if len(sentence_feat) == 0:
                result.append(None)
            else:
                sentence_feat = np.array(sentence_feat)
                result.append(sentence_feat.max(0))
        return result
    
    def fit_transform(self, sentences):
        return self.transform(sentences)
    
class W2vMeanPoolingEmbedding(Word2VecSentenceEmbedding):
    def transform(self, sentences):
        result = []
        for sentence in sentences:
            sentence_feat = self._embedding_model.transform_sentence(sentence)
            sentence_feat = [x for x in sentence_feat if x is not None]
            if len(sentence_feat) == 0:
                result.append(None)
            else:
                sentence_feat = np.array(sentence_feat)
                result.append(sentence_feat.mean(0))
        return result
    
    def fit_transform(self, sentences):
        return self.transform(sentences)
    
class W2vIdfPoolingEmbedding(Word2VecSentenceEmbedding):
    def fit(self, sentences):
        tfidf = TfidfVectorizer(tokenizer=str.split, token_pattern=None)
        tfidf.fit([' '.join(x) for x in sentences])
        
        self.idf = {}
        for v, word in zip(tfidf.idf_, tfidf.get_feature_names_out()):
            self.idf[word] = v
    
    def transform(self, sentences):
        if 'idf' not in dir(self):
            raise Exception('Please fit model first')
        
        result = []
        for sentence in sentences:
            sentence_feat = self._embedding_model.transform_sentence(sentence)
            sentence_idf = [self.idf[x] if x in self.idf else None for x in sentence]
            
            sentence_final_feat = None
            for x, y in zip(sentence_feat, sentence_idf):
                if x is None:
                    continue
                if y is None:
                    continue
                    
                if sentence_final_feat is None:
                    sentence_final_feat = x * y
                else:
                    sentence_final_feat += x * y
            
            result.append(sentence_final_feat)
        return result
    
    def fit_transform(self, sentences):
        self.fit(sentences)
        return self.transform(sentences)
    
class W2vSifPoolingEmbedding(Word2VecSentenceEmbedding):
    def __init__(self, embedding_model, base_method='idf'):
        if base_method == 'mean':
            self._base_model = W2vMeanPoolingEmbedding(embedding_model)
        elif base_method == 'max':
            self._base_model = W2vMaxPoolingEmbedding(embedding_model)
        elif base_method == 'idf':
            self._base_model = W2vIdfPoolingEmbedding(embedding_model)
    
    def fit(self, sentences):
        result = self._base_model.fit_transform(sentences)
        result = np.array(result)
        self._remove_pc(result)
    
    def transform(self, sentences):
        result = self._base_model.transform(sentences)
        result = np.array(result)
        return self._remove_pc(result)
    
    def fit_transform(self, sentences):
        self._base_model.fit(sentences)
        return self._base_model.transform(sentences)
    
    def _compute_pc(self, X,npc=1):
        """
        Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: component_[i,:] is the i-th pc
        """
        self._svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        self._svd.fit(X)
        return self._svd.components_

    def _remove_pc(self, X, npc=1):
        """
        Remove the projection on the principal components
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: XX[i, :] is the data point after removing its projection
        """
        if '_svd' not in dir(self):
            pc = self._compute_pc(X, npc)
        else:
            pc = self._svd.components_
        
        if npc==1:
            XX = X - X.dot(pc.transpose()) * pc
        else:
            XX = X - X.dot(pc.transpose()).dot(pc)
        return XX