from collections import Counter
from operator import itemgetter
import numpy as np
import jieba

class TFIDF():
    def __init__(self, reload_jieba_idf=True):
        self.idf = {}
        self.median_idf = 0
        
        if reload_jieba_idf:
            jieba_idf, median_idf = jieba.analyse.TFIDF().idf_loader.get_idf()  
            
            for word, value in jieba_idf.items():
                self.idf[word] = round(1000000 / value)
            
            self.median_idf = round(1000000 / median_idf)
    
    def partial_fit(self, sentences):
        for sentence in sentences:
            sentence_word_counter = Counter(sentence)
            for word in sentence_word_counter.keys():
                if len(word.strip()) < 2:
                    continue

                if word in self.idf:
                    self.idf[word] += 1
                else:
                    self.idf[word] = self.median_idf + 1
    
    def fit(self, sentences):
        if isinstance(sentences[0], str):
            sentences = [sentences]
        
        self.partial_fit(sentences)
        
        if self.median_idf == 0 and len(self.idf) > 0:
            self.median_idf = np.mean(list(self.idf.values()))
    
    def predict(self, sentences, topK=3, update=False):
        if isinstance(sentences[0], str):
            sentences = [sentences]
        
        result = []
        for sentence in sentences:
            sentence_idf = {}
            sentence_word_counter = Counter(sentence)

            for word in sentence_word_counter.keys():
                if len(word.strip()) < 2:
                    continue

                if word in self.idf:
                    sentence_idf[word] = self.idf[word] / sentence_word_counter[word]
                    if update:
                        self.idf[word] += 1
                else:
                    if update:
                        self.idf[word] = self.median_idf + 1
                        self.idf[word] += 1
                        sentence_idf[word] = self.idf[word] / sentence_word_counter[word]
                    else:
                        sentence_idf[word] = self.median_idf / sentence_word_counter[word]

            sentence_idf = sorted(sentence_idf.items(), key=itemgetter(1), reverse=False)
            sentence_idf = [x[0] for x in sentence_idf]
            result.append(sentence_idf[:topK])
        
        return result
