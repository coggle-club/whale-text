from collections import Counter
from operator import itemgetter
import numpy as np

import jieba
import jieba.analyse

import whaletext

class TFIDF():
    def __init__(self, reload_jieba_idf=True):
        self.idf = {}
        self.median_idf = 0
        
        if reload_jieba_idf:
            jieba_idf, median_idf = jieba.analyse.TFIDF().idf_loader.get_idf()  
            
            for word, value in jieba_idf.items():
                self.idf[word] = 1000000000 / value
            
            self.median_idf = 1000000000 / median_idf
    
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
    
    def extract_keywords(self, sentences, topK=3, update=False):
        result = []
        
        if isinstance(sentences, str):
            sentences = [sentences]
        
        for sentence in sentences:
            if isinstance(sentence, str):
                sentence = whaletext.tokenize.jieba_tokenize(sentence)
                
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
        
        if len(sentences) == 1:
            return result[0]
        else:
            return result
    
    def summarization(self, paragraph, topK=3, update=False):
        setences = whaletext.tokenize.sent_tokenize(paragraph)
        setences_idf = []
        
        for setence in setences:
            words = whaletext.tokenize.jieba_tokenize(setence)
            words_idf = [
                self.idf[word] for word in words if word in self.idf
            ]
            
            if len(words_idf) == 0:
                setences_idf.append(self.median_idf)
            else:
                setences_idf.append(np.mean(words_idf))
            
        setences_idf_idx = np.argsort(setences_idf)[::-1]
        
        if topK > len(setences_idf_idx):   
            return [setences[x] for x in setences_idf_idx[:]]
        else:
            return [setences[x] for x in setences_idf_idx[:topK]]