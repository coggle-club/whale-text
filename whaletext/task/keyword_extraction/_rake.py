import networkx as nx
import numpy as np
import operator
from collections import Counter

import whaletext

class RAKE():
    def __init__(self):
        pass
    
    def _calculate_graph_weight(self, sentences):    
        sentences_words = []
        
        word_counter = Counter()
        g = nx.Graph()
        
        for sentence in sentences:
            if isinstance(sentence, str):
                words = whaletext.tokenize.jieba_tokenize(sentence)
                sentences_words.append(words)
                
                nword = len(words)
                word_counter.update(words)

                word_items = []
                for i in range(nword):
                    for j in range(max(i-5, 0), min(i+5, nword)):
                        if i == j:
                            continue
                        
                        word_items.append([words[i], words[j]])
                g.add_edges_from(word_items)
             
        word_weight = {}
        for word in word_counter:
            word_weight[word] = g.degree(word) / word_counter[word]
        
        return sentences_words, word_weight
    
    def extract_keywords(self, sentences, topK=3, update=False):
        if isinstance(sentences, str):
            sentences = [sentences]
        
        sentences_words, word_weight = self._calculate_graph_weight(sentences)
        
        result = []
        for words in sentences_words:
            word_weight_single = {word:word_weight[word] for word in words if word in word_weight}
            sorted_x = sorted(word_weight_single.items(), key=operator.itemgetter(1), reverse=True)
            sorted_x = [x[0] for x in sorted_x if len(x[0]) > 1][:topK]
            result.append(sorted_x)
    
        if len(sentences) == 1:
            return result[0]
        else:
            return result
    
    def summarization(self, paragraph, topK=3):
        sentences = whaletext.tokenize.sent_tokenize(paragraph)
        sentences_words, word_weight = self._calculate_graph_weight(sentences)

        setences_weight = []
        for words in sentences_words:
            setences_weight.append(np.mean([
                word_weight[word] for word in words if word in word_weight
            ]))
            
        setences_idf_idx = np.argsort(setences_weight)[::-1]
        
        if topK > len(setences_idf_idx):   
            return [sentences[x] for x in setences_idf_idx[:]]
        else:
            return [sentences[x] for x in setences_idf_idx[:topK]]