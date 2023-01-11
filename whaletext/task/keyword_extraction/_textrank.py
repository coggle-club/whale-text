import networkx as nx
from collections import defaultdict
from itertools import combinations

import whaletext

def similarity(s1, s2):
    if not len(s1) or not len(s2):
        return 0.0
    return len(s1.intersection(s2))/(1.0 * (len(s1) + len(s2)))

class TextRank():
    def __init__(self):
        pass
    
    def extract_keywords(self, sentences, topK=3):
        if isinstance(sentences, str):
            sentences = [sentences]

        result = []
        for sentence in sentences:
            if isinstance(sentence, str):
                sentence = whaletext.tokenize.jieba_tokenize(sentence)
            
            cm = defaultdict(int)
            for idx in range(len(sentence)):
                for idy in range(idx, idx + 2):
                    if idy > len(sentence) - 1:
                        break
                        
                    if len(sentence[idx]) < 2:
                        continue

                    if len(sentence[idy]) < 2:
                        continue

                    cm[(sentence[idx], sentence[idy])] += 1

            graph_data = [(key[0], key[1], value) for key,value in cm.items()]

            G = nx.Graph()
            G.add_weighted_edges_from(graph_data)
            
            nodes_with_score = nx.pagerank(G)
            nodes_with_score = sorted(nodes_with_score.items(), key=lambda x:x[1], reverse=True)[:topK]

            result.append([x[0] for x in nodes_with_score])
    
        if len(sentences) == 1:
            return result[0]
        else:
            return result

    def summarization(self, paragraph, topK=3):
        sentences = whaletext.tokenize.sent_tokenize(paragraph)
        words = [set(x) for x in sentences]
        scores = []
        for i in range(len(sentences)):
            for j in range(i, len(sentences)):
                if i == j:
                    continue
                    
                scores.append((i, j, similarity(words[i], words[j])))
        
        g = nx.Graph()
        g.add_weighted_edges_from(scores)
        pr = nx.pagerank(g)
        
        result = sorted(((i, pr[i], s) for i, s in enumerate(sentences) if i in pr), key=lambda x: pr[x[0]], reverse=True)
        result = [x[2] for x in result][:topK]

        return [''.join(x) for x in result]