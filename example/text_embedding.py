import whaletext
from gensim.test.utils import common_texts

# TFIDF编码
model = whaletext.embedding.BowEmbedding()
model.fit_transform([' '.join(x) for x in common_texts])
model.key_to_index

# FastText编码
model = whaletext.embedding.FastTextEmbedding(sentences=common_texts)
model['human']
model.similar_by_word('human')
model.transform_sentence(['humane', 'system'])
model.key_to_index

# Word2Vec编码
model = whaletext.embedding.Word2VecEmbedding(sentences=common_texts)
model['human']
model.similar_by_word('human')
model.transform_sentence(['humane', 'system'])
model.key_to_index

# Doc2Vec编码
model = whaletext.embedding.Doc2VecEmbedding(sentences=common_texts)
model['human']
model.similar_by_word('human')
model.infer_sentence(['human', 'system'])
model.key_to_index