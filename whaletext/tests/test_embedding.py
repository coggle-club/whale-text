import whaletext
from gensim.test.utils import common_texts


def test_tfidf_embedding():
    model = whaletext.embedding.BowEmbedding()
    model.fit_transform([' '.join(x) for x in common_texts])
    model.key_to_index
    assert model != None

def test_fasttext_embedding():
    model = whaletext.embedding.FastTextEmbedding(sentences=common_texts)
    model['human']
    model.similar_by_word('human')
    model.transform_sentence(['humane', 'system'])
    assert model != None

def test_w2v_embedding():
    model = whaletext.embedding.Word2VecEmbedding(sentences=common_texts)
    model['human']
    model.similar_by_word('human')
    model.transform_sentence(['humane', 'system'])
    assert model != None

def test_doc2vec_embedding():
    model = whaletext.embedding.Doc2VecEmbedding(sentences=common_texts)
    model['human']
    model.similar_by_word('human')
    model.infer_sentence(['human', 'system'])
    assert model != None
