import whaletext
import jieba

import numpy as np
def cosine_sim(emb1, emb2):
    inn = (emb1 * emb2).sum()
    emb1norm = np.sqrt((emb1 * emb1).sum())
    emb2norm = np.sqrt((emb2 * emb2).sum())
    scores = inn / emb1norm / emb2norm
    return scores

lcqmc_train, lcqmc_valid, lcqmc_test = whaletext.datasets.load_lcqmc()
sentences1 = [jieba.lcut(x) for x in lcqmc_test['query1'].iloc[:]]
sentences2 = [jieba.lcut(x) for x in lcqmc_test['query2'].iloc[:]]

model = whaletext.embedding.Word2VecEmbedding(sentences=sentences1 + sentences2, vector_size=50)
embeeding_model = whaletext.task.W2vMeanPoolingEmbedding(model)
sentences1_feat = embeeding_model.transform(sentences1)
sentences2_feat = embeeding_model.transform(sentences2)
cosine_sim = [embeeding_model.cosine_sim(x, y) for x, y in zip(sentences1_feat, sentences2_feat)]
lcqmc_test['cosine_sim'] = cosine_sim
whaletext.metrics.pearson_corr(lcqmc_test['cosine_sim'], lcqmc_test['label'])

model = whaletext.embedding.Word2VecEmbedding(sentences=sentences1 + sentences2, vector_size=50)
embeeding_model = whaletext.task.W2vMaxPoolingEmbedding(model)
sentences1_feat = embeeding_model.transform(sentences1)
sentences2_feat = embeeding_model.transform(sentences2)
cosine_sim = [embeeding_model.cosine_sim(x, y) for x, y in zip(sentences1_feat, sentences2_feat)]
lcqmc_test['cosine_sim'] = cosine_sim
whaletext.metrics.pearson_corr(lcqmc_test['cosine_sim'], lcqmc_test['label'])

model = whaletext.embedding.Word2VecEmbedding(sentences=sentences1 + sentences2, vector_size=50)
embeeding_model = whaletext.task.W2vIdfPoolingEmbedding(model)
embeeding_model.fit(sentences1 + sentences2)
sentences1_feat = embeeding_model.transform(sentences1)
sentences2_feat = embeeding_model.transform(sentences2)
cosine_sim = [embeeding_model.cosine_sim(x, y) for x, y in zip(sentences1_feat, sentences2_feat)]
lcqmc_test['cosine_sim'] = cosine_sim
whaletext.metrics.pearson_corr(lcqmc_test['cosine_sim'], lcqmc_test['label'])


model = whaletext.embedding.Word2VecEmbedding(sentences=sentences1 + sentences2, vector_size=50)
embeeding_model = whaletext.task.W2vSifPoolingEmbedding(model, 'idf')
embeeding_model.fit(sentences1 + sentences2)
sentences1_feat = embeeding_model.transform(sentences1)
sentences2_feat = embeeding_model.transform(sentences2)
cosine_sim = [embeeding_model.cosine_sim(x, y) for x, y in zip(sentences1_feat, sentences2_feat)]
lcqmc_test['cosine_sim'] = cosine_sim
whaletext.metrics.pearson_corr(lcqmc_test['cosine_sim'], lcqmc_test['label'])

model = whaletext.embedding.Doc2VecEmbedding(sentences=sentences1 + sentences2, vector_size=50)
sentences1_feat = [model.infer_sentence(x) for x in sentences1]
sentences2_feat = [model.infer_sentence(x) for x in sentences2]
cosine_sim = [cosine_sim(x, y) for x, y in zip(sentences1_feat, sentences2_feat)]
lcqmc_test['cosine_sim'] = cosine_sim
whaletext.metrics.pearson_corr(lcqmc_test['cosine_sim'], lcqmc_test['label'])