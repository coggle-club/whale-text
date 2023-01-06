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
# lcqmc_test = lcqmc_test.head(100)
sentences1 = [jieba.lcut(x) for x in lcqmc_test['query1'].iloc[:]]
sentences2 = [jieba.lcut(x) for x in lcqmc_test['query2'].iloc[:]]

model = whaletext.embedding.Word2VecModel(sentences=sentences1 + sentences2, vector_size=50)
for sentence_model in [
    whaletext.task.sentence_embedding.W2vMeanPooling,
    whaletext.task.sentence_embedding.W2vMaxPooling,
    whaletext.task.sentence_embedding.W2vIdfPooling,
    whaletext.task.sentence_embedding.W2vSifPooling,
]:
    embeeding_model = sentence_model(model)
    embeeding_model.fit(sentences1 + sentences2)
    sentences1_feat = embeeding_model.transform(sentences1)
    sentences2_feat = embeeding_model.transform(sentences2)
    cosine_result = [embeeding_model.cosine_sim(x, y) for x, y in zip(sentences1_feat, sentences2_feat)]
    lcqmc_test['cosine_sim'] = cosine_result
    score = whaletext.metrics.pearson_corr(lcqmc_test['cosine_sim'], lcqmc_test['label'])
    print(sentence_model, score)

model = whaletext.embedding.Doc2VecModel(sentences=sentences1 + sentences2, vector_size=50)
sentences1_feat = [model.infer_sentence(x) for x in sentences1]
sentences2_feat = [model.infer_sentence(x) for x in sentences2]
cosine_result = [cosine_sim(x, y) for x, y in zip(sentences1_feat, sentences2_feat)]
lcqmc_test['cosine_sim'] = cosine_result
score = whaletext.metrics.pearson_corr(lcqmc_test['cosine_sim'], lcqmc_test['label'])
print('Doc2VecModel', score)
