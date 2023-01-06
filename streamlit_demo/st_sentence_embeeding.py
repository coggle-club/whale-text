import whaletext
import streamlit as st
import jieba

import numpy as np
def cosine_sim(emb1, emb2):
    inn = (emb1 * emb2).sum()
    emb1norm = np.sqrt((emb1 * emb1).sum())
    emb2norm = np.sqrt((emb2 * emb2).sum())
    scores = inn / emb1norm / emb2norm
    return scores

@st.cache(allow_output_mutation=True)
def get_model():
    lcqmc_train, lcqmc_valid, lcqmc_test = whaletext.datasets.load_lcqmc()
    sentences1 = [jieba.lcut(x) for x in lcqmc_test['query1'].iloc[:]]
    sentences2 = [jieba.lcut(x) for x in lcqmc_test['query2'].iloc[:]]

    model = whaletext.embedding.Word2VecModel(sentences=sentences1 + sentences2, vector_size=25)
    mean_pooling_model = whaletext.task.sentence_embedding.W2vMeanPooling(model)
    mean_pooling_model.fit(sentences1 + sentences2)

    max_pooling_model = whaletext.task.sentence_embedding.W2vMaxPooling(model)
    max_pooling_model.fit(sentences1 + sentences2)

    idf_pooling_model = whaletext.task.sentence_embedding.W2vIdfPooling(model)
    idf_pooling_model.fit(sentences1 + sentences2)

    sif_pooling_model = whaletext.task.sentence_embedding.W2vSifPooling(model)
    sif_pooling_model.fit(sentences1 + sentences2)

    return model, mean_pooling_model, max_pooling_model, idf_pooling_model, sif_pooling_model

st.markdown('#### 文本输入')
s1 = st.text_input('句子1', '明天天气怎么样？')
s2 = st.text_input('句子2', '明天天气不会很好')

model, mean_pooling_model, max_pooling_model, idf_pooling_model, sif_pooling_model = get_model()

if s1.strip() and s2.strip():
    if st.button('确认'):
        sentences1_feat = mean_pooling_model.transform([s1])[0]
        sentences2_feat = mean_pooling_model.transform([s2])[0]
        mean_cosine_sim = cosine_sim(sentences1_feat, sentences2_feat)

        sentences1_feat = max_pooling_model.transform([s1])[0]
        sentences2_feat = max_pooling_model.transform([s2])[0]
        max_cosine_sim = cosine_sim(sentences1_feat, sentences2_feat)

        sentences1_feat = idf_pooling_model.transform([s1])[0]
        sentences2_feat = idf_pooling_model.transform([s2])[0]
        idf_cosine_sim = cosine_sim(sentences1_feat, sentences2_feat)

        sentences1_feat = idf_pooling_model.transform([s1])[0]
        sentences2_feat = idf_pooling_model.transform([s2])[0]
        sif_cosine_sim = cosine_sim(sentences1_feat, sentences2_feat)

        st.markdown('#### 分析结果')
        st.code(f'''edit_distance: {whaletext.similarity.edit_distance(s1, s2)}
cosine_distance: {whaletext.similarity.cosine_distance(s1, s2)}
hamming_distance: {whaletext.similarity.hamming_distance(s1, s2)}
jaccard_distance: {whaletext.similarity.jaccard_distance(s1, s2)}
longest_substr_length: {whaletext.similarity.longest_substr_length(s1, s2)}
prefix_length: {whaletext.similarity.prefix_length(s1, s2)}

MeanPooling Cosine相似度: {mean_cosine_sim}
MaxPooling Cosine相似度: {max_cosine_sim}
IdfPooling Cosine相似度: {idf_cosine_sim}
SifPooling Cosine相似度: {sif_cosine_sim}''')