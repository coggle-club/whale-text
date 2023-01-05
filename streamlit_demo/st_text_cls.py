import streamlit as st
import whaletext

import jieba
import whaletext

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

@st.cache(allow_output_mutation=True)
def get_model():
    # 加载数据集
    data = whaletext.datasets.load_waimai()
    data = data.sample(5000)

    # 文本分词
    word_text = [jieba.lcut(x) for x in data['review']]
    data['text'] = [' '.join(x) for x in word_text]

    # BernoulliNB
    model = whaletext.task.MLBasicModel(
        embedding_model = whaletext.embedding.BowEmbedding(tokenizer=str.split, token_pattern=None),
        ml_model = LogisticRegression(),
    )
    model.fit(data['text'].iloc[:], data['label'].iloc[:])
    return model

st.markdown('#### 文本输入')
st.markdown('模型来自外卖文本数据集')
s1 = st.text_input('', '说实话，味道没图片看起来好吃')

st.markdown('#### 分类结果')
model = get_model()

if len(s1.strip()) > 0:
    s1 = ' '.join(jieba.lcut(s1))
    print(s1)
    st.code(f'''模型结果[差评/好评]: {model.predict_proba([s1])[0]}''')