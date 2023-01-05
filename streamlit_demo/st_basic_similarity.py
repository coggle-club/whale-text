import streamlit as st
import whaletext

st.markdown('#### 文本输入')
s1 = st.text_input('句子1', '我们要学习计算机视觉。')
s2 = st.text_input('句子2', '我们要学习自然语言处理。')

st.markdown('#### 分析结果')
st.code(f'''edit_distance: {whaletext.similarity.edit_distance(s1, s2)}
cosine_distance: {whaletext.similarity.cosine_distance(s1, s2)}
hamming_distance: {whaletext.similarity.hamming_distance(s1, s2)}
jaccard_distance: {whaletext.similarity.jaccard_distance(s1, s2)}
longest_substr_length: {whaletext.similarity.longest_substr_length(s1, s2)}
prefix_length: {whaletext.similarity.prefix_length(s1, s2)}''')