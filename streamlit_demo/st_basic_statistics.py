import streamlit as st
import whaletext

st.markdown('#### 文本输入')
lines = st.text_area('', 
'''我们爱数据科学！
我们爱计算机视觉，还有Pytorch！
''',)

st.markdown('#### 分析结果')
for line in lines.split('\n'):
    if len(line.strip()) == 0:
        continue

    st.markdown('- ' + line)
    st.code(f'''character_count: {whaletext.statistics.character_count(line)}
emoji_character_count: {whaletext.statistics.emoji_character_count(line)}
duplicates_character_count: {whaletext.statistics.duplicates_character_count(line)}
punctuations_count: {whaletext.statistics.punctuations_count(line)}
chinese_character_count: {whaletext.statistics.chinese_character_count(line)}
whitespaces_count: {whaletext.statistics.whitespaces_count(line)}''')
