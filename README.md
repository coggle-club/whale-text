# whale-text

#### 项目介绍

whale-text包含NLP解决方案、NLP基础技术、解决方案和模型：

- 支持NLP基础统计功能，如文本相似度计算、句子编码和文本关键词挖掘；
- 支持文本词向量训练、BERT模型使用、文本编码和检索等进阶功能；
- 支持可以展示可视化的NLP算法Demo，支持将算法进行部署和打包；


#### 安装方法

建议Python3.6+环境（Linux和Mac支持较好），安装命令：

```
pip3 install git+https://gitee.com/coggle/whale-text -U
```

#### 使用案例

完整使用案例可以参考 `example`文件夹。

- 文本基础统计

```python
import whaletext

whaletext.statistics.character_count('我们学习数据科学，我们学习Python😆')
whaletext.statistics.chinese_character_count('我们学习数据科学，我们学习Python😆')
```

- 句子相似度计算

```python
import whaletext

s1 = '英雄联盟什么英雄最好'
s2 = '英雄联盟最好英雄是什么'
print(whaletext.similarity.longest_substr_length(s1, s2))
print(whaletext.similarity.edit_distance(s1, s2))
```

- 词向量训练

```python
import whaletext
from gensim.test.utils import common_texts

model = whaletext.embedding.Word2VecEmbedding(sentences=common_texts)
model['human']
model.similar_by_word('human')
model.transform_sentence(['humane', 'system'])
model.key_to_index
```

- 文本分类

```python
import jieba
import whaletext

from sklearn.linear_model import LogisticRegression

# 加载数据集
data = whaletext.datasets.load_waimai()
data = data.sample(5000)

# 文本分词
word_text = [jieba.lcut(x) for x in data['review']]
data['text'] = [' '.join(x) for x in word_text]

# BernoulliNB
model = whaletext.task.MLBasicModel(
    embedding_model = whaletext.embedding.BowEmbedding(tokenizer=str.split, token_pattern=None),
    ml_model = BernoulliNB(),
)
model.fit(data['text'].iloc[:4000], data['label'].iloc[:4000])
score = model.predict(data['text'].iloc[4000:]) == data['label'].iloc[4000:]
print('BernoulliNB', score.mean())
```

#### 项目架构


项目开发计划：[https://shimo.im/sheets/NJkbE9vMBmTvn0qR/Rnoja/](https://shimo.im/sheets/NJkbE9vMBmTvn0qR/Rnoja/)

![](https://cdn.coggle.club/img/whale-text.jpg)

#### 常见问题

1. 项目是用来替换现有的NLP库的吗？

> 不是，我们主要是提供常见的解决方案，并不是替代NLP基础库。

2. 项目包含深度学习代码吗？需要GPU吗？

> 部分模型包含，但我们会优先添加无监督或基础模型。

3. 项目中模型有自动调参功能吗？

> 没有，不包含模型自动调参功能。

#### 参与贡献

1. 欢迎各位参与贡献，开发逻辑：数据集、评价指标、模型。
2. 模型方法优先考虑无监督方法 和 落地性较强的模型。
3. 请撰写使用代码和案例。