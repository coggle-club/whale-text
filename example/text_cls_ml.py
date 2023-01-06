import jieba
import whaletext

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

# 加载数据集
data = whaletext.datasets.load_waimai()
data = data.sample(5000)

# 文本分词
word_text = [jieba.lcut(x) for x in data['review']]
data['text'] = [' '.join(x) for x in word_text]

# BernoulliNB
model = whaletext.task.classification.MLBasicModel(
    embedding_model = whaletext.embedding.BoW(tokenizer=str.split, token_pattern=None),
    ml_model = BernoulliNB(),
)
model.fit(data['text'].iloc[:4000], data['label'].iloc[:4000])
score = model.predict(data['text'].iloc[4000:]) == data['label'].iloc[4000:]
print('BernoulliNB', score.mean())

# LogisticRegression
model = whaletext.task.classification.MLBasicModel(
    embedding_model = whaletext.embedding.BoW(tokenizer=str.split, token_pattern=None),
    ml_model = LogisticRegression(),
)
model.fit(data['text'].iloc[:4000], data['label'].iloc[:4000])
score = model.predict(data['text'].iloc[4000:]) == data['label'].iloc[4000:]
print('LogisticRegression', score.mean())

# LinearSVC
model = whaletext.task.classification.MLBasicModel(
    embedding_model = whaletext.embedding.BoW(tokenizer=str.split, token_pattern=None),
    ml_model = LinearSVC(),
)
model.fit(data['text'].iloc[:4000], data['label'].iloc[:4000])
score = model.predict(data['text'].iloc[4000:]) == data['label'].iloc[4000:]
print('LinearSVC', score.mean())