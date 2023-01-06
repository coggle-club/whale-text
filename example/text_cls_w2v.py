import jieba
import whaletext
from sklearn.linear_model import LogisticRegression

data = whaletext.datasets.load_waimai()
data = data.sample(5000)

word_text = [jieba.lcut(x) for x in data['review']]
data['text'] = [' '.join(x) for x in word_text]

embeeding_model = whaletext.task.sentence_embedding.W2vMeanPooling(
    whaletext.embedding.Word2VecModel(
        sentences=word_text, vector_size=50
    )
)
model = whaletext.task.classification.MLBasicModel(
    embedding_model = embeeding_model,
    ml_model = LogisticRegression(max_iter=1000),
)

model.fit(word_text[:4000], data['label'].iloc[:4000])
score = model.predict(word_text[4000:]) == data['label'].iloc[4000:]
print('W2V with LogisticRegression', score.mean())