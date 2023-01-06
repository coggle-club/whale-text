import jieba
import jieba.analyse

import whaletext
tfidf = whaletext.task.information_extraction.TFIDF(reload_jieba_idf=True)

result1 = tfidf.predict(jieba.lcut('人工智能是我们的未来，我们要不断学习。'), topK=3)
result2 = jieba.analyse.tfidf('人工智能是我们的未来，我们要不断学习。', topK=3)

print('whaletext (tfidf): ', result1)
print('jieba (tfidf): ', result2)