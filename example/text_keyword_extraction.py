import jieba
import jieba.analyse

import whaletext

data = whaletext.datasets.load_waimai()

model = whaletext.task.keyword_extraction.TFIDF(reload_jieba_idf=False)
print(data['review'].iloc[0], model.extract_keywords(data['review'].iloc[0]))
print(data['review'].iloc[1], model.extract_keywords(data['review'].iloc[1]))
print(model.summarization('。'.join(data['review'].iloc[:20].values)))

model = whaletext.task.keyword_extraction.RAKE()
print(data['review'].iloc[0], model.extract_keywords(data['review'].iloc[0]))
print(data['review'].iloc[1], model.extract_keywords(data['review'].iloc[1]))
print(model.summarization('。'.join(data['review'].iloc[:20].values)))

model = whaletext.task.keyword_extraction.TextRank()
print(data['review'].iloc[0], model.extract_keywords(data['review'].iloc[0]))
print(data['review'].iloc[1], model.extract_keywords(data['review'].iloc[1]))
print(model.summarization('。'.join(data['review'].iloc[:20].values)))