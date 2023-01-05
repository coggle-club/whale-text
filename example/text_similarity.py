import whaletext

# 句子相似度计算
s1 = '英雄联盟什么英雄最好'
s2 = '英雄联盟最好英雄是什么'
print(whaletext.similarity.longest_substr_length(s1, s2))
print(whaletext.similarity.edit_distance(s1, s2))
print(whaletext.similarity.cosine_distance(s1, s2))
print(whaletext.similarity.hamming_distance(s1, s2))
print(whaletext.similarity.jaccard_distance(s1, s2))
print(whaletext.similarity.prefix_length(s1, s2))