import whaletext

def test_similarity():

    s1 = '英雄联盟什么英雄最好'
    s2 = '英雄联盟最好英雄是什么'
    assert whaletext.similarity.longest_substr_length(s1, s2) != None
    assert whaletext.similarity.edit_distance(s1, s2) != None
    assert whaletext.similarity.cosine_distance(s1, s2) != None
    assert whaletext.similarity.hamming_distance(s1, s2) != None
    assert whaletext.similarity.jaccard_distance(s1, s2) != None
    assert whaletext.similarity.prefix_length(s1, s2)  != None