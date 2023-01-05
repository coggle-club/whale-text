import whaletext

def test_statistics():
    assert whaletext.statistics.character_count('我们学习数据科学，我们学习Python😆') != None
    assert whaletext.statistics.chinese_character_count('我们学习数据科学，我们学习Python😆') != None
    assert whaletext.statistics.duplicates_character_count('我们学习数据科学，我们学习Python😆') != None
    assert whaletext.statistics.duplicates_word_count('我们学习数据科学，我们学习Python😆') != None
    assert whaletext.statistics.emoji_character_count('我们学习数据科学，我们学习Python😆') != None
    assert whaletext.statistics.english_character_count('我们学习数据科学，我们学习Python😆') != None
    assert whaletext.statistics.punctuations_count('我们学习数据科学，我们学习Python😆') != None
    assert whaletext.statistics.sentence_length('我们学习数据科学，我们学习Python😆') != None
    assert whaletext.statistics.whitespaces_count('我们学习数据科学，我们学习Python😆') != None
    assert whaletext.statistics.word_count('我们学习数据科学，我们学习Python😆') != None