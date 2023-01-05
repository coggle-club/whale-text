import emoji
import string
import re
import jieba

def sentence_length(s):
    return len(s)

def character_count(s):
    return len(set(s))

def whitespaces_count(s):
    return len([x for x in s if x == ' '])

def duplicates_character_count(s):
    count = 0
    for idx, c in enumerate(s):
        if c in s[:idx]:
            count += 1
            
    return count

def emoji_character_count(s):
    return len([c for c in s if c in emoji.EMOJI_DATA])

def english_character_count(s):
    return len([c for c in s if c in string.ascii_letters])

chinese_re = re.compile(r'[\u4e00-\u9fff]+')
def chinese_character_count(s):
    result = chinese_re.findall(s)
    return len(''.join(result))

english_punctuation = string.punctuation
chinese_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
def punctuations_count(s):
    return len([c for c in s if c in english_punctuation or c in chinese_punctuation])

def word_count(s):
    return len(jieba.lcut(s))

def duplicates_word_count(s):
    word = jieba.lcut(s)
    
    count = 0
    for idx, c in enumerate(word):
        if c in english_punctuation or c in chinese_punctuation:
            continue

        if c in s[:idx]:
            count += 1
            
    return count