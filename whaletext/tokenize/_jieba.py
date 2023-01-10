import jieba
import jieba.posseg as pseg

def jieba_tokenize(s, return_pos=False):
    result = pseg.lcut(s)
    if not return_pos:
        return [x.word for x in result]
    else:
        return [(x.word, x.flag) for x in result]