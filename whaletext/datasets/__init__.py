import pandas as pd
import numpy as np

def load_waimai():
    '''外卖评论数据集
    '''
    data = pd.read_csv('https://mirror.coggle.club/dataset/waimai_10k.csv')
    return data

def load_lcqmc():
    '''LCQMC文本匹配数据集
    '''
    pass