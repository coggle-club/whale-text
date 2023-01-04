import pandas as pd

def load_waimai():
    '''外卖评论数据集
    '''
    data = pd.read_csv('https://mirror.coggle.club/dataset/waimai_10k.csv')
    return data

def load_lcqmc():
    '''LCQMC文本匹配数据集
    '''
    train = pd.read_csv('https://mirror.coggle.club/dataset/LCQMC.test.data.zip', 
            sep='\t', names=['query1', 'query2', 'label'])

    valid = pd.read_csv('https://mirror.coggle.club/dataset/LCQMC.test.data.zip', 
            sep='\t', names=['query1', 'query2', 'label'])

    test = pd.read_csv('https://mirror.coggle.club/dataset/LCQMC.test.data.zip', 
            sep='\t', names=['query1', 'query2', 'label'])

    return train, valid, test