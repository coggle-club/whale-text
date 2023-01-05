import pandas as pd
import os

LOCAL_DATASET_PATH = os.path.expanduser('~/') + '.local/whaletext/'
if not os.path.exists(LOCAL_DATASET_PATH):
    try:
        os.makedirs(LOCAL_DATASET_PATH)
    except:
        pass

def load_waimai():
    '''外卖评论数据集
    '''
    if os.path.exists(LOCAL_DATASET_PATH + 'waimai_10k.csv'):
        data = pd.read_csv(LOCAL_DATASET_PATH + 'waimai_10k.csv')
    else:
        data = pd.read_csv('https://mirror.coggle.club/dataset/waimai_10k.csv')
        data.to_csv(LOCAL_DATASET_PATH + 'waimai_10k.csv', index=None)
    return data

def load_lcqmc():
    '''LCQMC文本匹配数据集
    '''
    if os.path.exists(LOCAL_DATASET_PATH + 'LCQMC.train'):
        train = pd.read_csv(LOCAL_DATASET_PATH + 'LCQMC.train')
        valid = pd.read_csv(LOCAL_DATASET_PATH + 'LCQMC.valid')
        test = pd.read_csv(LOCAL_DATASET_PATH + 'LCQMC.test')
    else:
        train = pd.read_csv('https://mirror.coggle.club/dataset/LCQMC.train.data.zip', 
                sep='\t', names=['query1', 'query2', 'label'])

        valid = pd.read_csv('https://mirror.coggle.club/dataset/LCQMC.valid.data.zip', 
                sep='\t', names=['query1', 'query2', 'label'])

        test = pd.read_csv('https://mirror.coggle.club/dataset/LCQMC.test.data.zip', 
                sep='\t', names=['query1', 'query2', 'label'])

        train.to_csv(LOCAL_DATASET_PATH + 'LCQMC.train', index=None)
        valid.to_csv(LOCAL_DATASET_PATH + 'LCQMC.valid', index=None)
        test.to_csv(LOCAL_DATASET_PATH + 'LCQMC.test', index=None)

    return train, valid, test