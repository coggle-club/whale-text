from scipy import stats

def pearson_corr(arr1, arr2):
    return stats.pearsonr(arr1, arr2)