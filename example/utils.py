import whaletext

s = '我们今天去吃饭了'
print(s, whaletext.utils._lang_detect(s))

s = 'we go to rest today.'
print(s, whaletext.utils._lang_detect(s))

print(whaletext.utils._check_env(['numpy', 'pandas', 'joblib']))