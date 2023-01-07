import whaletext

s = '我们今天去吃饭了'
print(s, whaletext.utils.lang_detect(s))

s = 'we go to rest today.'
print(s, whaletext.utils.lang_detect(s))

print(whaletext.utils.check_env(['numpy', 'pandas', 'joblib']))