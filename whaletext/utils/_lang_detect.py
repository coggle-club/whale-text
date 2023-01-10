import re

def lang_detect(s):
    if re.search(r"[\u4e00-\u9FFF]", s):
        return 'zh'
    else:
        return 'en'

def convert_zh2simplified(s):
    # 将繁体字转换为简体字
    pass