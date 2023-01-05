# whale-text

#### 项目介绍

whale-text包含NLP解决方案、NLP基础技术、解决方案和模型：

- 提供基础NLP解决方案，如文本相似度计算、句子无监督编码和文本关键词挖掘
- 提供工业落地的解决方法，如文本检索、文本分类和文本匹配
- 提供可以展示可视化的NLP算法Demo，支持将算法进行部署和打包；

#### 目录说明

```
/whaletext/ # 源代码目录
    /datasets/                      # 加载和定义数据✅
    /metrics/                       # 评价指标
    /embedding/                     # 加载和训练词向量✅
    /models/                        # 定义NLP模型
    /similarity/                    # 文本相似度计算✅
    /augmentation/                  # 文本数据增强
    /task/  
	    /sentence_embedding/        # NLP下游任务：句子嵌入编码✅
	    /retrieval/                 # NLP下游任务：文本检索（布尔检索、反向索引）
	    /classification/            # NLP下游任务：文本分类
	    /matching/                  # NLP下游任务：文本匹配
	    /keyword_extraction/        # NLP下游任务：关键词抽取
	    /entity_recognition/        # NLP下游任务：实体抽取
	    /relation_extraction/       # NLP下游任务：关系抽取
	    /summarization/             # NLP下游任务：文本摘要
	    /translation/               # NLP下游任务：文本翻译
	    /error_correction           # NLP下游任务：文本纠错
	    /question_answering         # NLP下游任务：文本问答
	/deploy/                        # 模型Demo和打包部署
/benchmarks/                        # 模型速度和精度对比
/doc/  
/requirements.txt   
README.md
/setup.py
```

#### 安装方法

```
pip install 
```

#### 参与贡献

1. 欢迎各位参与贡献，项目的逻辑如下：确定任务数据集、评价指标、模型，逐步添加细节。
2. 模型方法优先考虑无监督方法 和 落地性较强的模型。
3. 请撰写单元测试 `pytest --cov ./whale-text/`
