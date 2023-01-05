from setuptools import setup, find_packages

setup(
    name='whaletext',
    version='0.1',
    description='Simple tools for NLP task',
    author='Yuzhong Liu',
    author_email='finlayliu@qq.com',
    packages = find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'jieba',
        'gensim',
        'emoji',
        'joblib'
    ],
)