# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import time

''' 封装TF-IDF '''
def tfidf(train, test, mode="other", params_tfidf=None, n_components=128):
    ##### 处理空值 #####
    train["comment_text"].fillna("unknown", inplace=True)
    test["comment_text"].fillna("unknown", inplace=True)
    merge = pd.concat([train.iloc[:,0:2],test.iloc[:,0:2]])
    df = merge.reset_index(drop=True)
    clean_corpus = df.comment_text
    
    ##### 确认模式 #####
    if mode == "other":
        #初始化一套参数，然后用自定义的参数去替换更改后的
        params = {
            "min_df":100, "max_features":100000, 
            "strip_accents":'unicode', "analyzer":'word', "ngram_range":(1,1),
            "use_idf":1, "smooth_idf":1, "sublinear_tf":1,
            "stop_words":'english'
            }
        for item, value in params_tfidf.items():
            params[item] = params_tfidf[item]
    else: #mode = "unigrams"/"bigrams"/"charngrams"
        ''' 内置3套参数 '''
        if mode == "unigrams": #单个词
            params = {
                "min_df":100, "max_features":100000, 
                "strip_accents":'unicode', "analyzer":'word', "ngram_range":(1,1),
                "use_idf":1, "smooth_idf":1, "sublinear_tf":1,
                "stop_words":'english'
                }
        elif mode == "bigrams": #两个词
            params = {
                "min_df":100, "max_features":30000, 
                "strip_accents":'unicode', "analyzer":'word', "ngram_range":(2,2),
                "use_idf":1, "smooth_idf":1, "sublinear_tf":1,
                "stop_words":'english'
                }                
        elif mode == "charngrams": #长度为4的字符
            params = {
                "min_df":100, "max_features":30000, 
                "strip_accents":'unicode', "analyzer":'char', "ngram_range":(1,4),
                "use_idf":1, "smooth_idf":1, "sublinear_tf":1,
                "stop_words":'english'
                }                
        else:
            print("mode error...")
            return
            
    #获取tfidf后的稀疏矩阵sparse
    train_tfidf, test_tfidf, features_tfidf = getTfidfVector(clean_corpus, #之后的参数都是TfidfVectorizer()的参数
            min_df=params["min_df"], max_features=params["max_features"], 
            strip_accents=params["strip_accents"], analyzer=params["analyzer"], ngram_range=params["ngram_range"],
            use_idf=params["use_idf"], smooth_idf=params["smooth_idf"], sublinear_tf=params["sublinear_tf"],
            stop_words=params["stop_words"])
    #获取pca后的np      
    pca_train_tfidf = pca_compression(train_tfidf, n_components=n_components)
    pca_test_tfidf = pca_compression(test_tfidf, n_components=n_components)
    #获取添加特征名后的pd
    n = params["ngram_range"][0] #生成特征列名时的n的值
    pd_pca_train_tfidf = pd.DataFrame(pca_train_tfidf, columns=["tfidf"+str(n)+"gram"+str(x) for x in range(1, n_components+1)])
    pd_pca_test_tfidf = pd.DataFrame(pca_test_tfidf, columns=["tfidf"+str(n)+"gram"+str(x) for x in range(1, n_components+1)])
    #保存成csv
#    pd_pca_train_tfidf.to_csv("pd_pca_train_tfidf.csv")
#    pd_pca_test_tfidf.to_csv("pd_pca_test_tfidf.csv")
    return pd_pca_train_tfidf, pd_pca_test_tfidf
    
''' TF-IDF Vectorizer '''
def getTfidfVector(clean_corpus, #之后的参数都是TfidfVectorizer()的参数
            min_df=100, max_features=100000, 
            strip_accents='unicode', analyzer='word',ngram_range=(1,1),
            use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english'):
    
    tfv = TfidfVectorizer(min_df=min_df, max_features=max_features, 
            strip_accents=strip_accents, analyzer=analyzer, ngram_range=ngram_range,
            use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf,
            stop_words = stop_words)
    tfv.fit(clean_corpus)
    features_tfidf = np.array(tfv.get_feature_names())
    train_tfidf =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
    test_tfidf = tfv.transform(clean_corpus.iloc[train.shape[0]:])
    return train_tfidf, test_tfidf, features_tfidf

''' PCA降维（默认128维） '''
def pca_compression(train_tfidf, n_components):
    np_train_tfidf = train_tfidf.toarray()
    pca = PCA(n_components=n_components)
    pca_train_tfidf = pca.fit_transform(np_train_tfidf)
    return pca_train_tfidf
    
#单元测试
if __name__ == "__main__":
    ##### 导入数据 #####
#    start_time = time.time()
#    print("Load data start")
    train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
    test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")
#    print("Load data ok...", time.time()-start_time)
    ##### tfidf #####
#    print("tfidf start", time.time()-start_time)
    tfidf(train, test, mode="other", params_tfidf={"ngram_range":(2,2)}, n_components=128)
#    print("End...", time.time()-start_time)