# -*- coding: utf-8 -*-
# 复制所有代码，精简到最简版
##### 导入包 #####
#basics
import pandas as pd 
import numpy as np
#misc
import gc
import time
import warnings
from scipy.misc import imread
#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import matplotlib_venn as venn
#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
import spacy
from scipy.sparse import csr_matrix, hstack
from scipy import sparse, io
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from scipy import sparse
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   
#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
#ReferenceData
from Ref_Data import blocked_ips
from Ref_Data import APPO

#settings
start_time = time.time()
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()

##### 特征工程 #####
def makeIndirectFeatures(train, test, df):
    ## Indirect features——共11个特征
    #9个：单词句子数、数、非重复单词数、字母数、标点数、大写字母的单词/字母数、标题数、停顿词数、单词平均长度
    df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
    df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))
    df['count_unique_word']=df["comment_text"].apply(lambda x: len(set(str(x).split())))
    df['count_letters']=df["comment_text"].apply(lambda x: len(str(x)))
    df["count_punctuations"] =df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df["count_stopwords"] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    
    #derived features
    #2个：非重复词占比、标点占比
    df['word_unique_percent']=df['count_unique_word']*100/df['count_word']
    df['punct_percent']=df['count_punctuations']*100/df['count_word']
    
    #serperate train and test features
    train_feats=df.iloc[0:len(train),]
#    test_feats=df.iloc[len(train):,]
    #join the tags
    train_tags=train.iloc[:,2:]
    train_feats=pd.concat([train_feats,train_tags],axis=1)
    
    # 限定值的范围
    train_feats['count_sent'].loc[train_feats['count_sent']>10] = 10 
    train_feats['count_word'].loc[train_feats['count_word']>200] = 200
    train_feats['count_unique_word'].loc[train_feats['count_unique_word']>200] = 200
    return train_feats, train_tags
    
def makeLeakyFeatures(train, test, df):
    ## Leaky features——共8个特征
    df['ip']=df["comment_text"].apply(lambda x: re.findall("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}",str(x)))
    df['count_ip']=df["ip"].apply(lambda x: len(x))
    df['link']=df["comment_text"].apply(lambda x: re.findall("http://.*com",str(x)))
    df['count_links']=df["link"].apply(lambda x: len(x))
    df['article_id']=df["comment_text"].apply(lambda x: re.findall("\d:\d\d\s{0,5}$",str(x)))
    df['article_id_flag']=df.article_id.apply(lambda x: len(x))
    df['username']=df["comment_text"].apply(lambda x: re.findall("\[\[User(.*)\|",str(x)))
    df['count_usernames']=df["username"].apply(lambda x: len(x))
    #check if features are created
    #df.username[df.count_usernames>0]
    
    leaky_feats = df[["ip","link","article_id","username","count_ip","count_links","count_usernames","article_id_flag"]]
    leaky_feats_train = leaky_feats.iloc[:train.shape[0]]
#    leaky_feats_test=leaky_feats.iloc[train.shape[0]:]
    return leaky_feats_train

    
def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    #小写化：Hi与hi等同
    comment=comment.lower()
    #去除\n
    comment=re.sub("\\n","",comment)
    #去除IP
    comment=re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}","",comment)
    #去除usernames
    comment=re.sub("\[\[.*\]","",comment)
    
    #分离句子为单词
    words=tokenizer.tokenize(comment)
    
    # 省略词替换（参考APPO、nltk）：you're -> you are  
    words=[APPO[word] if word in APPO else word for word in words]
    words=[lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]
    
    clean_sent=" ".join(words)
    # remove any non alphanum,digit character
    #clean_sent=re.sub("\W+"," ",clean_sent)
    #clean_sent=re.sub("  "," ",clean_sent)
    return(clean_sent)
	
#corpus.iloc[12235]
#clean(corpus.iloc[12235])

def getTfidfVector(clean_corpus):
    '''
    TF-IDF Vectorizer
    '''
    ### 单个词 ###
    tfv = TfidfVectorizer(min_df=100,  max_features=100000, 
                strip_accents='unicode', analyzer='word',ngram_range=(1,1),
                use_idf=1,smooth_idf=1,sublinear_tf=1,
                stop_words = 'english')
    tfv.fit(clean_corpus)
#    features = np.array(tfv.get_feature_names())
    
    train_unigrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
#    test_unigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])
    
    print("total time till unigrams",time.time()-start_time)
    
    ### 两个词 ###
    tfv = TfidfVectorizer(min_df=100,  max_features=30000, 
                strip_accents='unicode', analyzer='word',ngram_range=(2,2),
                use_idf=1,smooth_idf=1,sublinear_tf=1,
                stop_words = 'english')
    
    tfv.fit(clean_corpus)
#    features = np.array(tfv.get_feature_names())
    train_bigrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
#    test_bigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])
    
    print("total time till bigrams",time.time()-start_time)
    
    ### 长度为4的字符 ###
    tfv = TfidfVectorizer(min_df=100,  max_features=30000, 
                strip_accents='unicode', analyzer='char',ngram_range=(1,4),
                use_idf=1,smooth_idf=1,sublinear_tf=1,
                stop_words = 'english')
    
    tfv.fit(clean_corpus)
#    features = np.array(tfv.get_feature_names())
    train_charngrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
#    test_charngrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])
    
    print("total time till charngrams",time.time()-start_time)
    
    return train_bigrams,train_charngrams,train_unigrams
   
##### 9-Baseline Model #####
#Credis to AlexSanchez https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb#261316
#custom NB model
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self
    
#Just the indirect features -- meta features
def train1(target_x, target_y):
	print("Using only Indirect features")
	model = LogisticRegression(C=3)
	X_train, X_valid, y_train, y_valid = train_test_split(target_x, target_y, test_size=0.33, random_state=2018)
	train_loss = []
	valid_loss = []
	importance=[]
	preds_train = np.zeros((X_train.shape[0], len(y_train)))
	preds_valid = np.zeros((X_valid.shape[0], len(y_valid)))
	for i, j in enumerate(TARGET_COLS):
		print('Class:= '+j)
		model.fit(X_train,y_train[j])
		preds_valid[:,i] = model.predict_proba(X_valid)[:,1]
		preds_train[:,i] = model.predict_proba(X_train)[:,1]
		train_loss_class=log_loss(y_train[j],preds_train[:,i])
		valid_loss_class=log_loss(y_valid[j],preds_valid[:,i])
		print('Trainloss=log loss:', train_loss_class)
		print('Validloss=log loss:', valid_loss_class)
		importance.append(model.coef_)
		train_loss.append(train_loss_class)
		valid_loss.append(valid_loss_class)
	print('mean column-wise log loss:Train dataset', np.mean(train_loss))
	print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))

	print("total time till Indirect feat model",time.time()-start_time)

#Using all direct features
def train2(target_x, target_y):
	print("Using all features except leaky ones")
	model = NbSvmClassifier(C=4, dual=True, n_jobs=-1)
	X_train, X_valid, y_train, y_valid = train_test_split(target_x, target_y, test_size=0.33, random_state=2018)
	train_loss = []
	valid_loss = []
	preds_train = np.zeros((X_train.shape[0], len(y_train)))
	preds_valid = np.zeros((X_valid.shape[0], len(y_valid)))
	for i, j in enumerate(TARGET_COLS):
		print('Class:= '+j)
		model.fit(X_train,y_train[j])
		preds_valid[:,i] = model.predict_proba(X_valid)[:,1]
		preds_train[:,i] = model.predict_proba(X_train)[:,1]
		train_loss_class=log_loss(y_train[j],preds_train[:,i])
		valid_loss_class=log_loss(y_valid[j],preds_valid[:,i])
		print('Trainloss=log loss:', train_loss_class)
		print('Validloss=log loss:', valid_loss_class)
		train_loss.append(train_loss_class)
		valid_loss.append(valid_loss_class)
	print('mean column-wise log loss:Train dataset', np.mean(train_loss))
	print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))

	print("total time till NB base model creation",time.time()-start_time)

##### 根据数据，保存特征 #####
if __name__ == "__main__1":
    from time import clock
    tic = clock()
    ##### 导入数据 #####
    train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
    test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")
    ##### 处理空值 #####
    train["comment_text"].fillna("unknown", inplace=True)
    test["comment_text"].fillna("unknown", inplace=True)
    merge = pd.concat([train.iloc[:,0:2],test.iloc[:,0:2]])
    df = merge.reset_index(drop=True)
    ##### 特征工程1：Indirect features #####
    train_IndirectFeats, train_tags = makeIndirectFeatures(train, test, df) ##### 添加的 #####
    ##### 特征工程2：Leaky features #####
    train_LeakyFeats = makeLeakyFeatures(train, test, df)
    
    ##### 清洗语料 #####
    corpus = merge.comment_text
    clean_corpus = corpus.apply(lambda x :clean(x))
    print("total time till Cleaning",time.time()-start_time)
    ##### 特征工程3：Direct features #####
    train_bigrams, train_charngrams, train_unigrams = getTfidfVector(clean_corpus) ##### 添加的 #####
    ##### 保存特征 #####
    train_IndirectFeats.to_csv("train_IndirectFeats.csv")
    train_LeakyFeats.to_csv("train_LeakyFeats.csv")
    train_tags.to_csv("train_tags.csv")
    io.mmwrite("train_bigrams.mtx", train_bigrams)
    io.mmwrite("train_charngrams.mtx", train_charngrams)
    io.mmwrite("train_unigrams.mtx", train_unigrams)
    '''
    ##### 训练模型1：逻辑回归 #####
    SELECTED_COLS_IndirectFeats=['count_sent', 'count_word', 'count_unique_word',
           'count_letters', 'count_punctuations', 'count_words_upper',
           'count_words_title', 'count_stopwords', 'mean_word_len',
           'word_unique_percent', 'punct_percent']
#    SELECTED_COLS_LeakyFeats=['ip', 'count_ip', 'link', 'count_links',
#           'article_id', 'article_id_flag', 'username', 'count_usernames']
    SELECTED_COLS_LeakyFeats=['count_ip', 'count_links','article_id_flag',  'count_usernames']    

    target_x = pd.concat([train_IndirectFeats[SELECTED_COLS_IndirectFeats],train_LeakyFeats[SELECTED_COLS_LeakyFeats]],axis=1)
    
    TARGET_COLS=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    target_y=train_tags[TARGET_COLS]
    
    train1(target_x, target_y)
    
    
    ##### 训练模型2：NbSVM #####
    target_x = hstack((train_bigrams,train_charngrams,train_unigrams,train_IndirectFeats[SELECTED_COLS_IndirectFeats])).tocsr()
    
    train2(target_x, target_y)
    
    toc = clock()
    print("Time:" + str(toc-tic) + "s") 
    print("End...")
    '''
    
##### 使用特征，训练模型 #####
if __name__ == "__main__":
    from time import clock
    tic = clock()
    ##### 载入特征 #####
    train_IndirectFeats = pd.read_csv('train_IndirectFeats.csv',encoding='utf-8')
    train_LeakyFeats = pd.read_csv('train_LeakyFeats.csv',encoding='utf-8')
    train_bigrams = io.mmread("train_bigrams.mtx")
    train_charngrams = io.mmread("train_charngrams.mtx")
    train_unigrams = io.mmread("train_unigrams.mtx")
    train_tags = pd.read_csv('train_tags.csv',encoding='utf-8')
    print("Load features OK...")
    ##### 训练模型1：逻辑回归 #####
    SELECTED_COLS_IndirectFeats=['count_sent', 'count_word', 'count_unique_word',
           'count_letters', 'count_punctuations', 'count_words_upper',
           'count_words_title', 'count_stopwords', 'mean_word_len',
           'word_unique_percent', 'punct_percent']
#    SELECTED_COLS_LeakyFeats=['ip', 'count_ip', 'link', 'count_links',
#           'article_id', 'article_id_flag', 'username', 'count_usernames']
    SELECTED_COLS_LeakyFeats=['count_ip', 'count_links','article_id_flag',  'count_usernames']    

    TARGET_COLS=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    target_y = train_tags[TARGET_COLS]
    '''
    ##### 只用IndirectFeats #####
    print("Use IndirectFeats...")
    target_x = train_IndirectFeats[SELECTED_COLS_IndirectFeats]
    train2(target_x, target_y)
    
    ##### 只用LeakyFeats #####
    print("Use LeakyFeats...")
    target_x = train_LeakyFeats[SELECTED_COLS_LeakyFeats]
    train2(target_x, target_y)
    
    ##### 只用DirectFeats #####
    print("Use irectFeats...")
    target_x = hstack((train_bigrams,train_charngrams,train_unigrams)).tocsr()
    train2(target_x, target_y)
    '''
    ##### 用所有Feats #####
    print("Use all Feats...")
    target_x = pd.concat([train_IndirectFeats[SELECTED_COLS_IndirectFeats],train_LeakyFeats[SELECTED_COLS_LeakyFeats]],axis=1)
    target_x = hstack((train_bigrams,train_charngrams,train_unigrams,target_x)).tocsr()
    train2(target_x, target_y)
    
    toc = clock()
    print("Time:" + str(toc-tic) + "s") 
    print("End...")