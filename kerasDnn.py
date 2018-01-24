from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, Embedding
from keras.models import Model, Sequential
from keras.layers import LSTM,Bidirectional,GlobalMaxPool1D,Dropout,GRU,add
from keras.layers.pooling import MaxPool1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.layers.merge import concatenate
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
import time
import random

''' 全连接神经网络 '''
class dnn:
    def __init__(self):
        self.epochs = 1
        self.learning_rate = 0.002
        self.keep_prob = 0.2
        self.batch_size = 128
        
        self.model = Sequential([
            Dense(512, activation='relu', input_shape=(384,)),
            Dropout(self.keep_prob),
            Dense(512, activation='relu'),
            Dropout(self.keep_prob),
            Dense(6, activation='sigmoid')
        ])

        self.model.compile(loss = 'binary_crossentropy',
                   optimizer = Adam(lr=self.learning_rate),
                   metrics = ['accuracy'])
        
    def fit(self, X, Y, verbose=1):
        self.model.fit(X, Y, batch_size=self.batch_size, epochs=self.epochs, verbose=verbose)

    def predict(self, X, verbose=1):
        return self.model.predict(X, verbose=verbose)
        
    def evaluate(self, X, Y, verbose=1):
        return self.model.evaluate(X, Y, verbose=verbose)

''' 交叉验证 '''
def cv(get_model, X, Y, test, K=10):
    def getMean(results, mean="harmonic_mean"):
        ''' 计算均值 '''
        if mean == "harmonic_mean": #调和平均数
            test_predicts = np.zeros(results[0].shape)
            for fold_predict in results:
                if 0 in fold_predict: #防止除以0
                    test_predicts += 1./(fold_predict + random.uniform(-1,1)/1e10)
                else:
                    test_predicts += 1./fold_predict
            test_predicts = len(results) / test_predicts
        elif mean == "geometric_mean": #几何平均数
            test_predicts = np.ones(results[0].shape)
            for fold_predict in results:
                test_predicts *= fold_predict
            test_predicts **= (1. / len(results))
        elif mean == "arithmetic_mean": #算术平均数
            test_predicts = np.zeros(results[0].shape)
            for fold_predict in results:
                test_predicts += fold_predict
            test_predicts /= len(results)
        return test_predicts
    
    ### k折交叉验证 ###
    kf = KFold(len(X), n_folds=K, shuffle=False)

    results=[]
    scores=[]
    ### 交叉验证 ###
    for i, (train_index, valid_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        train_x = X[train_index]
        train_y = Y[train_index]
        valid_x = X[valid_index]
        valid_y = Y[valid_index]
    
        model = get_model()
        model.fit(train_x, train_y)
        results.append(model.predict(test)) #预测结果
        score = model.evaluate(valid_x, valid_y) #验证集上的loss、acc
        print("valid set score:", score)
        scores.append(score) 
        '''
        results:    list的len为10
        results[0]: np的shape为(153164, 6)
        '''

    test_predicts = getMean(results, mean="harmonic_mean")
    np.savetxt('scores_mean.csv', np.array(scores).mean(axis=0), delimiter=',')
    print("valid set mean score:", np.array(scores).mean(axis=0))

    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    sample_submission = pd.read_csv('/home/qianruifeng/xgy/Kaggle/input/sample_submission.csv')
    sample_submission[list_classes] = test_predicts
    sample_submission.to_csv("baseline.csv.gz", index=False, compression='gzip')
    print("save csv OK...")

''' 训练 '''
def train():
    print("load data start...")
    train_x = np.loadtxt('/home/qianruifeng/xgy/Kaggle/code/train_tdidf_pca.csv', delimiter=',')
    test_x = np.loadtxt('/home/qianruifeng/xgy/Kaggle/code/test_tdidf_pca.csv', delimiter=',')
    train_y = pd.read_csv('/home/qianruifeng/xgy/Kaggle/code/labels.csv').values
    print("load data ok...", time.time()-start_time)
    getModel = lambda:dnn() #传函数
    cv(getModel, train_x, train_y, test_x, K=10)

if __name__ == "__main__":
    start_time = time.time()
    train()
    print("End...", time.time()-start_time)