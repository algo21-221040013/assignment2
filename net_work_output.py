import pandas as pd
import numpy as np
import os 
from itertools import product
from tqdm.notebook import tqdm
import statsmodels.api as sm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
# import keras
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
# **************************************************************************************
# net work train
def net_work_train(x_train_all, y_train_all, x_test_all, y_test_all):
    with tf.device("/gpu:0"):   # gpu版本尝试
        # 对每个sample的最后一个ylabel进行预测
        model1 = Sequential()

        # model1.add(Dense(32))  # 维度为 32 的全连接层
        timesteps = x_train_all.shape[1]
        data_dim = x_train_all.shape[2]-3
        # data_dim = 5

        model1.add(LSTM(128, return_sequences=True,
                    input_shape=(timesteps, data_dim)))  # 返回维度为 128 的向量序列

        model1.add(Dropout(0.5))    # dropout 0.5

        model1.add(LSTM(128, return_sequences=False))  # 返回维度为 128 的向量序列

        model1.add(Dense(32))  # 维度为 32 的全连接层
        model1.add(Dense(3, activation='softmax'))    # softmax 后的最终结果


        # model1.add(LSTM(512, return_sequences=True))  # 返回维度为 1024 的向量序列
        # model1.add(LSTM(512//2, return_sequences=True))  # 返回维度为 512 的向量序列
        # model1.add(LSTM(256//2, return_sequences=False))  # 返回维度为 256 的向量序列

        # model1.add(Dense(128))  # 维度为 128 的全连接层
        # model1.add(Dense(6))  # 维度为 6 的全连接层
        # model1.add(Dense(3, activation='softmax'))    # softmax 后的最终结果

        model1_opt = optimizers.Adam(lr = 0.0005)
        model1.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        early_stop = EarlyStopping(min_delta=0.001,patience=3)   

        # 对y序列进行预测 
        his = model1.fit(x_train_all[:,:,:-3], y_train_all,          
                batch_size=1024, epochs=30,
                validation_data=(x_test_all[:,:,:-3], y_test_all),
                callbacks=[early_stop])
        return his

def pic_his(his,k,d,n,path):
    plt.figure(figsize=(16,20))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(his.history['loss'],label='train_loss')
    ax1.plot(his.history['val_loss'], label='val_loss')
    ax1.set_title('loss',fontdict={'size':16})

    plt.legend(prop = {'size':16})
    plt.tick_params(labelsize=15)
    plt.xticks(range(0,len(his.history['val_loss'])))
    plt.xlabel('epoch',size=16)

    ax2 = plt.subplot(2,1,2)
    ax2.plot(his.history['accuracy'],label='train_acc')
    ax2.plot(his.history['val_accuracy'], label='val_acc')
    plt.legend(prop = {'size':16})
    plt.tick_params(labelsize=15)
    plt.xticks(range(0,len(his.history['val_loss'])))
    plt.xlabel('epoch',size=16)
    ax2.set_title('accuracy',fontdict={'size':16})

    # plt.set_ticks_position('top')
    # plt.subplots_adjust(hspace=0.1)
    fig_path = os.path.join(path, 'loss_allstock_k%d_d%d_n%d.png' %(k,d,n))
    plt.savefig(fig_path, bbox_inches='tight')
