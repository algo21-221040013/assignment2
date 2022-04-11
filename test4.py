# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from itertools import product
from tqdm.notebook import tqdm
import statsmodels.api as sm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import datetime 

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
# import keras
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping


# %%
# 90-30-1 数据
all_stock_result = pd.read_pickle('/code/task4-imbalance/train_result_k90_d30_n1/all_stock_result_k90_d30_n1.pkl')


x_train_split = [x[0] for x in all_stock_result]
x_train_all = np.concatenate(x_train_split)

y_train_split = [y[2] for y in all_stock_result]
y_train_all = np.concatenate(y_train_split)

x_test_split = [x[3] for x in all_stock_result]
x_test_all = np.concatenate(x_test_split)

y_test_split = [y[5] for y in all_stock_result]
y_test_all = np.concatenate(y_test_split)

# %%
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



    model1_opt = optimizers.Adam(lr = 0.0005)
    model1.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    early_stop = EarlyStopping(min_delta=0.001, patience=100, monitor="val_accuracy")   

    # 对y序列进行预测
    his = model1.fit(
            x_train_all[:,:,:-3], 
        y_train_all, 
            batch_size=64, 
            epochs=30,
            validation_data=(x_test_all[:,:,:-3], y_test_all),
        #     callbacks=[early_stop]
            )


pd.to_csv(his, '/code/task4-imbalance/logs/history_90_30.csv')

# %%
# fig output
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
plt.savefig('/code/task4-imbalance/logs/loss_allstock_90_30.png' , bbox_inches='tight')



