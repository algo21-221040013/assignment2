# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import os 
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
# all_stock_result = {}
# for code_path in os.listdir('/code/task4-imbalance/train_result'):
#     all_stock_result[code_path[18:-4]] = pd.read_pickle(os.path.join('/code/task4-imbalance/train_result', code_path))

all_stock_result = pd.read_pickle('/code/task4-imbalance/all_stock_result.pkl')

x_train_split = [x[0] for x in list(all_stock_result.values())]
x_train_all = np.concatenate(x_train_split)
# x_train_all = np.concatenate(x_train_split)[:100]

y_train_split = [y[2] for y in list(all_stock_result.values())]
y_train_all = np.concatenate(y_train_split)
# y_train_all = np.concatenate(y_train_split)[:100]

x_test_split = [x[3] for x in list(all_stock_result.values())]
x_test_all = np.concatenate(x_test_split)
# x_test_all = np.concatenate(x_test_split)[:100]

y_test_split = [y[5] for y in list(all_stock_result.values())]
y_test_all = np.concatenate(y_test_split)
# y_test_all = np.concatenate(y_test_split)[:100]


# %%
# pd.to_pickle(all_stock_result, 'all_stock_result.pkl')


# %%
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

early_stop = EarlyStopping(min_delta=0.002,patience=2)   

# 对y序列进行 
his = model1.fit(x_train_all[:,:,:-3], y_train_all,          
          batch_size=1024, epochs=30,
          validation_data=(x_test_all[:,:,:-3], y_test_all),
          callbacks=[early_stop])

# model1.save('/code/task4-imbalance/models/model_l_bp_l_2d_allstock.h5')       # 


# %%

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
plt.savefig('/code/task4-imbalance/train_result/loss_allstock.png' , bbox_inches='tight')


df_result = pd.concat([pd.Series(x) for x in his.history.values()],axis=1)
df_result.columns = ['train_loss','train_acc','val_loss','val_acc']
df_result.index.names=['epoch']
df_result.iloc[-1:,:].round(4)

best_result = pd.DataFrame([[np.argmax(df_result.train_acc),df_result.train_loss.min(), df_result.train_acc.max(),np.argmax(df_result.val_acc),df_result.val_loss.min(), df_result.val_acc.max()]],columns=['best_train_epoch','train_loss','train_acc','best_val_epoch','val_loss','val_acc']).round(4)

# pd.to_csv(best_result, 'best_result_all_stock.csv')
best_result.to_csv('/code/task4-imbalance/train_result/best_result_all_stock.csv')

# his.history


