# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
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
import time
import warnings
warnings.filterwarnings("ignore")
# from evaluate_func import *
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# %%
market_data_train = pd.read_pickle('/code/20211027_alldays_market_data_cleanadd/market_data_train.pkl')

# %% [markdown]
# # 输入数据标准化

# %%
# def normalize_time(x):

def second_to_time(second,add = True):       # 跳过11.30-13.00，分时，原始数据的时间是1e7的形式
    if add==False:
        if second>41400:
            second += 3600*1.5
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return h*1e7 + m*1e5 + s*1e3

def time_to_second(time):
    h = time // 1e7
    min = time%1e7//1e5
    s = time % 1e5 /1e3
    result = 3600*h + 60*min+s
    if result>=46800:      # 13点以后的时间，转成11.30之后的时间，让上下午的时间连起来
        result -= 1.5*3600
    return result
    


# %%
# market_data_train[(20191204, 1)]

def ofi_calc(data):
    OFI =[]                      # 返回一个长度为5的各个档位OFI
    for m in range(1,6):        # 第m档数据
        
        e = []

        for tal in range(1, data.shape[0]):    
            # W
            if data[tal,4+m] > data[tal-1,4+m]:   # if b_m(tal_n) > b_m(tal_n-1) 
                w = data[tal, 14+m]
            elif data[tal,4+m] == data[tal-1,4+m]:     # if b_m(tal_n) == b_m(tal_n-1) 
                w = data[tal, 14+m] - data[tal-1, 14+m]
            else:                                            # # if b_m(tal_n) < b_m(tal_n-1)                        
                w = data[tal-1, 14+m]

            # V
            if data[tal, 9+m] > data[tal-1, 9+m]:   # if a_m(tal_n) > a_m(tal_n-1) 
                v = -data[tal-1, 19+m]
            elif data[tal,4+m] == data[tal-1,4+m]:     # if a_m(tal_n) == a_m(tal_n-1) 
                v = data[tal, 19+m] - data[tal-1, 19+m]
            else:                                            # # if a_m(tal_n) < a_m(tal_n-1)                        
                v = data[tal, 19+m]

           
            e.append(w-v)
        
        OFI.append(np.sum(e))
    return OFI

class input_data_one_day_one_stock:
    def __init__(self, date, code, k=60, d=40, n=1, time_window_ofi=10, part_fraction=1):
        self.date = date
        self.code = code

        self.time_window_ofi = time_window_ofi     # 计算ofi的时间窗口长度,默认是十秒
        self.k = k                                 # 历史k时间段计算价格波动作为上下界阈值
        self.d = d                                 # 持仓时间阈值                                     
        self.n = n                                 # n倍价格波动作为阈值
        self.part_fraction = part_fraction         # 数据n抽1
        self.data = market_data_train[(date, code)]

    def normalized_input_data(self):
        time_window_ofi = self.time_window_ofi
        data = self.data

        diff_data = np.diff(data,axis=0)[:,[3,4,5,10,15,20]]     # 差分数据：保留成交量，成交金额，买一价+量，卖一价+量六个差分数据
        ask1_bid1_delta = (data[1:,10] - data[1:,5]).reshape(-1,1)                 # 卖一价减去买一价
        # input_8 = np.hstack((data[1:,3].reshape(-1,1), diff_data,ask1_bid1_delta))  # 除了ofi以外的其他数据 
        
        # input_8 = np.hstack((data[1:,[3,4]], diff_data,ask1_bid1_delta))  # 除了ofi以外的其他数据 

        input_8 = np.hstack((data[1:,[3,4]], diff_data, ask1_bid1_delta))  # 除了ofi以外的其他数据,加上ytrue 

        # second_time = pd.Series(data[:,1]).apply(time_to_second)    # 把时间转化成秒，且上下午的时间是连起来的

        ofi_list = []
        # for t in second_time:       # v1
        # for t in second_time[::self.part_fraction]:    # v2
        for t in data[2::self.part_fraction,25]:    # v3   time_sec
            # ofi = ofi_calc(data[(second_time>t-time_window_ofi) & (second_time<=t)])      # 这个时间段的5档ofi
            ofi = ofi_calc(data[(data[:,25]>t-time_window_ofi) & (data[:,25]<=t)])      # 这个时间段的5档ofi--v3
            ofi_list.append(ofi)       # 时间序列上的五档ofi结果
        # ofi_list = np.array(ofi_list)[1:] 
        ofi_list = np.array(ofi_list)   # v3

        # input_result_array = np.hstack((input_8, ofi_list))      # v1
        input_result_array = np.hstack((input_8[1::self.part_fraction], ofi_list))      # v2
        input_result_array = np.hstack((input_result_array, data[2::self.part_fraction,[27,28,29]]))      # v2

        # return input_result_array[1:]     # 输出1：长度是为了和trple barrier的长度一致     v1
        return input_result_array     # 输出1：长度是为了和trple barrier的长度一致    v2

    def triple_barrier_labeling(self, bar = 'time'):
        k =self.k
        d =self.d
        n =self.n
        data = self.data

        # mid_price = pd.Series(data[:,26])     # 将中间价格转化成series，方便后续的滚动窗口操作,固定slcie数量，即窗口长度，**todo**
        
        # 计算每个时间点前k个时间点之内的价格波动率

        return_data = np.diff(data[:,26]) / data[1:,26]    # 收益率array
        second_time = pd.Series(data[:,1]).apply(time_to_second)    # 把时间转化成秒，且上下午的时间是连起来的
        
        volitility_list = []                   # 窗口为K的历史收益波动率序列第一个值肯定是0,其长度为data长度-1
     

        for t in data[2::self.part_fraction, 25]:              # v3
            volitility = return_data[(data[1:,25]>t-k) & (data[1:,25]<=t)].std()
            volitility_list.append(volitility)
        
        volitility_array = np.array(volitility_list)
      

        # 从第二个return_data开始计算是否触碰边界,即第三个midprice时间开始计算label      v2
        up_list = list((1 + n * volitility_array) * data[2::self.part_fraction,26])          # 价格上界
        down_list = list((1 - n * volitility_array) * data[2::self.part_fraction,26])        # 价格下界

        label_list = []
        # for n, t in enumerate(second_time[2:]):
        for n, t in enumerate(data[2::self.part_fraction, 25]):
            if bar == 'time':
                price_tmp = data[2:,26][(data[2:, 25]>t) & (data[2:, 25]<=t + d)]   # 对每个时间点，price_tmp为其往后d时间内的价格    
            # elif bar == 'tick':
            #     price_tmp = data[2+n:2+n+d,26]                                            # 对每个时间点，price_tmp为其往后d个tick内的价格    
            label_array = np.where( price_tmp> up_list[n], 1, np.where(price_tmp < down_list[n], -1, 0))
            label = label_array[label_array!=0][0] if len(label_array[label_array!=0]) > 0 else 0
            label_list.append(label)

        return np.array(label_list).reshape(-1,1)

    def input_data_n_label(self):
        result = np.hstack((self.normalized_input_data(), self.triple_barrier_labeling()))
        result = result[~np.isnan(result).any(axis=1)]     # 去掉inputdata缺失的数据
        return result
    
    def change_input_data_n_label(self):
        result = np.hstack((self.normalized_input_data(), self.triple_barrier_labeling()))
        result = result[~np.isnan(result).any(axis=1)]     # 去掉inputdata缺失的数据
        return result








class input_data_one_stock_all_days():
    def __init__(self,  code, k=60, d=40, n=1, time_window_ofi=10, part_fraction=1):
        # self.date = date
        self.code = code

        self.time_window_ofi = time_window_ofi     # 计算ofi的时间窗口长度,默认是十秒
        self.k = k                                 # 历史k时间段计算价格波动作为上下界阈值
        self.d = d                                 # 持仓时间阈值                                     
        self.n = n                                 # n倍价格波动作为阈值
        self.part_fraction = part_fraction         # 数据n抽1

        # self.data = market_data_train[(date, code)]

    def get_time_train_test(self, prop=0.5):     # prop指的是用所有数据中的训练集和测试集长度的比例，0.5指的是一半训练一半测试
        code = self.code

        self.date_list = np.unique([x[0] for x in list(market_data_train.keys()) if x[1]==code])
        self.train_date = self.date_list[:int(len(self.date_list) * prop)]
        self.test_date = self.date_list[int(len(self.date_list)* prop):]
 
        return self.train_date, self.test_date

    def map_net_prepare(self,x):
        date = x
        code = self.code
        result = input_data_one_day_one_stock(date, code, k=self.k, d=self.d, n=self.n, time_window_ofi=self.time_window_ofi,).input_data_n_label() # v1
        # result = input_data_one_day_one_stock(date, code, k=self.k, d=self.d, n=self.n, time_window_ofi=self.time_window_ofi, part_fraction=self.part_fraction).input_data_n_label() # v2
        return result
    
    def map_change_ofi_y(self,x):
        date = x
        code = self.code
        
        # 输出ofi和y_label
        result = input_data_one_day_one_stock(date, code, k=self.k, d=self.d, n=self.n, time_window_ofi=self.time_window_ofi,).change_input_data_n_label() # v1
        return result
    
    def train_n_test_data(self, n_threds = 32, ft_num=15, tmp_x =3):   # 默认用32个核跑
        '''
        输出顺序：x_train, y_train_sequential, y_train_last, x_test, y_test_sequential, y_test_last
        '''
        # with Pool(n_threds) as pool:      # 日期并行版本 v1
        #     # train_data = pool.map(map_net_prepare, x_before20201231)
        #     train_data_index, test_data_index = self.get_time_train_test()

        #     train_data = pool.map(self.map_net_prepare, train_data_index)    # code==self.code的训练集数据
        #     test_data = pool.map(self.map_net_prepare, test_data_index)      # code==self.code的测试集数据
        t1 = time.time()
        train_data_index, test_data_index = self.get_time_train_test()
        train_data = []
        test_data = []
        for train_date in train_data_index:
            train_data.append(self.map_net_prepare(train_date))
        for test_date in test_data_index:
            test_data.append(self.map_net_prepare(test_date))



        # timesteps = self.k
        timesteps = int(self.k / tmp_x)     # 窗口长度,默认使用时间窗口长度的1/3，即60s的窗口长度为20个tick
        x_train_list = []
        part_fraction=self.part_fraction


        scaler = StandardScaler().fit(np.concatenate(train_data)[:,:-4])         # 得到训练集的标准化参数，只标准化inputdata，ytrue和ylabel不变

        for day in range(len(train_data)):
            if len(train_data[day]) > timesteps:     # v2
                train_day = scaler.transform(train_data[day][:,:-4]) # 分出每天的train_data，避免跨天数据重叠
                train_day = np.hstack((train_day, train_data[day][:,-4:]))

                for second in range(len(train_day)-timesteps):     # 每天按照时间分出sample   
                    x_train_list.append(train_day[second:second + timesteps])

        x_train = np.concatenate(x_train_list, axis=0).reshape(-1, timesteps, ft_num)[::part_fraction,:,:-1]
        y_train_sequential = np.concatenate(x_train_list, axis=0).reshape(-1, timesteps, ft_num)[::part_fraction,:,-1]      # ytrain序列,shape = (247685,timesteps, 3)
        y_train_last = keras.utils.to_categorical(y_train_sequential[:,-1]  ,num_classes=3)
        # y_train_sequential = keras.utils.to_categorical(y_train_sequential, num_classes=3)        
        # y_train_last = y_train_sequential[:,-1,:]  

        x_test_list = []
        for day in range(len(test_data)):
            if len(test_data[day]) > timesteps:  
                test_day = scaler.transform(test_data[day][:,:-4]) # 分出每天的test_data，避免跨天数据重叠,使用测试集的数据进行标准化
                test_day = np.hstack((test_day, test_data[day][:,-4:]))

                for second in range(len(test_day)-timesteps):     # 每天按照时间分出sample   
                    x_test_list.append(test_day[second:second + timesteps])

        x_test = np.concatenate(x_test_list, axis=0).reshape(-1, timesteps, ft_num)[::part_fraction,:,:-1]
        y_test_sequential = np.concatenate(x_test_list, axis=0).reshape(-1, timesteps, ft_num)[::part_fraction,:,-1]      # ytest序列,shape = (247685,timesteps, 3)
        y_test_last = keras.utils.to_categorical(y_test_sequential[:,-1]  ,num_classes=3)
        t2 = time.time()
        # y_test_sequential = keras.utils.to_categorical(y_test_sequential, num_classes=3)        
        # y_test_last = y_test_sequential[:,-1,:]  

        # 10抽1    v1
        # x_train = x_train[::part_fraction,:,:]
        # y_train_sequential = y_train_sequential[::part_fraction,:,:]
        # y_train_last = y_train_last[::part_fraction,:]
        # x_test = x_test[::part_fraction,:,:]
        # y_test_sequential = y_test_sequential[::part_fraction,:,:]
        # y_test_last = y_test_last[::part_fraction,:]
        # return x_train,  y_train_last, x_test,  y_test_last
        # print(self.code, 'is done,time cost', t2-t1)
        return x_train,  y_train_sequential, y_train_last, x_test,  y_test_sequential, y_test_last

    def change_y_ofi(self, n_threds = 32, ft_num=15, tmp_x =3):   # 默认用32个核跑
        with Pool(n_threds) as pool:  
            # train_data = pool.map(map_net_prepare, x_before20201231)
            train_data_index, test_data_index = self.get_time_train_test()

            train_data = pool.map(self.map_change_ofi_y, train_data_index)    # code==self.code的训练集数据
            test_data = pool.map(self.map_change_ofi_y, test_data_index)      # code==self.code的测试集数据
        
