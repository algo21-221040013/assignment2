# %%
from input_data_prepare import *
from net_work_output import *
import pandas as pd
from multiprocessing import Pool
from datetime import datetime
import os
code_list = pd.read_pickle('/code/task4-imbalance/code_list.pkl')


# %%
# train_test = {}
# for code in code_list[339:]:
#     train_test[code] = input_data_one_stock_all_days(code,part_fraction=10,).train_n_test_data(ft_num=18,  n_threds=16)
#     print(code, 'is done')
#     pd.to_pickle(train_test[code], '/code/task4-imbalance/train_result/train_test_result_%d.pkl' % code)
# # pd.to_pickle(train_test,  '/code/task4-imbalance/train_result/train_test_result_all.pkl')


# code parellel

# k = input('历史多久做滑窗/s')
# d = input('未来多久的持仓时间阈值/s')
# n = input('几倍标准差')

# k,d,n = (150,120,1)    # v1
k,d,n = (90,30,1)    # v2
k,d,n = (120,30,1)    # v3
path = '/code/task4-imbalance/train_result_k%d_d%d_n%d' %(k,d,n)
if not os.path.exists(path):
    os.mkdir(path)
def code_parallel(code):
    t1 = time.time()
    result = input_data_one_stock_all_days(code,part_fraction=10,k=k,d=d,n=n).train_n_test_data(ft_num=18)
    t2 = time.time()
    code_path = os.path.join(path, str(code) + '.pkl')
    pd.to_pickle(result, code_path)
    print(code,' is done,time cost ',t2-t1,'now time is ',datetime.now().strftime("%H:%M:%S"))
    return result

t1 = time.time()
with Pool(16) as pool:
    part_result = pool.map(code_parallel, code_list)
# print('start to pickle')
# pd.to_pickle(part_result, '/code/task4-imbalance/train_result/code_parallel_train_result_k%d_d%d_n%d.pkl' %(k,d,n))
# part_result.to_pickle('/code/task4-imbalance/train_result/code_parallel_train_result_k%d_d%d_n%d.pkl' %(k,d,n))
all_stock_result = []
for code_path in os.listdir(path):
    all_stock_result.append(pd.read_pickle(os.path.join(path, code_path)))
pd.to_pickle(all_stock_result, os.path.join(path, 'all_stock_result_k%d_d%d_n%d.pkl' %(k,d,n)) )

t2 = time.time()
print('total time',t2-t1)
# %%
# network
# all_stock_result = []
# for code_path in os.listdir(path):
#     all_stock_result.append(pd.read_pickle(os.path.join(path, code_path)))
# pd.to_pickle(all_stock_result, 'all_stock_result_k%d_d%d_n%d.pkl' %(k,d,n))

# x_train_split = [x[0] for x in all_stock_result]
# x_train_all = np.concatenate(x_train_split)

# y_train_split = [y[2] for y in all_stock_result]
# y_train_all = np.concatenate(y_train_split)

# x_test_split = [x[3] for x in all_stock_result]
# x_test_all = np.concatenate(x_test_split)

# y_test_split = [y[5] for y in all_stock_result]
# y_test_all = np.concatenate(y_test_split)

# # his
# his = net_work_train(x_train_all, y_train_all, x_test_all, y_test_all)

# pd.to_pickle(his, 'his_k%d_d%d_n%d.pkl' %(k,d,n))

# # pic
# pic_his(his, k, d, n, path)
# %%
