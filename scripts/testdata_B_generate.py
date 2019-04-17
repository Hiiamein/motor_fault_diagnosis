
# coding: utf-8

# # 数据读取

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import torch
import torchvision
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


traindata_0_path = r'../../data/motor_fault/Motor_tain/Negative/'
traindata_1_path = r'../../data/motor_fault/Motor_tain/Positive//'
testdata_path = r'../../data/motor_fault/Motor_testP/'


# **电机正反转处理：分别用正转和反转数据训练两个模型，预测也分别预测，最后只要有一个模型判断样本为故障，则认为该电机故障**  
# **下面0代表负例或正常电机，1代表正例或故障电机**

# In[ ]:


# 读取反转测试数据
testdata_B = []
 motor_name_B = []
for i in os.listdir(testdata_path):
    if 'B' in i:
        single_testdata_B_path = os.path.join(testdata_path, i)
        testdata_B.append(pd.read_csv(single_testdata_B_path))
        motor_name_B.append(i)
num_test_B = len(testdata_B)
print("Backward test data number: ", num_test_B)


# # 数据查看和探索

# In[ ]:


print("testdata_B[0].head()","\n", testdata_B[0].head())
print("testdata_B[0].shape: ",testdata_B[0].shape)


# # 特征提取
# 思路：对每一个电机的数据，用滑动窗口法提取特征，提取的特征如下：
# ![image.png](attachment:image.png)

# In[7]:


def feature_extraction(series_data, window_width=10000, steps=100):
    # series_data is input
    # series_data = np.array(traindata_0_F[0].iloc[:,0])

    # sliding window
    window_width = 10000
    t_start = 0
    windows = []
   
    while t_start+window_width <= len(series_data):
        windows.append(series_data[t_start:t_start+window_width])
        t_start += steps
    windows_num = len(windows)
    windows = np.array(windows)
    #print("windows.shape: {},t_start: {}".format(windows.shape, t_start))

    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    f6 = []
    f7 = []
    f8 = []
    f9 = []
    f10 = []
    f11 = []
    for i in range(windows_num):
        temp = windows[i]
        data_max = np.max(temp)
        data_min = np.min(temp)
        # 时域特征
        f1_temp = data_max - data_min
        f1.append(f1_temp)
        f2_temp = np.sqrt(np.mean(temp ** 2))
        f2.append(f2_temp)
        f3_temp = f2_temp / abs(np.mean(temp))
        f3.append(f3_temp)
        f4_temp = data_max / abs(np.mean(temp))
        f4.append(f4_temp)
        f5_temp = np.mean((abs(temp) - np.mean(temp)) ** 4) / (f2_temp ** 4)
        f5.append(f5_temp)
        f6_temp = data_max / f2_temp
        f6.append(f6_temp)
        f7_temp = f2_temp / np.mean(temp)
        f7.append(f7_temp)
        # 频域特征
        '''
        f8_temp = np.sum(temp * temp) / (2 * np.pi * np.sum(temp ** 2)) #分子不对
        f8.append(f8_temp)
        f9_temp = np.sum(temp ** 2) / (4 * (np.pi ** 2) * np.sum(temp ** 2)) #分子不对
        f9.append(f9_temp)
        f10_temp = 
        f10.append(f10_temp)
        f11_temp = 0
        f11.append(f11_temp)
        '''
        
    f1 = np.array(f1)
    f2 = np.array(f2)
    f3 = np.array(f3)
    f4 = np.array(f4)
    f5 = np.array(f5)
    f6 = np.array(f6)
    f7 = np.array(f7)
    f8 = np.array(f8)
    f9 = np.array(f9)
    f10 = np.array(f10)
    f11 = np.array(f11)

    features = np.concatenate((f1[:, np.newaxis], f2[:, np.newaxis], f3[:, np.newaxis],
                               f4[:, np.newaxis], f5[:, np.newaxis], f6[:, np.newaxis], 
                               f7[:, np.newaxis]), axis=1)
    return features


# **目前频域特征还有问题，只用前7个时域特征即可**

# In[ ]:


# 反转的测试数据testdata_B
X_test_B = []
X_test_B_motor_name = []
for i in range(len(testdata_B)):
    features1 = feature_extraction(np.array(testdata_B[i].iloc[:,0]), window_width=10000, steps=50)
    features2 = feature_extraction(np.array(testdata_B[i].iloc[:,1]), window_width=10000, steps=50)
    features = np.concatenate((np.array(features1), np.array(features2)),axis=1)
    X_test_B.append(features)
    for j in range(features.shape[0]):
        X_test_B_motor_name.append(motor_name_B[i])
X_test_B = np.array(X_test_B)
print("X_test_B.shape", X_test_B.shape)
X_test_B = X_test_B.reshape((-1, 14))
print("X_test_B.shape", X_test_B.shape)


# In[ ]:


# 保存数据，其中X_test_B.csv里面是测试数据的X，motor_name_test_B.csv是测试数据对应行的电机名
X_test_B = np.concatenate((X_test_B, label), axis=1)
print("X_test_B.shape: ", X_test_B.shape)
print("len(X_test_B_motor_name): ",len(X_test_B_motor_name))
np.savetxt("../../data/motor_fault/X_test_B.csv", X_test_B, delimiter=",")
np.savetxt("../../data/motor_fault/motor_name_test_B.csv", motor_name_test_B, delimiter=",")

