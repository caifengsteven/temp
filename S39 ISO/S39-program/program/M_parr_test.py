# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:42:51 2020
1输入必须为元组
2必须在if __name__=='__main__' 内调用，否者无法运行 切记切记
@author: adair2019
"""

import math
import datetime
import numpy as np
import multiprocessing as mp
import os


def get_file_name(file_dir,file_type):
    L=[]
    L_s = []   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == file_type:  
                L.append(os.path.join(root, file))  
                L_s.append(file)
    return L,L_s

def save_name(fn):
    return {'re':len(fn)}

if __name__ == '__main__':
    pn = 'E:\BaiduNetdiskDownload\yuqer_factors'
    
    _,fns = get_file_name(pn,'.csv')
    num_cores = int(mp.cpu_count())
    num_cores = int(np.floor(num_cores/2))
    
    cut_fns = []
    for i in range(num_cores):
        cut_fns.append(fns[i::num_cores])
    
    pool = mp.Pool(num_cores)
    results = [pool.apply_async(save_name, args=(sub_fns,)) for sub_fns in cut_fns]
    results = [p.get() for p in results]
"""
def train_on_parameter(name, param):
    result = 0
    for num in param:
        result += math.sqrt(num * math.tanh(num) / math.log2(num) / math.log10(num))
    return {name: result}


if __name__ == '__main__':

    start_t = datetime.datetime.now()

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores)
    param_dict = {'task1': list(range(10, 30000000)),
                  'task2': list(range(30000000, 60000000)),
                  'task3': list(range(60000000, 90000000)),
                  'task4': list(range(90000000, 120000000)),
                  'task5': list(range(120000000, 150000000)),
                  'task6': list(range(150000000, 180000000)),
                  'task7': list(range(180000000, 210000000)),
                  'task8': list(range(210000000, 240000000))}
    results = [pool.apply_async(train_on_parameter, args=(name, param)) for name, param in param_dict.items()]
    results = [p.get() for p in results]

    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")
"""