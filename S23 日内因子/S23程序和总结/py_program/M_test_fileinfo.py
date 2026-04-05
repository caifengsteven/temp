# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:39:07 2019

@author: adair2019
"""

import pandas as pd
import numpy as np
import os,time,sys
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://root:liudehua@localhost:3306/S23?charset=utf8')

def get_file_name(file_dir,file_type):
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == file_type:  
                L.append(os.path.join(root, file))             
    return L

file_dir = r'I:\data\Stk_TradeByTrade_2017'
fns = get_file_name(file_dir,'.csv')
x = pd.DataFrame({'x':fns})
x.to_csv('filelist.csv')