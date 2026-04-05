"""
# coding: utf-8

# 由于接下来想写一篇用到GcForest的策略，所以先把GcForest的介绍以及小热身的简单应用搬过来。
周志华教授和冯霁博士在2017年2月28日发表的论文 

# * 这个模型是串联结构的
# * 模型是多粒度扫描的
# * 是一种基于树集成的模型

# >1、 在大数据集上，分别和DNN类、LR、RF、SVM等算法做了对比。
# 	 1）图像分类（数据集：MNIST）：略低于DNN，基本持平（99.05%与98.96%）；
# 	 2）人脸识别（数据集：ORL）：人脸图像数量不同时，gcForest都最好；
# 	 3）音乐分类（数据集：GTZAN）：gcForest最好（65.67%）；
# 	 4）手部运动识别（数据集：sEMG）：gcForest最好（55.93%）；
# 	 5）情感分类（数据集：IMDB）：gcForest最好（89.32%）。
# >2、在几个小数据集上表现和很好。
# 
# 总的来说，gcForest有如下若干优点：
# 1. 计算开销小
# 2. 模型效果好
# 3. 超参数少，模型对超参数调节不敏感，并且一套超参数可使用到不同数据集
# 4. 可以适应于不同大小的数据集，模型复杂度可自适应伸缩
# 5. 每个级联的生成使用了交叉验证，避免过拟合
# 6. 相对于DNN这个大黑盒，gcForest更容易进行理论分析

https://uqer.datayes.com/v3/community/share/5b54b96e0b8be70153ffa366
https://github.com/kingfengji/gcForest

import sys
import talib as ta

ema = ta.EMA(close, timeperiod=30)
macd = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod = 9)[0]
momentum = ta.MOM(close, timeperiod=10)
rsi = ta.RSI(close, timeperiod=14)
linreg = ta.LINEARREG(close, timeperiod=14)
var = ta.VAR(close, timeperiod=5, nbdev=1)#获取当前的收盘价的希尔伯特变换
cycle = ta.HT_DCPERIOD(close)#获取平均真实波动范围指标ATR,时间段为14
atr = ta.ATR(df['highestIndex'].values, df['lowestIndex'].values, df['closeIndex'].values, timeperiod=14)#把每根k线的指标放入数组X中，并转置
Xa = pd.DataFrame([df["tradeDate"],df['openIndex'],df['closeIndex'], df['highestIndex'], 
                  df['lowestIndex'], df['turnoverVol'],ema, macd, linreg, momentum, rsi, var, cycle, atr]).T
"""
# In[ ]:

from sqlalchemy import create_engine
import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gcforest import GCForest
import matplotlib.pyplot as plt
#sys.path.insert(0, "D:\works\works2020\adair2020_W\some\S42\ref\gcForest-master\gcForest-master\lib\gcforest")
import warnings
import sys
from datetime import date,datetime

warnings.filterwarnings('ignore')

# must be set before using
with open('para.json', 'r', encoding='utf-8') as f:
    para = json.load(f)

pn = para['yuqerdata_dir']

user_name = para['mysql_para']['user_name']
pass_wd = para['mysql_para']['pass_wd']
port = para['mysql_para']['port']

db_name1 = 'yuqerdata'
# eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name)
eng_str = 'mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name, pass_wd, port, db_name1)
engine = create_engine(eng_str)

sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''

##前复权需要换为后复权
def chs_factor(ticker='000005', begin=None, end=None,
               field=[u'symbol', u'tradeDate', u'openPrice',
                      u'highestPrice', u'lowestPrice', u'closePrice', u'turnoverVol',
                      u'turnoverValue', u'dealAmount', u'chgPct',
                      'turnoverRate', u'marketValue', u'accumAdjFactor']):
    sql_str1 = sql_str_select_data1 % (','.join(field), ticker, begin, end)
    dataday = pd.read_sql(sql_str1, engine)
    dataday = dataday.applymap(lambda x: np.nan if x == 0 else x)
    dataday.rename(columns={'symbol': 'ticker'}, inplace=True)
    ## 对数据补全
    return dataday.fillna(method='ffill')


def get_industry_class(t):
    sql_str1 = '''select ticker,industryID1 from yuqerdata.yq_industry where 
                industryVersionCD="010303" and intodate <= "%s" and 
                (outDate>"%s" or outDate is null)''' % (t, t)
    x = pd.read_sql(sql_str1, engine)
    return x


def get_month_calender():
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" order by endDate'''
    x = pd.read_sql(sql_str, engine)
    x = x['endDate'].values
    # b=[i.strftime('%Y-%m-%d') for i in x]
    return x


def get_calender():
    sql_str = '''select tradeDate from yuqerdata.yq_index where symbol = "000001" order by tradeDate'''
    x = pd.read_sql(sql_str, engine)
    x = x['tradeDate'].values
    # b=[i.strftime('%Y-%m-%d') for i in x]
    return x


cal = get_calender()


def get_IdxCons(intoDate, ticker='000300'):
    # nearst 时间
    sql_str1 = '''select symbol from yuqerdata.IdxCloseWeightGet where ticker = "%s"
            and tradingdate = (select tradingdate from yuqerdata.IdxCloseWeightGet where 
        ticker="%s" and tradingdate<="%s"  order by tradingdate desc limit 1)''' % (ticker,
                                                                                    ticker, intoDate)
    x = pd.read_sql(sql_str1, engine)
    x = x['symbol'].values
    return x

def get_index_data(index, begin='1990-01-01', end='2040-01-01'):
    r_m = pd.read_sql(
        '''select * from yq_index where symbol = "%s" and tradeDate>="%s" and tradeDate<="%s" order by tradeDate''' % (index, begin, end), engine)
    return r_m

def get_toy_config(b_type):
    if b_type == 'lo':
        c_num =2
    else:
         c_num =3   
    # 官方包中的demo配置
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 4
    # 由于只有上涨和不上涨两个类别 n_classes为2
    ca_config["n_classes"] = c_num
    ca_config["estimators"] = []
    # 把XGBoost去掉了 这个在运行中有报错 XGboost配置有点繁琐 目前先不用
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
         "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1})
    ca_config["estimators"].append(
        {"n_folds": 12, "type": "RandomForestClassifier", "n_estimators": 13, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 12, "type": "ExtraTreesClassifier", "n_estimators": 13, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 12, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    # 被注释掉的是多粒度扫描的配置 由于官方文档没有注解 还没有研究好 所以先不用
    # fg_config = {}
    # fg_config[""]
    # config["net"] = fg_config

    return config

# 把数据集分解成训练和测试子集， 参数test_size表示测试集所占比例
# 这里要注意 shuffle = False 不过我试过了随机的话预测准确度变化不大 
def gcforest_model_s42(X,y,b_type):    
    (X_tr, X_te, y_tr, y_te) = train_test_split(X, y, test_size=0.1,shuffle=False)
    standardScaler =StandardScaler()
    standardScaler.fit(X_tr)
    # standardScaler.mean_
    X_tr_std =standardScaler.transform(X_tr)
    X_te_std = standardScaler.transform(X_te)
    
    gc = GCForest(get_toy_config(b_type))
    #X_train_enc, X_test_enc = gc.fit_transform(X_train=X_tr.values, y_train=y_tr.values, 
    #                                           X_test=X_te.values, y_test=y_te.values)
    X_train_enc, X_test_enc = gc.fit_transform(X_train=X_tr_std, y_train=y_tr.values, 
                                               X_test=X_te_std, y_test=y_te.values)
    return standardScaler,gc

def gcforest_pred_s42(Xv,standardScaler,gc): 
    Xv_std=standardScaler.transform(Xv)
    y_pred = gc.predict(Xv_std)
    return y_pred

def gc_forest_val(df,startDate,endDate,split_date,b_type='lo'):
    close = df["closeIndex"].values
    Xa = pd.DataFrame([df["tradeDate"],df['openIndex'],df['closeIndex'], df['highestIndex'], 
                      df['lowestIndex'], df['turnoverVol']]).T
    Xa=Xa.dropna().reset_index().drop(['index'], axis=1)#去空值
    Xa["up"] = 2
    #用i遍历整个数据集
    if b_type=='lo':
        Xa["up"]=pd.DataFrame([1 if close[i]>1.005*close[i-1] else 0 for i in range(1, len(Xa))])
    else:
        Xa["up"]=pd.DataFrame([2 if close[i]/close[i-1]-1>0.005 else 0 if close[i]/close[i-1]-1<-0.005 else  1 for i in range(1, len(Xa))])
    #添加最后一个数据的标签为1
    Xa["up"][len(Xa)-1]=1
    #分组
    Y = Xa[(Xa['tradeDate']>split_date) & (Xa['tradeDate']<endDate)].copy()
    f = Xa[(Xa['tradeDate']>startDate) & (Xa['tradeDate']<=split_date)].copy()
    #建模集设置
    y = f["up"]
    X = f.drop(["up", "tradeDate"], axis=1)
    #验证集设置
    Xv = Y.copy()
    Xv=Xv.dropna().reset_index().drop(['index'], axis=1)#去空值
    tref = Xv.copy()
    Xv = Xv.drop(["up", "tradeDate"], axis=1)
    
    #建立模型
    standardScaler,gc = gcforest_model_s42(X,y,b_type)
    #预测
    y_pred = gcforest_pred_s42(Xv,standardScaler,gc)
    if b_type=='ls':
        y_pred = y_pred-1
    ind = pd.DataFrame({'tradingdate':tref.tradeDate.values,'f_val':y_pred}  )
    return ind,Xv

if __name__ == '__main__':    
    if len(sys.argv)>1:
        index_code = sys.argv[1]
        if len(sys.argv) >= 3:
            b_type = sys.argv[2]
        else:
            b_type = 'ls'
        if len(sys.argv) >= 4:
            method_sel = sys.argv[3]
        else:
            method_sel = 'index'
    else:
        index_code = '000001'
        b_type = 'ls'
        method_sel = 'index'
    print('%s-%s-%s' % (index_code,b_type,method_sel))
    #index_code = '399300'
    #index_code = '000905'
    #index_code = '000001'
    #index_code = '000016'
    #b_type = 'ls'
    #获取基础数据
    df=get_index_data(index=index_code)
    run_start = datetime.now()
    
    ind = pd.DataFrame()
    Xv = pd.DataFrame()
    for i in range(2010,2021):
        #startDate = date(i-5,1,1)
        startDate = date(2005,1,1)
        endDate = date(i+1,1,1)
        split_date= date(i,1,1)
        sub_ind,sub_Xv = gc_forest_val(df,startDate,endDate,split_date,b_type)
        ind = ind.append(sub_ind)
        Xv = Xv.append(sub_Xv)
       
       
    #信号延迟一天使用
    y_pred1 = ind.f_val.shift(1).values
    y_pred1[0] = 0
    
    print('*'*50)
    #print(y_pred)
    y = Xv['closeIndex'].pct_change().values
    y[0]=0
    
    yc=pd.DataFrame(data={'y_c':y*y_pred1},index=ind.tradingdate.values)
    
    yc.to_csv('gp_%s.csv' % index_code)
    
    plt.figure(figsize=(15, 5))
    plt.plot((yc+1).cumprod())
    plt.xticks(rotation=90)
    plt.title(index_code)
    
    run_end = datetime.now()
    print('total time : %s' % (run_end-run_start))