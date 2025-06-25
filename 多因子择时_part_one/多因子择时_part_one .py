
# coding: utf-8

# 此篇文档介绍了多因子择时的IC数据获取和weight数据的计算以及判断准则距离和p_value的计算 

# In[ ]:

#加载要用的包
import numpy as np
import pandas as pd
import scipy.stats as st
import numpy.linalg as nlg


# In[ ]:

factor_names = ['PE','ROE','RSI','NetProfitGrowRate']


# 日期的获取  月末日期 （周度也可以） 选取：20070101-20170101

# In[ ]:

data=DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=u"20070101",endDate='20170101',field=['calendarDate','isWeekEnd','isMonthEnd'],pandas="1")
data = data[data['isMonthEnd'] == 1]
#data = data[data['isWeekEnd'] == 1]
date_list = map(lambda x: x[0:4]+x[5:7]+x[8:10], data['calendarDate'].values.tolist())
print date_list


# In[ ]:

#施密特正交化函数，输入n个向量的dataframe，输出n个向量的dataframe　
def Schmidt(data):
    output = pd.DataFrame()
    mat = np.mat(data)
    output[0] = np.array(mat[:,0].reshape(len(data),))[0]
    for i in range(1,data.shape[1]):
        tmp = np.zeros(len(data))
        for j in range(i):
            up = np.array((mat[:,i].reshape(1,len(data)))*(np.mat(output[j]).reshape(len(data),1)))[0][0]
            down = np.array((np.mat(output[j]).reshape(1,len(data)))*(np.mat(output[j]).reshape(len(data),1)))[0][0]
            tmp = tmp+up*1.0/down*(np.array(output[j]))
        output[i] = np.array(mat[:,i].reshape(len(data),))[0]-np.array(tmp)
    output.index = data.index
    output.columns = data.columns
    return output


# In[ ]:

#定义一个计算因子IC的函数，输入当期日期(注意必须是每月最后一个交易日)，就可以得到因子IC值的dataframe，列名为四个因子
def get_currentIC(currentdate):
    print currentdate
    factor_api_field = ['ticker'] + factor_names
    lastdate = date_list[date_list.index(currentdate)-1]
    nextdate = date_list[date_list.index(currentdate)+1]   #要取下期的收益率序列
    factordata = DataAPI.MktStockFactorsOneDayGet(tradeDate=currentdate,secID=set_universe('A',currentdate),field=factor_api_field).set_index('ticker') #获取因子数据
    for i in range(len(factor_names)):
        signal = standardize(neutralize(winsorize(factordata[factor_names[i]].dropna().to_dict()),currentdate)) #去极值，标准化，中性化
        factordata[factor_names[i]][signal.keys()] = signal.values()
    factordata = factordata.dropna()
    factordata = Schmidt(factordata)                              #施密特正交化
    ############注意了 Return是股票的周或者月的回报率（不是每天的回报率）  所以date_list是周末日期是没错的 月末日期就不对了 月末用另外的函数
    #Return = DataAPI.MktEquwAdjGet(secID=set_universe('A',currentdate),beginDate=nextdate,endDate=nextdate,field=u"ticker,return").set_index('ticker') 周度
    Return = DataAPI.MktEqumAdjGet(secID=set_universe('A',currentdate),beginDate=nextdate,endDate=nextdate,field=u"ticker,return").set_index('ticker')#月度
    #print Return
    Return = pd.concat([factordata,Return],axis=1).dropna()
    IC = pd.DataFrame()
    #print IC
    index_re = pd.DataFrame()
    #print index_re
    for i in range(len(factor_names)):
        ic, p_value = st.pearsonr(Return[factor_names[i]],Return["return"])
        IC[factor_names[i]] = np.array([ic])                             #计算IC值，存入dataframe里
    
    #print lastdate    
    #下面计算指数的状态 可以增加判断的条件  比如加入换手率  （添加的判断条件　和　分类数目　均可按照自己的方式定义）
    aa=DataAPI.MktIdxdGet(tradeDate=lastdate,indexID=u"",ticker="000002",beginDate=u"",endDate=u"",exchangeCD=u"XSHG",field=u"",pandas="1")['preCloseIndex'].values              
    bb=DataAPI.MktIdxdGet(tradeDate=currentdate,indexID=u"",ticker="000002",beginDate=u"",endDate=u"",exchangeCD=u"XSHG",field=u"",pandas="1")['preCloseIndex'].values
    sign=(bb-aa)/aa
    if sign>0.01:
        cc=1
    elif sign<-0.01:
        cc=-1
    else:
        cc=0   
    #print cc
    index_re['index_re']=np.array([cc])
    #pd.merge(IC,index_re)
    IC.join(index_re)
    return IC.join(index_re)
get_currentIC(date_list[0])#取一个日期测试一下


# index_re -1代表当月跌幅大于1%  1代表当月涨幅大于1%  其余情况代表0 

# In[ ]:

IC_single_data=get_currentIC(date_list[0])
for i in range(1,len(date_list)-1):
    temp=get_currentIC(date_list[i])
    IC_single_data=IC_single_data.append(temp,ignore_index=True)
#filename=str(factor_names)+'weekly_IC_index_re_csv'
filename=str(factor_names)+'monthly_IC_index_re_csv'
IC_single_data.to_csv(filename)#十年间的因子IC数据存到本地


# In[ ]:

filename=str(factor_names)+'monthly_IC_index_re_csv'
aa=pd.read_csv(filename)


# In[ ]:

aa.head()


# In[ ]:

aa['PE'].hist(figsize=(12,6), bins=50)
aa['ROE'].hist(figsize=(12,6), bins=50)
aa['RSI'].hist(figsize=(12,6), bins=50)
aa['NetProfitGrowRate'].hist(figsize=(12,6), bins=50)


# 获取了IC数据 存储到了本地 下面根据这些数据计算weight数据 

# In[ ]:

#画图感受一下IC数据的效果 IC>0.1就证明预测效果很好了
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(18,6))
IC_single_data_monthly=pd.read_csv(str(factor_names)+'monthly_IC_index_re_csv')
ss=IC_single_data_monthly[factor_names]
ss.plot(title='index_full',figsize=(18,6))
s_0=IC_single_data_monthly[IC_single_data_monthly['index_re']==0][factor_names]
s_0.plot(title='index_re=0(left)',figsize=(18,6))
pd.rolling_mean(s_0, window=20,min_periods=10).plot(title='index_re=0(20 moving average)',figsize=(18,6))
s_1=IC_single_data_monthly[IC_single_data_monthly['index_re']==1][factor_names]
s_1.plot(title='index_re=1(left)',figsize=(18,6))
pd.rolling_mean(s_1, window=20,min_periods=10).plot(title='index_re=1(20 moving average)',figsize=(18,6))
s_2=IC_single_data_monthly[IC_single_data_monthly['index_re']==-1][factor_names]
s_2.plot(title='index_re=-1(left)',figsize=(18,6))
pd.rolling_mean(s_2, window=20,min_periods=10).plot(title='index_re=-1(20 moving average)',figsize=(18,6))


# In[ ]:

#日期对齐
print len(IC_single_data_monthly)
print len(date_list[:-1])
IC_single_data_monthly[IC_single_data_monthly.columns[0]]=date_list[:-1]
IC_single_data_monthly[-25:]


# In[ ]:

N = 12 #取前几期的因子IC值来计算当前的最优权重  (不是每天都计算IC 之前的datelist已经是周末了) 
def get_bestweight(date_pos):   #传入当前日期，得到当前日期及之前8期的数据所得到的最优权重
    #date = [date_list[date_list.index(currentdate)-i-1] for i in range(N)]  #取前8期日期
    IC = pd.DataFrame()
    for i in range(N):
        ic = IC_single_data_monthly[(factor_names)][date_pos+i:date_pos+i+1]    #计算每个日期的IC值 date_list中的IC值 这样对吗?
        IC = pd.concat([IC,ic],axis=0)
    #print IC#长度取决于N 就是根据这个来计算协方差矩阵
    
    mat = np.mat(IC.cov())                     #按照公式计算最优权重
    mat = nlg.inv(mat)
    weight = mat*np.mat(IC.mean()).reshape(len(mat),1)
    weight = np.array(weight.reshape(len(weight),))[0]
    return weight                          #返回最优权重值
#举个例子算一下
weight=pd.DataFrame([get_bestweight(0)])
weight.columns=factor_names
for i in range(1,len(date_list)-N):
    temp=pd.DataFrame([get_bestweight(i)])
    temp.columns=factor_names
    weight=weight.append(temp) 
weight.index=date_list[N-1:-1]
for i in range(0,len(weight)):#这里根据计算出来的权重 进行归一化
    sum_all=0
    for j in range(0,len(factor_names)):
        sum_all+=abs(weight[i:i+1])[factor_names[j]]
    #print type(sum_all[0])
    weight[i:i+1]=weight[i:i+1]/sum_all[0]
weight.to_csv(str(factor_names)+'_monthly_weight_12_none.csv')#把数据存在本地


# In[ ]:

factor_names = ['PE','ROE','RSI','NetProfitGrowRate']
a=pd.read_csv(str(factor_names)+'_monthly_weight_12_none.csv')
a[factor_names[0]].plot(figsize=(18,6))
a[factor_names[1]].plot(figsize=(18,6))
a[factor_names[2]].plot(figsize=(18,6))
a[factor_names[3]].plot(figsize=(18,6))


# 上面是无择时情况下计算的权重 下面计算择时情况下的权重  计算方法：首先根据index_re 把IC序列分成三个序列  然后获取当月的市场状态（根据index_re) 选取和此时状态相同的IC序列计算权重

# In[ ]:

#看一下三个序列的长度
ser_minus1=IC_single_data_monthly[IC_single_data_monthly['index_re']==-1]
ser_0=IC_single_data_monthly[IC_single_data_monthly['index_re']==0]
ser_1=IC_single_data_monthly[IC_single_data_monthly['index_re']==1]
print len(ser_minus1),len(ser_0),len(ser_1)


# In[ ]:

#下面是根据三个IC序列计算权重  程序跟上面的无择时的计算方式是一样的 只是序列不同（其实这里就有数据不够的情况  这个下面再讨论）
N = 12 
def get_bestweight(date_pos):   #传入当前日期，得到当前日期及之前8期的数据所得到的最优权重
    #date = [date_list[date_list.index(currentdate)-i-1] for i in range(N)]  #取前8期日期
    IC = pd.DataFrame()
    for i in range(N):
        ic = ser_1[(factor_names)][date_pos+i:date_pos+i+1]    #计算每个日期的IC值 date_list中的IC值 这样对吗?
        IC = pd.concat([IC,ic],axis=0)
    #print IC#长度取决于N 就是根据这个来计算协方差矩阵
    
    mat = np.mat(IC.cov())                     #按照公式计算最优权重
    mat = nlg.inv(mat)
    weight = mat*np.mat(IC.mean()).reshape(len(mat),1)
    weight = np.array(weight.reshape(len(weight),))[0]
    return weight                          #返回最优权重值
#举个例子算一下
import numpy as np
weight=pd.DataFrame([get_bestweight(0)])
weight.columns=factor_names
#print weight
for i in range(1,len(ser_1)-N):
    temp=pd.DataFrame([get_bestweight(i)])
    temp.columns=factor_names
    weight=weight.append(temp) 
weight.index=list(ser_1[ser_1.columns[0]])[N:]

for i in range(0,len(weight)):
    sum_all=0
    for j in range(0,len(factor_names)):
        sum_all+=abs(weight[i:i+1])[factor_names[j]]
    #print sum_all.values[0]
    weight[i:i+1]=weight[i:i+1]/sum_all[0]
weight_ser_1=weight
weight_ser_1.tail()


# In[ ]:

N = 12 #取前几期的因子IC值来计算当前的最优权重  是不是和调仓周期有关？ (不是每天都计算IC 之前的datelist已经是周末了) 确实是看你的调仓周期
def get_bestweight(date_pos):   #传入当前日期，得到当前日期及之前8期的数据所得到的最优权重
    #date = [date_list[date_list.index(currentdate)-i-1] for i in range(N)]  #取前8期日期
    IC = pd.DataFrame()
    for i in range(N):
        ic = ser_minus1[(factor_names)][date_pos+i:date_pos+i+1]    #计算每个日期的IC值 date_list中的IC值 这样对吗?
        IC = pd.concat([IC,ic],axis=0)
    #print IC#长度取决于N 就是根据这个来计算协方差矩阵
    
    mat = np.mat(IC.cov())                     #按照公式计算最优权重
    mat = nlg.inv(mat)
    weight = mat*np.mat(IC.mean()).reshape(len(mat),1)
    weight = np.array(weight.reshape(len(weight),))[0]
    return weight                          #返回最优权重值
#举个例子算一下
import numpy as np
weight=pd.DataFrame([get_bestweight(0)])
weight.columns=factor_names
for i in range(1,len(ser_minus1)-N):
    temp=pd.DataFrame([get_bestweight(i)])
    temp.columns=factor_names
    weight=weight.append(temp) 
weight.index=list(ser_minus1[ser_minus1.columns[0]])[N:]
for i in range(0,len(weight)):
    sum_all=0
    for j in range(0,len(factor_names)):
        sum_all+=abs(weight[i:i+1])[factor_names[j]]
    #print type(sum_all[0])
    weight[i:i+1]=weight[i:i+1]/sum_all.values[0]
weight_ser_minus1=weight
weight_ser_minus1.tail()


# In[ ]:

#实在没办法 选取了11 要不然没有数据
N = 11 
def get_bestweight(date_pos):   #传入当前日期，得到当前日期及之前8期的数据所得到的最优权重
    #date = [date_list[date_list.index(currentdate)-i-1] for i in range(N)]  #取前8期日期
    IC = pd.DataFrame()
    for i in range(N):
        ic = ser_0[(factor_names)][date_pos+i:date_pos+i+1]    #计算每个日期的IC值 date_list中的IC值 这样对吗?
        IC = pd.concat([IC,ic],axis=0)
    #print IC#长度取决于N 就是根据这个来计算协方差矩阵
    
    mat = np.mat(IC.cov())                     #按照公式计算最优权重
    mat = nlg.inv(mat)
    weight = mat*np.mat(IC.mean()).reshape(len(mat),1)
    weight = np.array(weight.reshape(len(weight),))[0]
    return weight                          #返回最优权重值
#举个例子算一下
import numpy as np
weight=pd.DataFrame([get_bestweight(0)])
weight.columns=factor_names
for i in range(1,len(ser_0)-N):
    temp=pd.DataFrame([get_bestweight(i)])
    temp.columns=factor_names
    weight=weight.append(temp) 
weight.index=list(ser_0[ser_0.columns[0]])[N:]
for i in range(0,len(weight)):
    sum_all=0
    for j in range(0,len(factor_names)):
        sum_all+=abs(weight[i:i+1])[factor_names[j]]
    #print type(sum_all[0])
    weight[i:i+1]=weight[i:i+1]/sum_all[0]
weight_ser_0=weight
weight_ser_0.head()


# In[ ]:

temp=pd.DataFrame(np.NAN,index=IC_single_data_monthly[IC_single_data_monthly.columns[0]],columns=IC_single_data_monthly.columns[1:])
#temp.combine_first(weight_ser_1).combine_first(weight_ser_0).combine_first(weight_ser_minus1)
temp.combine_first(weight_ser_1)


# In[ ]:

#数据存到本地
temp=pd.DataFrame(np.NAN,index=IC_single_data_monthly[IC_single_data_monthly.columns[0]],columns=IC_single_data_monthly.columns[1:])
temp=temp.combine_first(weight_ser_1).combine_first(weight_ser_0).combine_first(weight_ser_minus1)
temp.to_csv('monthly_weigt_12_11_choosing_time_csv')


# In[ ]:

temp=pd.read_csv('monthly_weigt_12_11_choosing_time_csv')
dis_1=temp[90:]
a=pd.read_csv(str(factor_names)+'_monthly_weight_12_none.csv')#  aim to get the last full data  due to the choosing time data have the problems 
dis_2=a[79:]
#change the index and rename the columns
dis_1.rename(columns={dis_1.columns[0]:'date'}, inplace = True)
dis_2.rename(columns={dis_2.columns[0]:'date'}, inplace = True)
dis_1=dis_1.set_index(np.arange(len(dis_1)))
dis_2=dis_2.set_index(np.arange(len(dis_2)))


# 下面是两个weight序列 计算distance  算法是：同一个日期的两个因子weight向量相减 绝对值求和/因子个数  代表了此时两个weight的距离

# In[ ]:

temp_minus=dis_1[factor_names]- dis_2[factor_names]
temp_minus=(np.abs(temp_minus))
f= lambda x:sum(x)/4
D=temp_minus.apply(f,axis=1)
D.plot()


# 下面是计算 两个合成IC序列（复合IC序列）的p_value  计算方法是：共用一个iC序列 根据不同的weight序列合成  得到两个序列再作统计量的检验

# In[ ]:

ser_1=ser_1.set_index(np.arange(len(ser_1)))
ser_0=ser_0.set_index(np.arange(len(ser_0)))
ser_minus1=ser_minus1.set_index(np.arange(len(ser_minus1)))#修改一下index方便加减
temp_IC=IC_single_data_monthly[-29:]#仅仅用到后面这些就够了
temp_IC=temp_IC.set_index(np.arange(len(temp_IC)))
dis_1['index_re']=temp_IC[-29:]['index_re']
te_b=dis_1[factor_names].set_index(np.arange(len(dis_1)))#择时权重
te_a=dis_2[factor_names].set_index(np.arange(len(dis_2)))#非择时权重
te_data=IC_single_data_monthly[-len(dis_1):][factor_names].set_index(np.arange(len(dis_1)))#IC序列


# In[ ]:

res=np.dot(te_a.as_matrix(),te_data.as_matrix().T)#get the diagonal element
np.shape(res)
IC_multi_normal=[]
for i in range(0,len(res)):
    IC_multi_normal.append(res[i,i])
print IC_multi_normal
res2=np.dot(te_b.as_matrix(),te_data.as_matrix().T)
IC_multi_ct=[]
for i in range(0,len(res2)):
    IC_multi_ct.append(res2[i,i])
print IC_multi_ct
plt.figure(figsize=(16,8))
ylist=range(0,len(IC_multi_normal))
plt.plot(ylist,IC_multi_normal,color='r')#红色的是正常的权重
plt.plot(ylist,IC_multi_ct)
from scipy.stats import ttest_ind
ttest_ind(IC_multi_normal,IC_multi_ct)# get the result p_value is very low t_value is very big  means they have the difference obviously 
#problem:
#(1):have less data  may 80+months  but we have three classes  only 29 data so the p_value we can only get 5 if we choose the window 24 
#(2):use the window_length 


# (1.936637918217383, 0.057841526435115635)是t值和p值    看到t值比较大 p值比较小 证明两个序列可以算是显著无关的  也就是说择时是有效果的

# In[ ]:

p_value=[]
for i in range(0,17):
    p_value.append( ttest_ind(IC_multi_normal[i:i+12],IC_multi_ct[i:12+i])[1])
D_draw=[]
for i in range(0,16):
#    print D[-16+i:-15+i]
    D_draw.append(D[-17+i:-16+i].values[0])
D_draw.append(D[-1:].values[0])


# In[ ]:

#p_value 
#D_draw
thr_p_value=[]
thr_D=[]
for i in range(0,len(p_value)):
    thr_p_value.append(0.15)
    thr_D.append(0.2)
plt.figure(figsize=(14,6))
ylist=range(0,len(thr_D))
plt.plot(ylist,thr_D,color='r')
plt.plot(ylist,D_draw,color='r')
plt.plot(ylist,p_value,color='g')
plt.plot(ylist,thr_p_value,color='g')


# 上面画出了距离和p值的序列图 绿线是p值 红线是距离  筛选的原则是：距离要大于阈值  p值要小于阈值   只有在这种情况下才可以认为择时是有效果的

# In[ ]:

df1=pd.DataFrame({'p_value':p_value,'D':D_draw})
df1.to_csv('monthly_d_p_value_12')#方便下次调用

