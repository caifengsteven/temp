
# coding: utf-8

# # 本文主要研究多因子动态权重调整的问题：
#   - 等权重
#   - IC加权平均值
#   - 回归系数加权
#   - 基于IC的IR最大化
#   - 基于协方差压缩IC的IR最大化
#   - 基于分辨效用
#   - 基于前期收益率  
#   
# 这是现在主要的动态调整方法，在此基础上还有股票池交集方法和基于前期的单调为权重，大家可以去研究，这里就不多赘述。水平有限，还请多多指教。  
# 由于本文主要介绍动态多因子的权重调整，且限于数据的缺失性，本文未做市值中性和行业中性，只进行了相关数据都做了极值处理和标准化的必要处理。

# ## 1.前期准备
# 
# 1.股票池为中证500
# 
# 2.数据时间选取 20140101-20170101
# 
# 3.本文选取了 'momentum','pe_ttm','roe_ttm','size','yoyroe','ps'六个因子
# 
# 4.对数据进行清洗、去极值、标准化

# ### 1.1 库的导入和数据的读取以及数据处理的函数

# In[ ]:

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
import seaborn as sns

                #极值处理
def medianpeak(x):
    y=np.copy(x)
    ymedian=np.median(y)
    zmid=np.median(np.abs(y-np.median(y)))
    y[y<=ymedian-5.2*zmid]=(ymedian-5.2*zmid)
    y[y>=ymedian+5.2*zmid]=(ymedian+5.2*zmid)
    return y

                #标准化
def nondimensionalize(x):
    y=np.copy(x)
    ymean=np.mean(y)
    ystd=np.std(y)
    y=(y-ymean)/ystd
    return y

                #选取的因子
key=['momentum','pe_ttm','roe_ttm','size','yoyroe','ps']  

                #读取相应数据
closeprice=pd.read_csv('closeprice.csv', encoding='GBK')
iszz500=pd.read_csv('isZZ500.csv', encoding='GBK')
maxupdown=pd.read_csv('maxupordown.csv', encoding='GBK')
iszz500=iszz500.T
iszz500.index=maxupdown.T.index
iszz500=iszz500.T
data={}
for i in range(len(key)):
    data[key[i]]= pd.read_csv(key[i]+'.csv', encoding='GBK')


# ### 1.2 参数的调整
# - 调仓周期
# - 分组数
# - 起始时间

# In[ ]:

t_step=20 #调仓周期
groupnum=10.0 #分组数
sart_time=0 #开始时间
date_range=closeprice.index[0:len(closeprice):t_step] #整个时间段
IClong=int(120.0/t_step) # IC的时间长度周期
rate=(closeprice-closeprice.shift(t_step))/closeprice.shift(t_step)# 收益率


# ## 标准——等权多因子

# In[ ]:

group={}
for j in range(len(date_range)-1):
    z500=iszz500[date_range[j]:date_range[j]+1].T[date_range[j]]
    zz500=z500[z500>0].index
    updown0=maxupdown[date_range[j]:date_range[j]+1].T[date_range[j]]
    updown=updown0[updown0==0].index
    index=zz500 & updown
    group[j]={}
    for i in range(len(key)):
        vaild_data=data[key[i]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize)
        vaild_data=vaild_data
        if i ==0:
            sdata=vaild_data
        else:
            sdata=sdata+vaild_data
    sdata=sdata.dropna().sort_values(by=date_range[j],ascending=False)
    for groupn in range(int(groupnum)):
        group[j][groupn]=sdata.iloc[int(groupn/groupnum*len(sdata)):int((groupn+1)/groupnum*len(sdata))].index


# In[ ]:

# 每组每期收益率
nrate={}
for i in range(int(groupnum)):
    nrate0=[]
    for j in range(IClong+1,len(date_range)-1):
         nrate0.append(rate[date_range[j+1]-2:date_range[j+1]-1][group[j][i]].T.mean().values[0])
    nrate[i]=pd.DataFrame(nrate0)
# 每组平均收益率
m=[]
for i in range(int(groupnum)): 
    m.append(nrate[i].mean().values[0])


# In[ ]:

# 每组平均收益率图
pd.DataFrame(m).plot(kind='bar',color=['violet','moccasin','cyan','#FFC0CB'])
plt.show()


# In[ ]:

# 每组单位净值图
for i in range(int(groupnum)):
    p_data0=pd.DataFrame(nrate[i])
    if i==0:
        p_data= p_data0
    else:
        p_data= pd.concat([p_data,p_data0],axis=1,join='inner')
p_data=p_data.T
p_data.index=np.arange(10)
p_data=p_data.T
p_data=(p_data+1).cumprod()
p_data.plot(figsize=(18,10),color=['fuchsia','hotpink','blueviolet','dodgerblue','cyan','palegreen','springgreen','coral','aquamarine' ,'orangered'])


# ## 2. IC的加权平均

# ### 2.1说明
# 本文中的IC是本期因子值和下期股票收益的皮尔逊相关系数。  
# 主要因为是按20个交易日的月调仓，所以本文是取前六个月的IC平均值作为本期的权重。

# In[ ]:

group={}
IC0={}
for j in range(len(date_range)-1):
    z500=iszz500[date_range[j]:date_range[j]+1].T[date_range[j]]
    zz500=z500[z500>0].index
    updown0=maxupdown[date_range[j]:date_range[j]+1].T[date_range[j]]
    updown=updown0[updown0==0].index
    index=zz500 & updown
    group[j]={}
    if j<IClong:
        for i in range(len(key)):
            vaild_data=data[key[i]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize)
            vaild_data=vaild_data
            if i ==0:
                sdata=vaild_data
            else:
                sdata=sdata+vaild_data
        sdata=sdata.dropna().sort_values(by=date_range[j],ascending=False)
    else:
        for i in range(len(key)):
            vaild_data=data[key[i]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize)
            vaild_data=vaild_data*pd.DataFrame(IC0).T.rolling(IClong).mean()[-1:][i].values[0]
            if i ==0:
                sdata=vaild_data
            else:
                sdata=sdata+vaild_data
        sdata=sdata.dropna().sort_values(by=date_range[j],ascending=False)
    for groupn in range(int(groupnum)):
        group[j][groupn]=sdata.iloc[int(groupn/groupnum*len(sdata)):int((groupn+1)/groupnum*len(sdata))].index
        if groupn==9:
            ic=[]
            for m in range(len(key)):
                mm=data[key[m]][index][date_range[j]:date_range[j]+1].T.dropna().T
                m_rate=rate[index][date_range[j+1]:date_range[j+1]+1].T.dropna().T
                a_index=mm.T.dropna().index & m_rate.T.dropna().index
                ic.append(np.corrcoef(m_rate[a_index],mm[a_index])[0][1])
            IC0[j]=ic


# ### 2.2 各分组收益率的计算

# In[ ]:

nrate={}
for i in range(int(groupnum)):
    nrate0=[]
    for j in range(IClong+1,len(date_range)-1):
         nrate0.append(rate[date_range[j+1]-2:date_range[j+1]-1][group[j][i]].T.mean().values[0])
    nrate[i]=pd.DataFrame(nrate0)

m=[]
for i in range(int(groupnum)): 
    m.append(nrate[i].mean().values[0])


# ### 2.3 每组平均收益率和单位净值

# In[ ]:

pd.DataFrame(m).plot(kind='bar',color=['violet','moccasin','cyan','#FFC0CB'])
plt.show()


# In[ ]:

for i in range(int(groupnum)):
    p_data0=pd.DataFrame(nrate[i])
    if i==0:
        p_data= p_data0
    else:
        p_data= pd.concat([p_data,p_data0],axis=1,join='inner')
p_data=p_data.T
p_data.index=np.arange(10)
p_data=p_data.T
p_data=(p_data+1).cumprod()
p_data.plot(figsize=(18,10),color=['fuchsia','hotpink','blueviolet','dodgerblue','cyan','palegreen','springgreen','coral','aquamarine' ,'orangered'])


# In[ ]:

IC=pd.DataFrame(IC0)
IC.index=key
IC=IC.T.rolling(IClong).mean().dropna()
IC=IC.T
#plt.figure(figsize=(3, 3)) 
IC.T.plot(figsize=(18,10),color=['fuchsia','hotpink','blueviolet','dodgerblue','cyan','palegreen'])
plt.show()


# ## 3.回归系数加权

# ### 3.1 说明
# - 本文取收益率与前期的因子值多元回归的系数，六个月的平均值为权重
# $$Y=X\beta+e$$
# $Y$为收益率，$X$为相应的前期因子值，$\beta$回归系数，本文用到的是最小二乘法，建议使用广义加权最小二乘法$GLS$

# In[ ]:

group={}
IC0={}
for j in range(len(date_range)-1):
    z500=iszz500[date_range[j]:date_range[j]+1].T[date_range[j]]
    zz500=z500[z500>0].index
    updown0=maxupdown[date_range[j]:date_range[j]+1].T[date_range[j]]
    updown=updown0[updown0==0].index
    index=zz500 & updown
    group[j]={}
    if j<IClong:
        for i in range(len(key)):
            vaild_data=data[key[i]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize)
            vaild_data=vaild_data
            if i ==0:
                sdata=vaild_data
            else:
                sdata=sdata+vaild_data
        sdata=sdata.dropna().sort_values(by=date_range[j],ascending=False)
    else:
        for i in range(len(key)):
            vaild_data=data[key[i]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize)
            vaild_data=vaild_data*pd.DataFrame(IC0).T.rolling(IClong).mean()[-1:][key[i]].values[0]
            if i ==0:
                sdata=vaild_data
            else:
                sdata=sdata+vaild_data
        sdata=sdata.dropna().sort_values(by=date_range[j],ascending=False)
    for groupn in range(int(groupnum)):
        group[j][groupn]=sdata.iloc[int(groupn/groupnum*len(sdata)):int((groupn+1)/groupnum*len(sdata))].index
        if groupn==9:
            ic=[]
            for m in range(len(key)):
                m_factor=data[key[m]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize).T
                if m==0:
                    m_index=m_factor.T.index
                else:
                    m_index= m_index & m_factor.T.index
            for m in range(len(key)):
                m_factor=data[key[m]][m_index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize).T
                if m==0:
                    mm=m_factor
                else:
                    mm=mm.append(m_factor)
            m_rate=rate[index][date_range[j+1]:date_range[j+1]+1].T.dropna().T
            Y=m_rate[m_index].T
            mx=mm
            mx.index=key
            X=mx.T
            X=sm.add_constant(X)
            model=sm.OLS(Y,X)
            result=model.fit()
            IC0[j]=result.params


# ### 3.2 各分组收益率的计算

# In[ ]:

nrate={}
for i in range(int(groupnum)):
    nrate0=[]
    for j in range(IClong+1,len(date_range)-1):
         nrate0.append(rate[date_range[j+1]-2:date_range[j+1]-1][group[j][i]].T.mean().values[0])
    nrate[i]=pd.DataFrame(nrate0)

m=[]
for i in range(int(groupnum)): 
    m.append(nrate[i].mean().values[0])


# ### 3.3 每组平均收益率和单位净值

# In[ ]:

pd.DataFrame(m).plot(kind='bar',color=['violet','moccasin','cyan','#FFC0CB'])
plt.show()


# In[ ]:

for i in range(int(groupnum)):
    p_data0=pd.DataFrame(nrate[i])
    if i==0:
        p_data= p_data0
    else:
        p_data= pd.concat([p_data,p_data0],axis=1,join='inner')
p_data=p_data.T
p_data.index=np.arange(10)
p_data=p_data.T
p_data=(p_data+1).cumprod()
p_data.plot(figsize=(18,10),color=['fuchsia','hotpink','blueviolet','dodgerblue','cyan','palegreen','springgreen','coral','aquamarine' ,'orangered'])


# ## 4.基于IC的IR最大化

# 4.1 说明
# 其IC的均值向量为 $\vec{\mathrm{IC}} = (\overline{\mathrm{IC}_1},\overline{\mathrm{IC}_2},\cdots,\overline{\mathrm{IC}_M})^T$，IC的协方差矩阵为 $\Sigma$。如果各因子权重向量为 $\vec{v}=(\overline{V_1},\overline{V_2},\cdots,\overline{V_M})^T$，则复合因子的IR值为：
# $$\mathrm{IR} =\frac{v^T\*\vec{\mathrm{IC}}} {\sqrt{v^T\* \Sigma\*v}}$$
# 求导以后，可以最优解为：
# $$v^\* = \delta \*\Sigma^{-1}\*\vec{\mathrm{IC}}$$
# 此处引用来自《MultiFactors Alpha Model - 基于因子IC的多因子合成0》

# ### 4.2 每日IC的计算
# 滚动窗口为120天，计算每一天的IC，使用了之前6个月的IC时间序列来计算IC均值向量和IC协方差矩阵

# In[ ]:

np.dot(pd.DataFrame(np.linalg.inv(IC.T.corr())).as_matrix(),pd.DataFrame(IC.T.mean()).values)
ir_rate=(closeprice-closeprice.shift(1))/closeprice.shift(1)
ir_rate=ir_rate[1:]
IR={}
for i in range(len(ir_rate)):
    z500=iszz500[date_range[j]:date_range[j]+1].T[date_range[j]]
    zz500=z500[z500>0].index
    updown0=maxupdown[date_range[j]:date_range[j]+1].T[date_range[j]]
    updown=updown0[updown0==0].index
    index=zz500 & updown
    ir=[]
    for j in range(len(key)):
        ir_index=data[key[j]][index][i:i+1].T.dropna().index
        ir.append(np.corrcoef(data[key[j]][index][i:i+1].T.dropna().T,ir_rate[ir_index][i:i+1])[0][1])
    IR[i]=ir
IR0=pd.DataFrame(IR).T
IR0.index=ir_rate.index
IR0=IR0.T
IR0.index=key
IR0=IR0.T


# In[ ]:

group={}
IC0={}
for j in range(len(date_range)-1):
    z500=iszz500[date_range[j]:date_range[j]+1].T[date_range[j]]
    zz500=z500[z500>0].index
    updown0=maxupdown[date_range[j]:date_range[j]+1].T[date_range[j]]
    updown=updown0[updown0==0].index
    index=zz500 & updown
    group[j]={}
    if j<IClong:
        for i in range(len(key)):
            vaild_data=data[key[i]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize)
            vaild_data=vaild_data
            if i ==0:
                sdata=vaild_data
            else:
                sdata=sdata+vaild_data
        sdata=sdata.dropna().sort_values(by=date_range[j],ascending=False)
    else:
        maxir= pd.DataFrame(np.dot(pd.DataFrame(np.linalg.inv(IR0[date_range[j]-119:date_range[j]].corr())).as_matrix(),pd.DataFrame(IR0[date_range[j]-119:date_range[j]].mean()).values)).T.values[0]
        for i in range(len(key)):
            vaild_data=data[key[i]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize)
            vaild_data=vaild_data*maxir[i]
            if i ==0:
                sdata=vaild_data
            else:
                sdata=sdata+vaild_data
        sdata=sdata.dropna().sort_values(by=date_range[j],ascending=False)
    for groupn in range(int(groupnum)):
        group[j][groupn]=sdata.iloc[int(groupn/groupnum*len(sdata)):int((groupn+1)/groupnum*len(sdata))].index


# ### 4.3 各分组收益率的计算

# In[ ]:

nrate={}
for i in range(int(groupnum)):
    nrate0=[]
    for j in range(IClong+1,len(date_range)-1):
         nrate0.append(rate[date_range[j+1]-2:date_range[j+1]-1][group[j][i]].T.mean().values[0])
    nrate[i]=pd.DataFrame(nrate0)

m=[]
for i in range(int(groupnum)): 
    m.append(nrate[i].mean().values[0])


# ### 4.4 每组平均收益率和单位净值

# In[ ]:

pd.DataFrame(m).plot(kind='bar',color=['violet','moccasin','cyan','#FFC0CB'])
plt.show()


# In[ ]:

for i in range(int(groupnum)):
    p_data0=pd.DataFrame(nrate[i])
    if i==0:
        p_data= p_data0
    else:
        p_data= pd.concat([p_data,p_data0],axis=1,join='inner')
p_data=p_data.T
p_data.index=np.arange(10)
p_data=p_data.T
p_data=(p_data+1).cumprod()
p_data.plot(figsize=(18,10),color=['fuchsia','hotpink','blueviolet','dodgerblue','cyan','palegreen','springgreen','coral','aquamarine' ,'orangered'])


# ## 5.基于压缩方差IC的IR最大化

# ### 5.1 说明
# 压缩的协方差估计：
# $$\hat \Sigma_{\mathrm{shrink}} = \lambda \Phi + (1-\lambda)*\hat{\Sigma}$$
# 此处引用来自《MultiFactors Alpha Model - 基于因子IC的多因子合成0》

# In[ ]:

lw = LedoitWolf()
group={}
IC0={}
for j in range(len(date_range)-1):
    z500=iszz500[date_range[j]:date_range[j]+1].T[date_range[j]]
    zz500=z500[z500>0].index
    updown0=maxupdown[date_range[j]:date_range[j]+1].T[date_range[j]]
    updown=updown0[updown0==0].index
    index=zz500 & updown
    group[j]={}
    if j<IClong+1:
        for i in range(len(key)):
            vaild_data=data[key[i]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize)
            vaild_data=vaild_data
            if i ==0:
                sdata=vaild_data
            else:
                sdata=sdata+vaild_data
        sdata=sdata.dropna().sort_values(by=date_range[j],ascending=False)
    else:
        maxir=pd.DataFrame(np.dot(pd.DataFrame(np.linalg.inv(lw.fit(IR0[date_range[j]-120:date_range[j]-1].corr().as_matrix()).covariance_)),pd.DataFrame(IR0[date_range[j]-120:date_range[j]-1].mean()).values)).T.values[0]
        for i in range(len(key)):
            vaild_data=data[key[i]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize)
            vaild_data=vaild_data*maxir[i]
            if i ==0:
                sdata=vaild_data
            else:
                sdata=sdata+vaild_data
        sdata=sdata.dropna().sort_values(by=date_range[j],ascending=False)
    for groupn in range(int(groupnum)):
        group[j][groupn]=sdata.iloc[int(groupn/groupnum*len(sdata)):int((groupn+1)/groupnum*len(sdata))].index


# ### 5.2 各分组收益率的计算

# In[ ]:

nrate={}
for i in range(int(groupnum)):
    nrate0=[]
    for j in range(IClong+1,len(date_range)-1):
         nrate0.append(rate[date_range[j+1]-2:date_range[j+1]-1][group[j][i]].T.mean().values[0])
    nrate[i]=pd.DataFrame(nrate0)

m=[]
for i in range(int(groupnum)): 
    m.append(nrate[i].mean().values[0])


# ### 5.3 每组平均收益率和单位净值

# In[ ]:

pd.DataFrame(m).plot(kind='bar',color=['violet','moccasin','cyan','#FFC0CB'])
plt.show()


# In[ ]:

for i in range(int(groupnum)):
    p_data0=pd.DataFrame(nrate[i])
    if i==0:
        p_data= p_data0
    else:
        p_data= pd.concat([p_data,p_data0],axis=1,join='inner')
p_data=p_data.T
p_data.index=np.arange(10)
p_data=p_data.T
p_data=(p_data+1).cumprod()
p_data.plot(figsize=(18,10),color=['fuchsia','hotpink','blueviolet','dodgerblue','cyan','palegreen','springgreen','coral','aquamarine' ,'orangered'])


# ## 6.基于选股因子边际效用和有效分散的动态区分度动量策略

# ### 6.1 说明
# 单因子第一组收益率与单因子最后一组收益率差值比上当期收益率第一组与最后一组的差值的比值为权重。考虑到延续性，在边际效用的基础上以6个月滚动加权平均。
# $$ weight= \frac{{(factor.firstrate-factor.lastrate)}}{{(rate.first-rate.last)} }$$

# In[ ]:

group={}
AC={}
for j in range(len(date_range)-1):
    z500=iszz500[date_range[j]:date_range[j]+1].T[date_range[j]]
    zz500=z500[z500>0].index
    updown0=maxupdown[date_range[j]:date_range[j]+1].T[date_range[j]]
    updown=updown0[updown0==0].index
    index=zz500 & updown
    group[j]={}
    if j<IClong:
        for i in range(len(key)):
            vaild_data=data[key[i]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize)
            vaild_data=vaild_data
            if i ==0:
                sdata=vaild_data
            else:
                sdata=sdata+vaild_data
        sdata=sdata.dropna().sort_values(by=date_range[j],ascending=False)
    else:
        for i in range(len(key)):
            vaild_data=data[key[i]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize)
            vaild_data=vaild_data*pd.DataFrame(AC).T.rolling(IClong).mean()[-1:][i].values[0]
            if i ==0:
                sdata=vaild_data
            else:
                sdata=sdata+vaild_data
        sdata=sdata.dropna().sort_values(by=date_range[j],ascending=False)
    a_rate=rate[index][date_range[j+1]:date_range[j+1]+1].T.dropna().sort_values(by=date_range[j+1],ascending=False)
    stand_rate=a_rate.iloc[int(0/10.0*len(sdata)):int((0+1)/10.0*len(sdata))].mean().values[0]-a_rate.iloc[int(9/10.0*len(sdata)):int((9+1)/10.0*len(sdata))].mean().values[0]
    for groupn in range(int(groupnum)):
        group[j][groupn]=sdata.iloc[int(groupn/groupnum*len(sdata)):int((groupn+1)/groupnum*len(sdata))].index
        if groupn==9:
            ac=[]
            for m in range(len(key)):
                a_data=data[key[m]][index][date_range[j]:date_range[j]+1].T.dropna().sort_values(by=date_range[j],ascending=False)
                head_data=a_data.iloc[int(0/groupnum*len(sdata)):int((0+1)/groupnum*len(sdata))].index
                tail_data=a_data.iloc[int((groupnum-1)/groupnum*len(sdata)):int(groupnum/groupnum*len(sdata))].index
                m_percent=(rate[index][date_range[j+1]:date_range[j+1]+1].T.dropna().T[head_data].T.mean().values[0]-rate[index][date_range[j+1]:date_range[j+1]+1].T.dropna().T[tail_data].T.mean().values[0])/stand_rate
                ac.append(m_percent)
            AC[j]=ac


# ### 6.2 各分组收益率的计算

# In[ ]:

nrate={}
for i in range(int(groupnum)):
    nrate0=[]
    for j in range(IClong+1,len(date_range)-1):
         nrate0.append(rate[date_range[j+1]-2:date_range[j+1]-1][group[j][i]].T.mean().values[0])
    nrate[i]=pd.DataFrame(nrate0)

m=[]
for i in range(int(groupnum)): 
    m.append(nrate[i].mean().values[0])


# ### 6.3 每组平均收益率和单位净值

# In[ ]:

pd.DataFrame(m).plot(kind='bar',color=['violet','moccasin','cyan','#FFC0CB'])
plt.show()


# In[ ]:

for i in range(int(groupnum)):
    p_data0=pd.DataFrame(nrate[i])
    if i==0:
        p_data= p_data0
    else:
        p_data= pd.concat([p_data,p_data0],axis=1,join='inner')
p_data=p_data.T
p_data.index=np.arange(10)
p_data=p_data.T
p_data=(p_data+1).cumprod()
p_data.plot(figsize=(18,10),color=['fuchsia','hotpink','blueviolet','dodgerblue','cyan','palegreen','springgreen','coral','aquamarine' ,'orangered'])


# ## 7. 取收益率最好前三组等权平均

# ### 7.1 说明
# 取每期最后一组收益率最高的前三个因子，等权相加，柔和成下一期因子

# In[ ]:

group={}
AC={}
for j in range(len(date_range)-1):
    z500=iszz500[date_range[j]:date_range[j]+1].T[date_range[j]]
    zz500=z500[z500>0].index
    updown0=maxupdown[date_range[j]:date_range[j]+1].T[date_range[j]]
    updown=updown0[updown0==0].index
    index=zz500 & updown
    group[j]={}
    if j<IClong:
        for i in range(len(key)):
            vaild_data=data[key[i]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize)
            vaild_data=vaild_data
            if i ==0:
                sdata=vaild_data
            else:
                sdata=sdata+vaild_data
        sdata=sdata.dropna().sort_values(by=date_range[j],ascending=False)
    else:
        for i in range(len(key)):
            vaild_data=data[key[i]][index][date_range[j]:date_range[j]+1].T.dropna().apply(medianpeak).apply(nondimensionalize)
            vaild_data=vaild_data*ac[key[i]].values[0]
            if i ==0:
                sdata=vaild_data
            else:
                sdata=sdata+vaild_data
        sdata=sdata.dropna().sort_values(by=date_range[j],ascending=False)
    for groupn in range(int(groupnum)):
        group[j][groupn]=sdata.iloc[int(groupn/groupnum*len(sdata)):int((groupn+1)/groupnum*len(sdata))].index
        if groupn==9:
            ac=[]
            for m in range(len(key)):
                a_data=data[key[m]][index][date_range[j]:date_range[j]+1].T.dropna().sort_values(by=date_range[j],ascending=False)
                tail_data=a_data.iloc[int((groupnum-1)/groupnum*len(sdata)):int(groupnum/groupnum*len(sdata))].index
                a_rate=rate[index][date_range[j+1]:date_range[j+1]+1].T.dropna().T[tail_data].T.mean().values[0]
                ac.append(a_rate)
            AC[j]=ac
            ac=pd.DataFrame(ac)
            ac.index=key
            ac=ac.sort_values(by=0,ascending=False)
            ac[0:3]=1
            ac[3:6]=0
            ac=ac.T


# ### 7.2 各分组收益率的计算

# In[ ]:

nrate={}
for i in range(int(groupnum)):
    nrate0=[]
    for j in range(IClong+1,len(date_range)-1):
         nrate0.append(rate[date_range[j+1]-2:date_range[j+1]-1][group[j][i]].T.mean().values[0])
    nrate[i]=pd.DataFrame(nrate0)

m=[]
for i in range(int(groupnum)): 
    m.append(nrate[i].mean().values[0])


# ### 7.3 每组平均收益率和单位净值

# In[ ]:

pd.DataFrame(m).plot(kind='bar',color=['violet','moccasin','cyan','#FFC0CB'])
plt.show()


# In[ ]:

for i in range(int(groupnum)):
    p_data0=pd.DataFrame(nrate[i])
    if i==0:
        p_data= p_data0
    else:
        p_data= pd.concat([p_data,p_data0],axis=1,join='inner')
p_data=p_data.T
p_data.index=np.arange(10)
p_data=p_data.T
p_data=(p_data+1).cumprod()
p_data.plot(figsize=(18,10),color=['fuchsia','hotpink','blueviolet','dodgerblue','cyan','palegreen','springgreen','coral','aquamarine' ,'orangered'])


# ## 8.小结
# 本文主要介绍了一些权重的选取方法，当然此中又很多不足之处，比如IC的延续性研究，如果更换调仓周期，IC的移动平均周期，是否会发生改变？回归如果出现遗漏变量，回归是有偏，又改如何？，等等的问题。还望大家不吝赐教。（如果大家需要相关数据，请联系，到时候会放在网盘上）

# In[ ]:



