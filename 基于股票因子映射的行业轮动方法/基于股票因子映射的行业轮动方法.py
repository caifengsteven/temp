# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:19:18 2020

@author: Asus
"""


'''
导读
A. 研究目的：本文基于优矿提供的因子数据构建个股因子并探索将个股因子映射到行业构建行业轮动模型的方法，文中部分方法参考光大证券《基于股票因子映射的行业轮动方法》（原作者：胡骥聪）中的研究方法，通过实证分析该方法在行业轮动中的适用性。

B. 文章结构：本文共分为4个部分，具体如下

一、数据准备和处理：个股因子映射到行业简介；利用uqer API获取研究所用的行情数据、个股因子数据、财务数据、行业分类数据；计算行业指数；

二、行业因子计算：个股因子映射到行业得到行业因子；

三、行业因子测试：结合行业收益率数据，对单因子和合成后的因子进行测试，测试内容包括计算IC、分组、多空收益等；

四、总结：对基于股票因子映射的行业轮动方法进行总结；

C. 研究结论：

基于优矿提供的因子数据构建个股因子并探索将个股因子映射到行业构建行业轮动模型的方法，选出的单因子在行业轮动中表现均比较好，合成后的因子表现好于单因子的表现，周度IC为6.8%，IR为1.39，多空年化收益为43.3%， 相比全A指数年化超额收益达到44%。
D. 时间说明

本文共有四个部分，第一部分约耗时9分钟，第二部分约耗时20分钟，其它部分耗时均在5分钟以内，总耗时在35分钟以内
特别说明
为便于阅读，本文将部分和文章主题无关的函数放在函数库里面：
https://uqer.datayes.com/community/share/eLNeQy0p3r0lRu9I5WoZ5YOw2ng0/private；密码：6278。
请前往查看并注意保密。请在运行之前，克隆上面的代码，并存成lib(右上角->另存为lib，不要修改名字)

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''

#coding:utf-8
import pandas as pd
import time
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import matplotlib as mpl
from CAL.PyCAL import *
mpl.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei'] 
import seaborn as sns
import lib.quant_tool as quant_util
from scipy import stats
import numpy as np
import gevent
import seaborn 

'''

第一部分：数据准备和处理
该部分耗时 9分钟
该部分内容为：

个股因子映射到行业简介；

通过API取出所需要的数据:包括行情数据、个股因子数据、财务数据、行业分类数据;

计算行业指数；

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
1.1 个股因子映射到行业简介

个股结构层面的轮动实际就是选股，行业结构层面或风格结构层面上即是业界中的“行业轮动”或“风格轮动”。在个股层面上，多因子选股是主流的“轮动”方式，每个alpha 因子实际上都从一个角度表达了对股票截面收益的观点。由于每个行业都是由其相应的成分股组成，那么通过整合其成分股的观点来作为整个行业的观点，理论上也能达到行业层面的“轮动”效果。
在选取个股因子构建行业因子时，因子在逻辑上要在行业层面可比。我们挑选了20多个在个股层面有较强选股效果的因子作为底层因子进行尝试， 选出了5个个具有轮动能力的指标：

1)GREV:分析师盈利预测变化趋势，过去60个交易日内的DAREV符号加和；

2)FiftyTwoWeekHigh:当前价格处于过去1年股价的位置, 计算方法为个股当前价格与过去1年股价最小值除以过去1年股价最大值和最小值之差；

3)REVS250: 过去1年的价格动量；

4)gm_y: 毛利率同比；

5)dtar_y: 资产负债率同比；

上述5个因子在构建逻辑上去不受行业的影响， 因而在行业之间可以直接比较。

   调试 运行
文档
 代码  策略  文档
1.2 通过API取出所需要的数据

‘GREV’,‘FiftyTwoWeekHigh’,'REVS250’可以直接通过API取到数据，‘gm_y’、'dtar_y’两个因子在计算时需要先获取对应的财务项，然后再基于财务数据衍生得到

'''
start_date = "2010-01-01"
end_date = "2020-05-31"
freq="week"
dates = pd.date_range(start_date,end_date,fred="D").astype(str)
calendar_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
week_end_list = calendar_df[calendar_df['isWeekEnd']==1]['calendarDate'].values
trade_date_list = calendar_df[calendar_df['isOpen']==1]['calendarDate'].values
month_end_list = calendar_df[calendar_df['isMonthEnd']==1]['calendarDate'].values


# 取数据： GREV、fiftytwowekhigh、revs250直接从API读取
t0 = time.time()
factor_df_list  = []
factor_list1 = ['GREV','FiftyTwoWeekHigh','REVS250']
i = 0
for td in week_end_list:
    if i%100==0:
        print '当前日期： %s-------------'%td
    i = i + 1    
    factor_dfi = DataAPI.MktStockFactorsOneDayProGet(tradeDate=td,secID=u"",ticker=u"",field=['ticker','tradeDate']+factor_list1,pandas="1")
    factor_df_list.append(factor_dfi)
factor_df = pd.concat(factor_df_list,axis=0)
print ('该部分耗时: %s 秒！'%(time.time() - t0))
print (factor_df.head().to_html())

#设置数据目录:可把读取的因子数据存下来，下次运行可直接读取本地文件
raw_data_dir = "./bottom_up"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)
factor_df.to_csv('%s/raw_factors.csv'%raw_data_dir,index=False,encoding='utf-8')   
#如果存储了数据，则可以直接读取即可
# factor_df = pd.read_csv('%s/raw_factors1.csv'%raw_data_dir,dtype={'ticker':str})


#计算连续几个季度的指标数据
def get_findata_nlatest(fin_data_frame, n, col_name='value', isannual=False):
    """
    :param fin_data_frame: df, column= ['ticker','pub_date','end_date',[col_name]], index=num, pub_date='%Y%m%d'
    :param n: int， n>=0, 计算多少期
    :param col_name: str, 财务字段名称
    :param isannual: bool, 为True时，取最近几期年报数据; 为False时，取最近几期财报数据（包括季度报告）。默认为False。
    :return: df, column= ['ticker','pub_date', 'end_date', 'value_0','value_1','value_2','value_3',...,'value_N'], 'ticker','pub_date' 是唯一性约束
    """
    fin_df = fin_data_frame[['ticker', 'pub_date', 'end_date', col_name]]
    fin_df.dropna(inplace=True)
    pub_list = ['pub_date_' + str(i) for i in range(n)]
    value_list = ['value_' + str(i) for i in range(n)]

    def get_end_date(date):
        if isannual:
            pre_end = str(int(date[:4])-1)+'1231'
        else:
            if date[4:6] == '03':
                pre_end = str(int(date[:4])-1)+'1231'
            elif date[4:6] == '06':
                pre_end = date[:4]+'0331'
            elif date[4:6] == '09':
                pre_end = date[:4]+'0630'
            elif date[4:6] == '12':
                pre_end = date[:4]+'0930'
        return pre_end

    def get_nlatest_perticker(df, col_name):
        tmp_df = df.copy()
        tmp_df.sort_values(['pub_date', 'end_date'], inplace=True)

        tmp_df_1 = tmp_df.copy()
        tmp_df_1.rename(columns={col_name: 'value_0', 'pub_date': 'pub_date_0', 'end_date': 'end_date_0'},
                        inplace=True)

        for i in range(1, n):
            # 标记i期前的财报日期
            tmp_df_1['end_date_' + str(i)] = tmp_df_1['end_date_'+str(i-1)].apply(get_end_date)
            # 计算i期前财务数据
            tmp_df_2 = tmp_df.rename(columns={col_name: 'value_' + str(i), 'pub_date': 'pub_date_' + str(i),
                                              'end_date': 'end_date_' + str(i)})
            tmp_df_1 = tmp_df_1.merge(tmp_df_2, on=['ticker', 'end_date_' + str(i)], how='left')
        # 计算增长率
        if tmp_df_1.empty:
            return None
        else:            
        # 去重
        # 标记最大pub_date，为记录可用时间
            tmp_df_1['max_pub_date'] = np.max(tmp_df_1[pub_list].fillna(method='ffill', axis=1), axis=1)
            tmp_df_1['max_pub_date'] = tmp_df_1['max_pub_date'].astype(np.int64).astype(np.str)
            tmp_df_1.sort_values(['max_pub_date', 'end_date_0'], inplace=True)
            tmp_df_1 = tmp_df_1.drop_duplicates(subset=['max_pub_date'], keep='last')
            tmp_df_1['max_end_date'] = tmp_df_1['end_date_0'].rolling(window=8, min_periods=1).max()
            tmp_df_1['max_end_date'] = tmp_df_1['max_end_date'].astype(np.int64).astype(np.str)
            tmp_df_1 = tmp_df_1[tmp_df_1['end_date_0'] == tmp_df_1['max_end_date']]  # 得到最新财报数据
            return tmp_df_1[['ticker', 'max_pub_date', 'max_end_date']+value_list]
    fin_df = fin_df.groupby(['ticker']).apply(get_nlatest_perticker, col_name)
    fin_df.rename(columns={'max_pub_date': 'pub_date', 'max_end_date': 'end_date'}, inplace=True)
    fin_df.reset_index(inplace=True, drop=True)
    return fin_df

#得到资产负债表的数据
def get_bs_item(items, start_date,end_date):
    fields = ["secID","publishDate","endDate"] + items
    Rev_cogs_df = DataAPI.FdmtBSGet(ticker=u"",secID=u"",reportType=u"",endDate=end_date,beginDate=start_date,field=fields,pandas="1")
    Rev_cogs_df = Rev_cogs_df[((Rev_cogs_df['secID'].str.startswith('0'))|(Rev_cogs_df['secID'].str.startswith('3'))|(Rev_cogs_df['secID'].str.startswith('6')))]
    Rev_cogs_df = Rev_cogs_df.rename(columns={'publishDate':'pub_date','secID':'ticker','endDate':'end_date'})
    Rev_cogs_df['pub_date'] = pd.to_datetime(Rev_cogs_df['pub_date']).dt.strftime('%Y%m%d')
    Rev_cogs_df['end_date'] = pd.to_datetime(Rev_cogs_df['end_date']).dt.strftime('%Y%m%d')
    return Rev_cogs_df

#读取利润表相关数据
def get_is_item(is_factors, start_date,end_date):
    fields = ["secID","publishDate","endDate"] + is_factors
    is_df = DataAPI.FdmtISQPITGet(ticker=u"",secID=u"",reportType=u"",endDate=end_date,beginDate=start_date,field=fields,pandas="1")
    is_df = is_df[((is_df['secID'].str.startswith('0'))|(is_df['secID'].str.startswith('3'))|(is_df['secID'].str.startswith('6')))]
    is_df = is_df.rename(columns={'publishDate':'pub_date','secID':'ticker','endDate':'end_date'})
    is_df['pub_date'] = pd.to_datetime(is_df['pub_date']).dt.strftime('%Y%m%d')
    is_df['end_date'] = pd.to_datetime(is_df['end_date']).dt.strftime('%Y%m%d')
    return is_df 

#对齐连续几个季度的指标数据
def aligin_q_data(Rev_cogs_df,col_name, n_quarters = 13,isannual=False):
    Rev_cogs_df = Rev_cogs_df.copy()    
    Rev_df = get_findata_nlatest(Rev_cogs_df.copy(), n_quarters, col_name=col_name, isannual=isannual)
    Rev_q_df = Rev_df.drop_duplicates()
    return Rev_q_df

#获取最新财务数据
def get_fin_data_latest(fin_data_frame, col_name=['value']):
    """
    获取最新财务数据
    :param fin_data_frame: financial column= ['ticker','pub_date',’end_date',[fin_value]], index=num, pub_date='%Y%m%d'
    :param col_name: list, column name of value, 可以有多个列
    :return: column= ['ticker','pub_date','end_date',[fin_value]], 'ticker','pub_date' 是唯一性约束
    """

    fin_df = fin_data_frame.copy()
    def get_latest_perticker(df, col_name):
        tmp_df = df.copy()
        
        tmp_df.dropna(subset=col_name, how='all', inplace=True)
        tmp_df.sort_values(['pub_date', 'end_date'], inplace=True)
        tmp_df.drop_duplicates(subset=['pub_date'], keep='last', inplace=True)
        tmp_df['max_end_date'] = tmp_df['end_date'].rolling(window=6, min_periods=1).max()
        tmp_df['max_end_date'] = tmp_df['max_end_date'].astype(np.int64).astype(np.str)
        tmp_df = tmp_df[tmp_df['end_date'] == tmp_df['max_end_date']]
        return tmp_df[['ticker', 'pub_date', 'end_date'] + col_name]

    fin_df = fin_df.groupby(['ticker']).apply(get_latest_perticker, col_name)
    fin_df.reset_index(inplace=True, drop=True)
    return fin_df

#对离散的信号值进行填充
def fillsignal(df, min_date=False,maxd='20200525'):
    df = df.copy()
    df = df.sort_values('pub_date')
    if min_date == False:
        mind = df['pub_date'].values[0]
    else:
        mind = min(min_date, df['pub_date'].values[0])
    fulld = pd.date_range(mind, maxd).astype(str)
    fulld = [i.replace("-","") for i in fulld]
    df = df.set_index('pub_date').reindex(fulld)
    df = df.fillna(method='pad')
    return df

#得到营业成本和营业收入，用于计算毛利率
t0 = time.time()
is_df = get_is_item(["revenue","COGS"],start_date, end_date)
is_df['gm'] = (is_df['revenue'] -is_df['COGS'] ) / is_df['revenue']
is_df = is_df[['ticker','pub_date','end_date','gm']]

#得到连续5个季度的毛利率数据，用于计算毛利率的同比
is_df1 = get_fin_data_latest(is_df.copy(),col_name=['gm'])
gm_df = aligin_q_data(is_df1.copy(),'gm', n_quarters = 5,isannual=False)
gm_df['gm_y'] = (gm_df['value_0'] - gm_df['value_4']) / abs(gm_df['value_4'])
gm_df = gm_df[['ticker','pub_date','gm_y']].dropna(subset=['gm_y'])

#对计算得到的因子值进行填充
gm_df['pub_date'] = gm_df['pub_date'].astype(str)
gm_df = gm_df.groupby('ticker', as_index=False).apply(lambda x: fillsignal(x,min_date=False,maxd=end_date.replace("-",""))).reset_index().drop(['level_0'], axis=1).rename(columns={'pub_date': 'date'})
gm_df['ticker'] = gm_df['ticker'].str.slice(0,6)
print ('该部分耗时: %s 秒！'%(time.time() - t0))


##得到总资产和总负债数据，用于计算资产负债率
t0 = time.time()
bs_factors = ['TAssets',"TLiab"] 
bs_df = get_bs_item(bs_factors, start_date,end_date)
bs_df = get_fin_data_latest(bs_df.copy(),col_name=['TAssets','TLiab'])
bs_df['dtar'] = bs_df['TLiab'] / bs_df['TAssets']
#得到连续5个季度的资产负债率数据，用于计算资产负债率的同比
dtar_df = aligin_q_data(bs_df.copy(),"dtar", n_quarters = 5,isannual=False)
dtar_df['dtar_y'] = (dtar_df['value_0'] - dtar_df['value_4']) / abs(dtar_df['value_4'])
dtar_df = dtar_df[['ticker','pub_date','dtar_y']].dropna(subset=['dtar_y'])
#对计算得到的因子值进行填充
dtar_df['pub_date'] = dtar_df['pub_date'].astype(str)
dtar_df = dtar_df.groupby('ticker', as_index=False).apply(lambda x: fillsignal(x,min_date=False,maxd=end_date.replace("-",""))).reset_index().drop(['level_0'], axis=1).rename(columns={'pub_date': 'date'})
dtar_df['ticker'] = dtar_df['ticker'].str.slice(0,6)
print ('该部分耗时: %s 秒！'%(time.time() - t0))

#对上述得到的几个因子值进行合并
factor_df['date'] = factor_df['tradeDate'].str.replace("-","")
del factor_df['tradeDate']
mer_factor_df = factor_df.merge(gm_df,on=['ticker','date'], how='outer').merge(dtar_df,on=['ticker','date'], how='outer')

'''

1.3 计算行业指数

   调试 运行
文档
 代码  策略  文档
1.3.1 通过API获取申万行业分类数据（已回填）

'''

#申万行业指数
def sw_stock_indus_info():
    sw_fields= ['secID','ticker','secShortName','oldTypeName','intoDate','outDate','isNew','industryName1']
    stock_indus_info = DataAPI.MdSwBackGet(secID=u"",ticker=u"",intoDate=u"",outDate=u"",field=sw_fields,pandas="1")
    stock_indus_info['outDate'] = stock_indus_info['outDate'].fillna("2030-01-01")
    return stock_indus_info
stock_indus_info = sw_stock_indus_info()


#得到个股行情数据
price_fields = ['secID','ticker','tradeDate','closePrice','marketValue','negMarketValue']
price_df_list =[]
i = 0
for dt in week_end_list:
    if i%50 ==0:
        print ("当前处理日期： %s-------------------"%dt)
    price_dfi=DataAPI.MktEqudAdjGet(secID=u"",ticker=u"",tradeDate=dt,beginDate="",endDate="",isOpen=u"1",field=price_fields,pandas="1")
    price_df_list.append(price_dfi)
    i = i + 1
    
    
#得到个股的周度收益
price_df = pd.concat(price_df_list)
price_df = price_df.sort_values(by=['ticker','tradeDate'])
price_df['ret'] = price_df.groupby('ticker')['closePrice'].pct_change()
price_df = price_df[((price_df['ticker'].str.startswith('00'))|(price_df['ticker'].str.startswith('300'))|(price_df['ticker'].str.startswith('60')))]
price_df = price_df[price_df['ticker'].str.len()==6]

#得到股票历史的行业分类
price_indus_df = price_df.merge(stock_indus_info, left_on=['secID','ticker'], right_on=['secID','ticker'], how='left')
price_indus_df['flag'] = (price_indus_df['tradeDate'] >= price_indus_df['intoDate']) &  (price_indus_df['tradeDate'] < price_indus_df['outDate'])
price_indus_df = price_indus_df[price_indus_df['flag']==True]
price_indus_df = price_indus_df[['ticker','tradeDate','ret','marketValue','negMarketValue','secShortName','industryName1']]
price_indus_df = price_indus_df.drop_duplicates(subset=['ticker','tradeDate'])

#计算行业指数,base_date需要大于price_indus_df['tradeDate']的最小值
def cal_indus_index(price_indus_df, weight_col='negMarketValue',base_date="2016-12-12"):
    price_indus_df = price_indus_df.copy()
    price_indus_df = price_indus_df.sort_values(by=['tradeDate','industryName1']).dropna()
    daily_indus_rtn_df = price_indus_df.groupby(['tradeDate','industryName1']).apply(lambda x:np.average(x['ret'], weights=x[weight_col]))
    daily_indus_rtn_df = daily_indus_rtn_df.reset_index().rename(columns={0:'daily_rtn'})
    daily_indus_rtn_df = daily_indus_rtn_df[daily_indus_rtn_df['tradeDate']>=base_date]
    daily_indus_rtn_df.loc[daily_indus_rtn_df['tradeDate']==base_date,'daily_rtn'] = 0
    daily_indus_rtn_df = daily_indus_rtn_df.sort_values(by=['industryName1','tradeDate'])
    daily_indus_rtn_df['cum_rtn'] = daily_indus_rtn_df.groupby(['industryName1']).apply(lambda x:(1+ x['daily_rtn']).cumprod()).values
    return daily_indus_rtn_df


#画行业指数
def plot_industy_index(daily_indus_rtn_df):
    daily_indus_rtn_df = daily_indus_rtn_df.copy()
    i=1
    fig = plt.figure(figsize=(25,60))
    for p in daily_indus_rtn_df['industryName1'].unique():    
        tmp_df = daily_indus_rtn_df[daily_indus_rtn_df['industryName1']==p]
        tmp_df = tmp_df.set_index('tradeDate').sort_index()
        tmp_df.index = pd.to_datetime(tmp_df.index, format='%Y-%m-%d')
        tmp_df.index = tmp_df.index.map(lambda x: x.date())
        ax = fig.add_subplot(len(daily_indus_rtn_df['industryName1'].unique()),4,i)
        tmp_df = tmp_df.dropna()
        ax.plot(tmp_df.index,tmp_df['cum_rtn'].values)
        ax.legend([p.decode("utf-8")],loc='upper left',fontsize=10,prop=font)
        i = i+1
        step = max(len(tmp_df) / 10, 1)
        baseline = range(int(len(tmp_df) / step) + 1)
        baseline = [int(x) * step for x in baseline]
        baseline = np.array([int(x) for x in baseline if x < len(tmp_df)])    
        label_list = np.array([i1.strftime("%Y-%m-%d") for i1 in tmp_df.index])
        width = 0.5
        plt.subplots_adjust(wspace=0.1, hspace=0.4)
        plt.tick_params(axis='both',which='major',labelsize=10)
    plt.show()
    return 

'''

1.3.3 计算行业指数并绘制行业指数走势图

'''

# 计算行业指数, 按流通市值加权
daily_indus_rtn_df = cal_indus_index(price_indus_df, 'negMarketValue', base_date="2010-01-15")
#画行业指数的走势图
plot_industy_index(daily_indus_rtn_df.copy())

'''

第二部分：行业因子计算
该部分耗时 20分钟
该部分内容为：

个股因子映射到行业得到行业因子的计算方法介绍
个股因子的处理：包括去极值、缺失值填充
行业因子生成
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
2.1 个股因子映射到行业简介

从股票因子映射成行业指标在逻辑上及构造上都非常容易理解：将行业的成分股股票因子通过一定加权方式求和即可得到该行业指标。为后文指代方便，通过这种方式构建的行业指标我们简称为SAMI（Stock Alpha Mapping Indicator）。
图片注释
下标 t 表示时期，i 表示个股；参数 sector 表示行业，alpha 表示个股因子；w表示权重.
alpha 是经过预处理的个股因子值,需要进行如下处理：
1)个股因子不能做行业中性化处理
在股票因子构造中，我们经常默认进行市值与行业的中性化处理，使得因子ICIR 更高，选股效果更为稳定。但如果我们需要最终映射出的行业指标能反映出轮动信息，同时保有足够大的区分度，那中性化处理只会适得其反;
2)异常值处理
异常值不仅在个股选股层面会对alpha效果产生影响，如果不适当处理，在映射到行业后可能造成的偏差更为严重。由于行业池远远小于股票池大小，如果某个行业的某一只成分股权重很高，同时该成分股的因子值在量级上明显异常，那么构造出的该行业指标大概率明显异于其它行业，从而影响到行业轮动效果。
3)缺失值填充根据因子特征选用不同方式
因子所表征的意义不同时，对缺失值的填充方式应相应调整。一般采用的填充方式有三种：行业均值（中位数）填充、前值填充、补零填充。如果因子的缺失值是因为无覆盖造成的，则适合补零填充，例如分析师预期变化，对于没有分析师覆盖的公司，其预期变化补零更为合理。其它的场景基本用行业均值（或中位数）填充即可。

   调试 运行
文档
 代码  策略  文档
2.2 个股因子的处理

1)采用3倍标准差的方式去极值
2)对缺失值进行填充：分析师因子填0，其他因子按照行业行业中位数填充
    
    
'''

# 信号去极值
def signal_winsorize_process(signal_df, factor_list, frequency, wins_method='mad', sigma=3, n_draw=5, mad=5, p_value=0.04, neut_method='OLS', exclude_factor_list=[]):
    '''
    signal_df: panel/横截面/时间序列数据, 列至少包括['ticker', 'date', factor_list]
    factor_list: 需要进行处理的因子列表
    wins: bool, 是否进行去极值处理
    stand: bool, 是否进行标准化处理
    neut: bool, 是否进行中性化
    mad: mad法去极值的倍数
    neut_method：['OLS', 'WLS', 'GLS'], 中性化方法
    exclude_factor_list: 不进行中性的因子
    返回：
         经过因子处理后的dframe
    '''

    def process_by_day(tdate, dframe_by_tdate):
        # 对每个因子进行中性化
        for col in factor_list:
            if len(dframe_by_tdate[col].dropna()) < 30:
                continue
            dframe_by_tdate[col] = signal_winsorize(dframe_by_tdate[col], method=wins_method, mad=mad, n_draw=n_draw, sigma=sigma, p_value=p_value)
            
        return dframe_by_tdate

    signal_df = signal_df.set_index('ticker')
    signal_df['date'] = signal_df['date'].apply(lambda x: x.replace('-', ''))
    # 利用协程进行计算
    jobs = [gevent.spawn(process_by_day, td, df) for td, df in signal_df.groupby(['date'])]
    gevent.joinall(jobs)
    new_frame_list = [result.value for result in jobs]
    dframe = pd.concat(new_frame_list, axis=0)
    dframe.reset_index(inplace=True)

    return dframe


def signal_winsorize(se_factor, method='mad', sigma=3, n_draw=5, mad=5, p_value=0.04):
    """
    对factor值进行去极值操作
    :param se_factor: pd.Series
    :param method: 去极值方法，可选的有3sigma、quantile
    :param p_value：去极值分位数设定，默认为0.01
    :return: pd.Series
    """

    factor = se_factor.copy()
    if method == '3sigma':
        for i in range(n_draw):
            upper = factor.mean() + sigma * factor.std(ddof=0)
            lower = factor.mean() - sigma * factor.std(ddof=0)

            factor[factor > upper] = upper
            factor[factor < lower] = lower
    elif method == 'quantile':
        upper = factor.quantile(1 - p_value / 2)
        lower = factor.quantile(p_value / 2)

        factor[factor > upper] = upper
        factor[factor < lower] = lower
    elif method == 'mad':
        median = factor.median()
        median_double = 1.4826 * (factor - median).abs().median()
        upper = median + mad * median_double
        lower = median - mad * median_double
        factor[factor > upper] = upper
        factor[factor < lower] = lower
    else:
        raise Exception('method should be 3sigma、 quantile or mad !!!')
    return factor

#填充行业中值
def fillna_indu_median(dframe, col_list, indu_name='industryName1'):
    '''
    dframe: panel/横截面/时间序列数据, 至少包含 ['ticker', 'tradeDate', col_list], tradeDate为"%Y%m%d"
    col_list: 需要进行中性化的因子列表
    返回：
        经过空值填充的dframe
    '''
    # 中位数填充空值
    
    def fill_na_media(df, col):
        df[col] = df[col].fillna(df[col].median())
        return df
    dframe = dframe.groupby(['tradeDate', indu_name]).apply(fill_na_media, col_list)
    return dframe

#去极值
t0 = time.time()
used_factors = mer_factor_df.columns.difference(['ticker','date','MktValue','NegMktValue'])
factor_df= mer_factor_df.sort_values(['ticker', 'date']).dropna(subset=["ticker","date"])
factor_df = factor_df[(factor_df!=np.inf)&(factor_df!=-np.inf)]
win_factor_df = signal_winsorize_process(factor_df.copy(),  list(used_factors),freq, wins_method='3sigma')

#填充空值
win_factor_df['tradeDate'] = pd.to_datetime(win_factor_df['date'] ,format="%Y%m%d").dt.strftime("%Y-%m-%d")
del win_factor_df['date']
week_price_indus_df = price_indus_df[price_indus_df['tradeDate'].isin(week_end_list)]

#把行业的股票填满
win_factor_df = win_factor_df.merge(week_price_indus_df[['ticker','tradeDate','industryName1','marketValue','negMarketValue']].copy(),on=['ticker','tradeDate'], how='right')
win_factor_df = win_factor_df.drop_duplicates(subset=['ticker','tradeDate'])

# 对空值填充,分析师因子填0
analyst_factors = ['GREV']
win_factor_df.loc[:,analyst_factors] = win_factor_df.loc[:,analyst_factors].fillna(0)

#其他因子填行业中值
fill_median_factors = used_factors.difference(analyst_factors)
fill_median_factors = list(fill_median_factors)
win_factor_df = fillna_indu_median(win_factor_df, fill_median_factors, indu_name='industryName1')
win_factor_df= win_factor_df[['ticker','tradeDate'] + list(used_factors)]
print ('该部分耗时: %s 秒！'%(time.time() - t0))

'''

2.3 生成行业因子

采用市值开方加权的方式生成行业因子

'''

#计算行业因子
def cal_indus_factor(stock_indus_df, factor_df,indus_col='industryName1', factor_cols=[], weight_col=None):
    stock_indus_df = stock_indus_df.copy()
    factor_df = factor_df.copy()
    df = stock_indus_df.merge(factor_df, on=['ticker','tradeDate'], how='left')
    indus_factor_list = []
    for c in factor_cols:
        if weight_col is None:
            dfi = df.groupby(['tradeDate',indus_col]).apply(lambda x: x[c].mean())
        else:
            dfi = df.groupby(['tradeDate',indus_col]).apply(lambda x: np.average(x[c], weights=x[weight_col]))
        dfi.name = c    
        indus_factor_list.append(dfi)
    indus_factor_df= pd.DataFrame(indus_factor_list).T
    return indus_factor_df


used_factors = win_factor_df.columns.difference(['ticker','tradeDate'])
week_price_indus_df['sqrt_cap'] = np.sqrt(week_price_indus_df['marketValue'])
indus_factor_df = cal_indus_factor(week_price_indus_df.copy(),win_factor_df.copy(),indus_col='industryName1', factor_cols=list(used_factors)+['marketValue','negMarketValue'], weight_col="sqrt_cap").reset_index()
print (indus_factor_df.head().to_html())

'''

第三部分：行业因子测试
该部分耗时 3分钟
该部分内容为：

计算行业收益率、指数收益率
对单因子进行测试，测试内容包括计算IC、分组、多空收益、多头相对指数收益等
对合成因子进行测试，，测试内容包括计算IC、分组、多空收益、多头相对指数收益等
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3.1 计算行业收益率、指数收益率

采用周度测试的方法，所以需要计算行业指数的周度收益率和指数的周度收益率


'''

#行业周度收益
week_indus_rtn_df = daily_indus_rtn_df[daily_indus_rtn_df['tradeDate'].isin(week_end_list)]
week_indus_rtn_df = week_indus_rtn_df.sort_values(by=['industryName1','tradeDate'])
week_indus_rtn_df['ret'] = week_indus_rtn_df.groupby('industryName1')['cum_rtn'].pct_change()
week_indus_rtn_df = week_indus_rtn_df.sort_values(by=['industryName1','tradeDate'])
week_indus_rtn_df['nxt_ret'] = week_indus_rtn_df.groupby('industryName1')['ret'].shift(-1)
week_indus_rtn_df =week_indus_rtn_df.dropna(subset=['nxt_ret'])

#得到宽基指数收益率
def get_mktidx_rtn(idx_name,start_date,end_date):
    MktIdx = DataAPI.MktIdxdGet(indexID=u"",ticker=idx_name,tradeDate=u"",beginDate=start_date,endDate=end_date,exchangeCD=u"XSHG,XSHE",field=u"ticker,closeIndex,secShortName,tradeDate",pandas="1")
    MktIdx = MktIdx.sort_values(by=['ticker','tradeDate'])
    MktIdx['ret'] = MktIdx.groupby('ticker')['closeIndex'].pct_change()
    return MktIdx

#指数周收益率,此处选通联全A指数
idx_df= get_mktidx_rtn("DY0001",start_date,end_date)
week_idx_df = idx_df[idx_df['tradeDate'].isin(week_end_list)]
week_idx_df = week_idx_df.sort_values(by=['tradeDate'])
week_idx_df['ret'] = week_idx_df['closeIndex'].pct_change()
week_idx_df = week_idx_df.sort_values(by=['tradeDate'])
week_idx_df['nxt_ret'] = week_idx_df['ret'].shift(-1)
week_idx_df =week_idx_df.dropna(subset=['nxt_ret'])

#计算IC
def ic_anlyst(factor_df, rtn_df, factor_col, nxt_rtn_col,cor_method='spearman'):
    factor_rtn_df = factor_df.merge(rtn_df, on=['ticker', 'tradeDate'])
    period_ic = factor_rtn_df.groupby('tradeDate').apply(lambda x: x[[factor_col,nxt_rtn_col]].corr(method=cor_method).values[0, 1])
    # print period_ic
    ic = period_ic.mean()
    std = period_ic.std()
    icir = ic / std
    ic_t = stats.ttest_1samp(period_ic.dropna(), 0)[0]
    ic_summary = pd.Series([ic, std, icir, ic_t], index = [u'IC均值', u'IC波动率',u'ICIR', u't值']) 
    return ic_summary

#计算超额收益
def excess_rtn(s):
    r = s.iloc[-1]/s.iloc[0] - 1 
    return r

#计算胜率
def winper(s):
    s = s[s!=0]
    return (s>0).sum() / float(len(s))

#计算最大回测
def maxDrawdown(s):
    cum_max = s.cummax()
    maxdrawdown =((cum_max-s)/cum_max).max()
    return maxdrawdown

#计算年化收益
def annual_rtn(s, l,step=250):
    r = s.iloc[-1]/s.iloc[0] - 1
    ar = r / l * step
    return ar

#计算信息比率
def cal_ir(s,step=250):
    m = s.mean()
    m1 = m*step
    std1 = s.std()* np.sqrt(step)
    ir = m1/std1
    return ir

def ls_perf_stats(perf,step=12):
    r=[]
    excess_rtn1 = excess_rtn(perf['cum_ret'].dropna())
    winper1 = winper(perf['period_ret'].dropna())
    maxDrawdown1 = maxDrawdown(perf['cum_ret'].dropna())
    ir1 = cal_ir(perf['period_ret'],step)
    ar1 = annual_rtn(perf['cum_ret'].dropna(),len(perf['cum_ret'].dropna()),step)
    gb_p = pd.Series([excess_rtn1,ar1,winper1,maxDrawdown1,ir1],index=['total_rtn','annual_rtn','winper','maxDrawdown','ir']) 
    return gb_p


#分组绘图
def plot_group_fig(perf,group_num,title=u"分组净值"):
    fig = plt.figure(figsize=(18,8))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    nav = []
    label_dict = {}
    for i in range(group_num):
        label_dict[i+1] = u'第%s组'%(i+1)
        if i == 0:
            label_dict[i+1] += '(low)'
        elif i == group_num-1:
            label_dict[i+1] += '(high)'
        gperf = perf[perf['group'] == i]
        nav = nav + [gperf['cum_ret'].values[-1]]
        _=ax1.plot(pd.to_datetime(gperf['tradeDate']), gperf[['cum_ret']], label=label_dict[i+1])
    ax1.set_title(title, fontproperties=font, fontsize=12)
    ax1.legend(loc=0, prop=font)
    ind = np.arange(group_num)
    ax2.bar(ind+1.0/group_num, nav, 0.3, color='r')
    ax2.set_xlim((0, ind[-1]+1))
    ax2.set_xticks(ind+0.35)
    ax2.set_title(title, fontproperties=font, fontsize=12)
    _=ax2.set_xticklabels([label_dict[i+1] for i in ind], fontproperties=font)
    return 

#多空净值曲线
def plot_ls_fig(perf1,title=u"net value"):    
    f= plt.figure(figsize=(18,4))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    ax1 = f.add_subplot(111)
    _ = perf1.plot(ax=ax1)
    _ = ax1.set_title(perf1.name)
    return     
#信号测试
def test_signal(indus_factor_df, rtn_df, fc,ic_test=True,group_test=True,ls_test=True, direction=1, annual_step=52,group_num=5,plot=True):
    if ic_test:
        ic_summary = ic_anlyst(indus_factor_df.copy(), rtn_df.copy(), fc, "nxt_ret",cor_method='spearman')
    else:
        ic_summary= []
        
    if ls_test:
        ls_perf,ls_bt_df = quant_util.long_short_backtest(indus_factor_df.copy(), rtn_df.copy(), fc, 'nxt_ret', direction)
        ls_summary = ls_perf_stats(ls_perf,annual_step)
        if plot:
            ls_perf = ls_perf.set_index('tradeDate')['cum_ret']
            ls_perf.name = "long-short net_value of %s" %fc
            plot_ls_fig(ls_perf, fc)
    else:
        ls_summary= []        
    if group_test:
        group_perf,group_bt_df = quant_util.simple_group_backtest(indus_factor_df.copy(), rtn_df.copy(), fc, 'nxt_ret', ngrp=group_num)
        if plot:
            plot_group_fig(group_perf, group_num,title="group net-value: %s"%fc)
    else:
        group_perf,group_bt_df = [], []
    return  ic_summary, ls_summary, group_perf,group_bt_df

#行业选择统计
def industry_select_stats(df, fc, group_num, show=True,flag=''):
    df = df.copy()
    group_stats = df[df['group'].isin([0,group_num -1])]
    sel_num_df = group_stats.groupby(['group','ticker'])['weight'].count().reset_index().rename(columns={'weight':'count','ticker':'industry'})
    if show:        
        f, ax= plt.subplots(nrows=1, ncols=2, figsize = (18, 4))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        g1 = sel_num_df[sel_num_df['group']==group_num-1].set_index('industry').sort_values(by=['count'],ascending=False)
        _ = g1["count"].plot(ax=ax[0],kind="bar")
        _ = ax[0].set_title(u"多头持仓次数:%s%s"%(fc,flag), fontsize=12,fontproperties=font)
        ax[0].set_xticklabels( ax[0].get_xticklabels(),fontproperties=font )
        g2 = sel_num_df[sel_num_df['group']==0].set_index('industry').sort_values(by=['count'],ascending=False)
        _ = g2["count"].plot(ax=ax[1],kind="bar")
        _ = ax[1].set_title(u"空头持仓次数:%s%s"%(fc,flag), fontsize=12,fontproperties=font)
        ax[1].set_xticklabels( ax[1].get_xticklabels(),fontproperties=font )
        plt.show()
    return g1, g2 

#多头相对基准的超额收益
def cal_long_perf(group_perf,indus_rtn_df,fc, group_num,annual_step):
    indus_weight_rtn  = indus_rtn_df.groupby('tradeDate')['nxt_ret'].mean().reset_index().rename(columns={'nxt_ret':'indus_period_ret'})
    factor_long_rtn_df = group_perf[group_perf['group']==group_num-1][['tradeDate','period_ret']].rename(columns={'period_ret':'long_position_period_ret'})
    indus_long_rtn_df = factor_long_rtn_df.merge(indus_weight_rtn, on='tradeDate')
    indus_long_rtn_df['period_ret'] = indus_long_rtn_df['long_position_period_ret'] - indus_long_rtn_df['indus_period_ret']
    indus_long_rtn_df['cum_ret'] = indus_long_rtn_df['period_ret'].fillna(0)
    indus_long_rtn_df['cum_ret'] = (1+indus_long_rtn_df['cum_ret']).cumprod()
    long_perf = ls_perf_stats(indus_long_rtn_df,annual_step)
    long_perf.name = fc
    return long_perf   

#画行业选择的分布图
def plot_indus_select_distribition(df, fc, point=15,group_num=5, flag='insample'):
    df = df.copy()
    df = df[['tradeDate','ticker','group']]
    df['long_flag'] = df['group'].apply(lambda x: 1 if x==group_num-1 else 0)
    df['short_flag'] = df['group'].apply(lambda x: 1 if x==0 else 0)
    long_weight_df = df.pivot_table(index='ticker',columns='tradeDate', values='long_flag')
    short_weight_df = df.pivot_table(index='ticker',columns='tradeDate', values='short_flag')
    try:
        long_weight_df.index = long_weight_df.index.map(lambda x: x.decode("utf-8"))
    except:
        pass
    try:
        short_weight_df.index = short_weight_df.index.map(lambda x: x.decode("utf-8"))
    except:
        pass
    f, ax= plt.subplots(nrows=2, ncols=1, figsize = (19, 17))
    _ = seaborn.heatmap(long_weight_df, alpha=1.0, annot=False, center=0.0, annot_kws={"size": 12},
                        linecolor='white', linewidth=0.02, ax=ax[0], cmap='RdYlGn_r',cbar=False)
                        
    ax[0].set_yticklabels(ax[0].get_yticklabels(), fontproperties=font)
    step = max(len(long_weight_df.columns) / point, 1)
    baseline = range(int(len(long_weight_df.columns) / step) + 1)
    baseline = [int(x) * step for x in baseline]
    baseline = np.array([x for x in baseline if x < len(long_weight_df.columns)])
    label_list = long_weight_df.columns
    ax[0].set_xticks(baseline + 0.5)
    ax[0].set_xticklabels(label_list[baseline], rotation=45)
    ax[0].set_title(u'多头行业选择分布%s：%s'%(flag,fc), fontproperties=font, fontsize=14)
    
    _ = seaborn.heatmap(short_weight_df, alpha=1.0, annot=False, center=0.0, annot_kws={"size": 12},
                        linecolor='white', linewidth=0.02, ax=ax[1], cmap='RdYlGn_r',cbar=False)
                        
    ax[1].set_yticklabels(ax[1].get_yticklabels(), fontproperties=font)
    step = max(len(short_weight_df.columns) / point, 1)
    baseline = range(int(len(short_weight_df.columns) / step) + 1)
    baseline = [int(x) * step for x in baseline]
    baseline = np.array([x for x in baseline if x < len(short_weight_df.columns)])
    label_list = short_weight_df.columns
    ax[1].set_xticks(baseline + 0.5)
    ax[1].set_xticklabels(label_list[baseline], rotation=45)
    ax[1].set_title(u'空头行业选择分布%s：%s'%(flag,fc), fontproperties=font, fontsize=14)
    return

'''

3.2 单因子测试

采用周度测试的方法，测试时间为2012年1月至2020年5月，测试内容包括IC、分组、多空收益、多头相对指数收益等

'''

group_num = 5
annual_step=52
indus_factor_df = indus_factor_df.rename(columns={'industryName1':'ticker'})
week_indus_rtn_df = week_indus_rtn_df.rename(columns={'industryName1':'ticker'})
ic_summary_list  = []
ls_summary_df_list  = []
long_perf_list = []
test_start_date = '2012-01-01'
indus_factor_df = indus_factor_df[indus_factor_df['tradeDate']>=test_start_date]
for fc in used_factors:
    ic_summary,ls_summary_df,group_perf,group_bt_df= test_signal(indus_factor_df, week_indus_rtn_df, fc,ic_test=True,group_test=True,ls_test=True, direction=1,  annual_step=annual_step,group_num=group_num,plot=True)
    ic_summary_list.append(ic_summary)
    ls_summary_df_list.append(ls_summary_df)
    long_indus,short_indus = industry_select_stats(group_bt_df,fc, group_num, True,"")
    _ = plot_indus_select_distribition(group_bt_df, fc, point=15,group_num=5, flag = "")
    long_perf = cal_long_perf(group_perf,week_idx_df,fc, group_num,annual_step)
    long_perf_list.append(long_perf)
    
ic_df = pd.DataFrame(ic_summary_list,index=used_factors).applymap(lambda x:round(x,3))     
print  ('ic 统计：', ic_df.to_html())
ls_df = pd.DataFrame(ls_summary_df_list,index=used_factors).applymap(lambda x:round(x,3))
print  ('long-short 统计：', ls_df.to_html())
long_perf_df= pd.DataFrame(long_perf_list,index=used_factors).applymap(lambda x:round(x,3))
print  ('多头相对指数统计：',  long_perf_df.to_html())

'''

1)从5个因子的表现情况来看，价格类因子表现最好，FiftyTwoWeekHigh和REVS250周度IC分别达到6.1%和5.7%,多空收益为34.8%和21.1%；5个因子的IC的T值均在2以上，行业多头均明显跑赢全A指数；
2)从各分组的单调性来看，FiftyTwoWeekHigh因子值最大的分组表现并不是最佳的，但第一组和第五组区分度较好，总体分组区分度尚可；
3)从行业的选择来看，FiftyTwoWeekHigh因子在食品饮料上选择最多，钢铁、采掘上选择较少，从行业的走势上也可以看出，食品饮料板块近年来连续创出新高，而钢铁、采掘等周期行业持续表现低迷；dtar_y因子选择最多的行业大多属于TMT行业，行业内公司多处于成长阶段，有举债扩张的内在动力，从行业选择的时间序列来看，因子很好的捕捉到了18、19年通信行业的优异表现；gm_y因子能体现出行业毛利率的变化，从行业选择的时间序列来看，因子也捕捉到了19年至今由于非洲猪瘟影响，整个农林牧渔板块的高景气度。

   调试 运行
文档
 代码  策略  文档
3.3 合成因子测试

'''

#因子相关性画图
def _plot_corr_heatmap(df, factor_name1, factor_name2):
    fig, ax = plt.subplots(figsize=((12, 6)))
    corr_df = df.groupby('FACTOR').apply(lambda x: x[factor_name1].mean()).reindex(factor_name2)
    corr_df = (corr_df).round(2)
    _ = seaborn.heatmap(corr_df, alpha=1.0, annot=True, center=0.0, annot_kws={"size": 12},
                        linecolor='white', linewidth=0.02, ax=ax, cmap='RdYlGn_r')
    plt.show()
    
#计算因子相关性
def cal_corr(df, factors, style_names,corr_method ):
    trade_date = df.iloc[0]['tradeDate']
    style_industry_corr_df = pd.concat([df[style_names], df[factors]], axis=1, keys=['df1', 'df2']).corr(method=corr_method).loc['df1']['df2']
    style_industry_corr_df.index.name = 'FACTOR'
    style_industry_corr_df.reset_index(inplace=True)
    style_industry_corr_df['TRADE_DATE'] = trade_date
    return style_industry_corr_df

#标准化
def zscore_factor(dframe, col_list):
    # 对df的col_list每一列进行zscore标准化
    def zscore_frame(df, col_list):
        df[col_list] = (df[col_list] - df[col_list].mean()) / df[col_list].std()
        return df
    dframe = dframe.groupby(['tradeDate']).apply(zscore_frame, col_list)
    return dframe

'''

3.3.1 测试因子的相关性

'''

sel_factors = ['GREV','FiftyTwoWeekHigh','REVS250','gm_y', 'dtar_y']
factor_corr_df= indus_factor_df[sel_factors+['ticker','tradeDate']].groupby('tradeDate').apply(lambda x: cal_corr(x, sel_factors, sel_factors,"spearman"))
_plot_corr_heatmap(factor_corr_df, sel_factors, sel_factors)


'''

从因子相关性来看，只有’FiftyTwoWeekHigh’和’REVS250’的相关性超过0.7，其他因子之间相关性均比较低，所以合成因子时，价格类因子只保留’FiftyTwoWeekHigh’。

   调试 运行
文档
 代码  策略  文档
3.3.2 对以上5个因子做等权合成，测试合成因子的表现

'''

#选取单因子表现好的合成theme
sel_factors.remove('REVS250')
sel_indus_factor_df = indus_factor_df[sel_factors+['ticker','tradeDate']]
std_sel_indus_factor_df = zscore_factor(sel_indus_factor_df, sel_factors)
std_sel_indus_factor_df["com_factor"] = std_sel_indus_factor_df[sel_factors].mean(axis=1)
std_sel_indus_factor_df[:3]

group_num = 5
annual_step=52
bt_dic = {}
insample_ic_summary_list  = []
outsample_ic_summary_list  = []
insample_ls_summary_df_list  = []
outsample_ls_summary_df_list  = []
insample_long_perf_list = []
outsample_long_perf_list = []
insample_long_perf_list1 = []
outsample_long_perf_list1 = []
fc = 'com_factor'

print ("*"*20 + fc + "*"*20)
ic_summary, ls_summary, group_perf,group_bt_df = test_signal(std_sel_indus_factor_df, week_indus_rtn_df, fc,ic_test=True,group_test=True,ls_test=True, direction=1,  annual_step=annual_step,group_num=group_num,plot=True)
long_indus,short_indus = industry_select_stats(group_bt_df,fc, group_num, True,"")
_ = plot_indus_select_distribition(group_bt_df, fc, point=15,group_num=5, flag = "")
long_perf = cal_long_perf(group_perf,week_idx_df,fc, group_num,annual_step)

ic_df = pd.DataFrame([ic_summary],index=[fc]).applymap(lambda x:round(x,3))     
print  ('ic 统计：', ic_df.to_html())

ls_df = pd.DataFrame([ls_summary],index=[fc]).applymap(lambda x:round(x,3))
print  ('long-short 统计 ：', ls_df.to_html())

long_perf_df= pd.DataFrame([long_perf],index=[fc]).applymap(lambda x:round(x,3))
print  ('多头相对指数统计：',  long_perf_df.to_html())

'''

合成的因子无论是IC、分组、多空、多头相对行业指数收益等都相比于单因子有明显的提升；行业多头选择最多的为食品饮料、计算机、传媒，空头选择最多的为采掘、有色金属、钢铁，从行业选择的时间序列也能看出，模型多头多次选择到了景气度向上变化的行业，如16年至今的食品饮料行业，19年至今的农林牧渔行业、14年下半年至15年上半年的非银金融行业、16年中至18年的钢铁行业等。

   调试 运行
文档
 代码  策略  文档
第四部分：结论
   调试 运行
文档
 代码  策略  文档
本文基于优矿提供的因子数据构建个股因子并探索将个股因子映射到行业构建行业轮动模型的方法，结果表明挑选出的5个因子构建的行业因子表现均比较好，IC的T值均在2以上，行业多空收益均大于8%，且多头均明显跑赢全A指数；合成的因子无论是IC、分组、多空、多头相对行业指数收益等都相比于单因子有明显的提升，因子周度IC为6.8%，多空收益43.3%，多头相对于指数的超额收益为44%；合成因子行业多头选择最多的为食品饮料、计算机、传媒，空头选择最多的为采掘、有色金属、钢铁，从行业选择的时间序列也能看出，模型多头多次选择到了景气度向上变化的行业。

'''

