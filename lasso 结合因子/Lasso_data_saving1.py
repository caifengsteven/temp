
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()


# In[ ]:

# 整合数据集
# 按月整理数据，因子按月更新

indus_map_dict = {}
stock_comp_dict = {}
for year in range(2010, 2020):
    begin_date = '%s-01-01'%(year)
    end_date = '%s-02-01'%(year + 1)
    
    # 获取股票信息 A股数据
    tickers_A = DataAPI.IdxConsGet(secID=u"",ticker=u"000002",isNew=u"",intoDate= begin_date,
                                 field=u"consTickerSymbol",pandas="1")['consTickerSymbol'].tolist()
    
    # 获取月份数据
    df_info = DataAPI.MktEqumGet(secID=u"",ticker=tickers_A, monthEndDate=u"",beginDate=begin_date, endDate=end_date,
                                 isOpen=u"",field=u"",pandas="1")
    df_info['month'] = df_info['endDate'].apply(lambda x: x[:7])
    
    # 获取行业标识
    indu_field = ['ticker', 'industryID1']
    indu_label = DataAPI.EquIndustryGet(secID=u"",ticker=tickers_A,industryVersionCD=u"010303",industry=u"",industryID=u"",industryID1=u"",
                                        industryID2=u"",industryID3=u"",intoDate=end_date,equTypeID=u"",field=indu_field,pandas="1")
    
    
    indu_map_dict = {k:v for k,v in zip(indu_label['ticker'], indu_label['industryID1'])}
    indus_map_dict[year] = indu_map_dict
    
    # 获取下个月的收益信息 Y
    df_ret = df_info.groupby(['month', 'ticker'])['return'].mean().unstack()
    df_ret = df_ret.shift(-1).iloc[:-1]
    for mon in df_ret.index:
        stock_comp_dict[mon] = df_ret.loc[mon].to_dict()
    print(year ,'has done')


# In[ ]:

# 获取月末日期
all_date = DataAPI.MktIdxdGet(indexID=u"",ticker=u"000002",tradeDate=u"",beginDate=u"",
                              endDate=u"",exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")['tradeDate'].tolist()
all_date.sort()
def get_month_end_date_by_year(year = '2010'):
    
    for i in all_date:
        if year in i:
            break
    begin_year_ind = all_date.index(i)
    
    for i in all_date[begin_year_ind:]:
        if year not in i:
            break
    end_year_ind = all_date.index(i)
    
    year_date = all_date[begin_year_ind : end_year_ind]
    
    end_mon_dates = []
    
    for mon in range(1, 13):
        if mon < 10: 
            mon_str = '%s-0%s'%(year, mon)
        else:
            mon_str = '%s-%s'%(year, mon)
        
        for i in year_date:
            if mon_str in i:
                break
        begin_mon_ind = year_date.index(i)
        
        for i in year_date[begin_mon_ind: ]:
            if mon_str not in i:
                break
        end_mon_ind = year_date.index(i)
        
        end_mon_dates.append(year_date[end_mon_ind - 1])
    return end_mon_dates


# In[ ]:

mon_end_dates = []
for year in range(2010, 2020):mon_end_dates += get_month_end_date_by_year(str(year)) 


# In[ ]:


def indu_market_scale(fac_df, fac_cols):
    # 按行业对因子进行标准化
    indu_list = fac_df['indu_style'].unique().tolist()
    scale_dfs = []
    for indu in indu_list:
        sub_df = fac_df[fac_df['indu_style'] == indu].copy()
        sub_df.fillna(sub_df.mean(), inplace = True)
        # 如果还有nan值，用全部数据的均值填充
        if not np.all(-np.isnan(sub_df[fac_cols].values)):
            sub_df.fillna(fac_df.mean(), inplace = True)
        sub_scale_df = scale.fit_transform(sub_df[fac_cols])
        sub_df.loc[:, fac_cols] = sub_scale_df
        scale_dfs.append(sub_df)
    return pd.concat(scale_dfs)

def get_fac_df_by_month(mon):
    
    # 获取股票成分
    for mon_date in mon_end_dates:
        if mon in mon_date:break
    ticker_chs = indus_map_dict[int(mon[:4])].keys()

    # 获取信息
    fac_df = DataAPI.MktStockFactorsOneDayGet(tradeDate= mon_date ,secID=u"",ticker= ticker_chs,field=u"",pandas="1")
    mak_df = DataAPI.MktEqudGet(secID=u"",ticker=ticker_chs ,tradeDate=mon_date ,beginDate=u"",endDate=u"",isOpen="",field=u"",pandas="1")
    mak_df.index = mak_df.pop('ticker')
    market_value_map = mak_df['marketValue'].to_dict()
    fac_df['return'] = fac_df['ticker'].map(stock_comp_dict[mon])
    # 去掉无收益数据
    fac_df['return'].dropna(inplace = True)

    fac_df['market_value'] = fac_df['ticker'].map(market_value_map)
    fac_df['indu_style'] = fac_df['ticker'].map(indus_map_dict[int(mon[:4])])
    # fac_cols = fac_df.columns[3:-2]
    # fac_scale_df = indu_market_scale(fac_df, fac_cols)
    return fac_df


# In[ ]:

for year in range(2010, 2020):
    for mon in range(1, 13):
        if mon < 10:
            mon_str = '%s-0%s'%(year, mon)
        else:
            mon_str = '%s-%s'%(year, mon)

            
        fac_df = get_fac_df_by_month(mon_str)
        fac_cols = fac_df.columns[3:-2]
        fac_mon_df = indu_market_scale(fac_df, fac_cols) 
        fac_mon_df['real_return'] = fac_df['return']
        fac_mon_df.to_csv('./fac_data/%s.csv'%mon_str)
        print(mon_str, 'has done')


# In[ ]:



