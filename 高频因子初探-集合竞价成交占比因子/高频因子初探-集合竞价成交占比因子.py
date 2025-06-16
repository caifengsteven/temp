# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:04:47 2020

@author: Asus
"""

''''
近半年来，通过高频数据的累计效应为低频选股提供额外信息这种思路受到各大研究团队的青睐，优矿社区上也有一些基于高频数据因子的分析测试，比如call大神的聪明钱因子等等。

在A股这个远非有效但竞争激烈的投资环境，开辟一片少有人研究过的乐土确实有很大的吸引力，本文也希望在这一方面做一点微小的尝试，欢迎讨论。

一般认为股价的表现情况和股票受关注程度有一定的关系，而受关注的股票，往往在集合竞价期间多空博弈就比较激烈，而且在集合竞价期间，由于大盘整体施加在个股上的压力较少。那么基于这种单独由个股特性而带来的，强烈的多空博弈是否对个股未来的收益有影响呢？

我们今天介绍的***集合竞价成交占比因子***就是在这方面的一个尝试，即我们认为集合竞价成交比例较多的股票会有超额收益，具体公式如下：

** 集合竞价成交占比 **

$$\mathrm{CMV}n = \frac{1}{n}\Sigma{i=1}^n\frac{\mathrm{CVOL}_i}{\mathrm{VOL}_i} ~~~ $$
其中CVOL表示每一日的集合竞价成交量，VOL表示每一日的股票总成交量，n为对集合竞价成交量占比去移动平均的天数。

'''
import time
import datetime
import numpy as np
import pandas as pd
import scipy.stats as st

import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')
import seaborn as sns
sns.set_style('white')
from matplotlib import dates
import sys
sys.path
sys.path.append('G:\dropbox\Dropbox\Dropbox\project folder from my asua computer\Project\lib')
#from CAL.PyCAL import font, Date, DateTime    # CAL.PyCAL中包含的font可以用于展示中文
import quant_util as qutil

import time
import pandas as pd
import numpy as np

def getMarketValueAll(universe, begin, end, file_name=None):
    """
    获取股票历史每日市值
    universe: list, secID组成的list
    begin: datetime string, 起始日期，格式为"%Y%m%d"
    end: datetime string, 终止日期，格式为"%Y%m%d"
    file_name: string, 以".csv"结尾且符合文件命名规范的字符串
    """
    print  ('MarketValue will be calculated for ' + str(len(universe)) + ' stocks:')
    count = 0
    secs_time = 0
    start_time = time.time()
    N = 50
    ret_data = pd.DataFrame()
    for stk in universe:
        data = qutil.MktEqudAdjGet(secID=stk, beginDate=begin, endDate=end, field='secID,tradeDate,marketValue')    # 拿取数据
        tmp_ret_data = data.sort('tradeDate')
        # 市值部分
        tmp_ret_data = tmp_ret_data[['tradeDate','marketValue']]
        tmp_ret_data.columns = ['tradeDate', stk]
        if ret_data.empty:
            ret_data = tmp_ret_data
        else:
            ret_data = ret_data.merge(tmp_ret_data, on='tradeDate', how='outer')
        # 打印进度部分
        count += 1
        if count > 0 and count % N == 0:
            finish_time = time.time()
            print (count)
            print ('  ' + str(np.round((finish_time-start_time) - secs_time, 0)) + ' seconds elapsed.')
            secs_time = (finish_time-start_time)
    if file_name:
        ret_data.to_csv(file_name)
    return ret_data


def getCallAuctionRatioAll(universe, begin, end, file_name=None):
    """
    计算集合竞价占比因子
    
    universe: list, secID组成的list
    begin: datetime string, 起始日期，格式为"%Y%m%d"
    end: datetime string, 终止日期，格式为"%Y%m%d"
    file_name： string, 以".csv"结尾且符合文件命名规范的字符串
    """
    cal_dates = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=begin, endDate=end).sort('calendarDate')
    cal_dates = cal_dates[cal_dates['isOpen']==1]
    all_dates = cal_dates['calendarDate'].values.tolist()
    all_dates = [x.replace('-', '') for x in all_dates]
    
    print ('CallAuctionRatio will be calculated for ' + str(len(all_dates)) + ' days:')
    count = 0
    secs_time = 0
    start_time = time.time()
    
    data = pd.DataFrame()   
    for dt in all_dates:        
        call_auction_data = DataAPI.MktBarHistOneDayGet(securityID=universe, date=dt, startTime='09:30', endTime='09:30',
                                                        field='ticker,barTime,totalVolume,totalValue')
        call_auction_data = call_auction_data[call_auction_data['barTime']=='09:30']
        call_auction_data = call_auction_data[[u'ticker', u'totalVolume', u'totalValue']]
        call_auction_data.columns = [u'ticker', u'callAuctionVol', u'callAuctionValue']

        vol_data = DataAPI.MktEqudGet(secID=universe, tradeDate=dt, 
                                      field='secID,tradeDate,ticker,turnoverVol,turnoverValue')

        dt_data = call_auction_data.merge(vol_data, on='ticker')
        dt_data = dt_data[[u'tradeDate', u'secID', u'callAuctionVol', u'callAuctionValue', u'turnoverVol',u'turnoverValue']]
        
        data = data.append(dt_data)
        
        # 打印进度部分
        count += 1
        if count > 0 and count % 10 == 0:
            finish_time = time.time()
            print (count)
            print ('  ' + str(np.round((finish_time-start_time) - secs_time, 0)) + ' seconds elapsed.')
            secs_time = (finish_time-start_time)
    data.to_csv('my_auction_data.csv')        
    ret_data = data[data['turnoverValue']>0]
    ret_data['CallAuctionRatio'] = ret_data['callAuctionVol']*1.0/ret_data['turnoverVol'] * 100.0
    ret_data = ret_data[['tradeDate', 'secID', 'CallAuctionRatio']]
    #ret_data = ret_data.set_index(['tradeDate', 'secID'])['CallAuctionRatio'].unstack()
    ret_data=ret_data.dropna()
    print (ret_data)
    ret_data = ret_data.set_index(['tradeDate', 'secID'])
    #print ret_data
    ret_data = ret_data['CallAuctionRatio']
    #ret_data = ret_data.drop_duplicate()
    print (np.where(ret_data.index.duplicated()))
    #print ret_data
    ret_data = ret_data.unstack()
    
    
    if file_name:
        ret_data.to_csv(file_name)
            
    return ret_data


begin_date = '20140101'  # 开始日期
end_date = '20180130'    # 结束日期

temp_universe = ['000001.XSHE' ,'000002.XSHE' ,'000008.XSHE' ,'000009.XSHE' ,'000012.XSHE' ,'000021.XSHE' ,'000024.XSHE' ,'000027.XSHE' ,'000031.XSHE' ,'000039.XSHE' ,'000046.XSHE' ,'000059.XSHE' ,'000060.XSHE' ,'000061.XSHE' ,'000063.XSHE' ,'000069.XSHE' ,'000100.XSHE' ,'000156.XSHE' ,'000157.XSHE' ,'000166.XSHE' ,'000333.XSHE' ,'000338.XSHE' ,'000400.XSHE' ,'000401.XSHE' ,'000402.XSHE' ,'000413.XSHE' ,'000415.XSHE' ,'000422.XSHE' ,'000423.XSHE' ,'000425.XSHE' ,'000503.XSHE' ,'000527.XSHE' ,'000528.XSHE' ,'000536.XSHE' ,'000538.XSHE' ,'000539.XSHE' ,'000540.XSHE' ,'000555.XSHE' ,'000559.XSHE' ,'000562.XSHE' ,'000568.XSHE' ,'000581.XSHE' ,'000596.XSHE' ,'000598.XSHE' ,'000612.XSHE' ,'000623.XSHE' ,'000625.XSHE' ,'000627.XSHE' ,'000629.XSHE' ,'000630.XSHE' ,'000651.XSHE' ,'000656.XSHE' ,'000671.XSHE' ,'000680.XSHE' ,'000685.XSHE' ,'000686.XSHE' ,'000690.XSHE' ,'000703.XSHE' ,'000709.XSHE' ,'000712.XSHE' ,'000718.XSHE' ,'000723.XSHE' ,'000725.XSHE' ,'000728.XSHE' ,'000729.XSHE' ,'000738.XSHE' ,'000750.XSHE' ,'000758.XSHE' ,'000768.XSHE' ,'000776.XSHE' ,'000778.XSHE' ,'000780.XSHE' ,'000783.XSHE' ,'000792.XSHE' ,'000793.XSHE' ,'000800.XSHE' ,'000807.XSHE' ,'000825.XSHE' ,'000826.XSHE' ,'000831.XSHE' ,'000839.XSHE' ,'000858.XSHE' ,'000869.XSHE' ,'000876.XSHE' ,'000878.XSHE' ,'000883.XSHE' ,'000895.XSHE' ,'000898.XSHE' ,'000917.XSHE' ,'000927.XSHE' ,'000933.XSHE' ,'000937.XSHE' ,'000938.XSHE' ,'000951.XSHE' ,'000959.XSHE' ,'000960.XSHE' ,'000961.XSHE' ,'000963.XSHE' ,'000968.XSHE' ,'000969.XSHE' ,'000970.XSHE' ,'000977.XSHE' ,'000983.XSHE' ,'000999.XSHE' ,'001979.XSHE' ,'002001.XSHE' ,'002007.XSHE' ,'002008.XSHE' ,'002024.XSHE' ,'002027.XSHE' ,'002028.XSHE' ,'002038.XSHE' ,'002044.XSHE' ,'002049.XSHE' ,'002051.XSHE' ,'002065.XSHE' ,'002069.XSHE' ,'002073.XSHE' ,'002074.XSHE' ,'002081.XSHE' ,'002085.XSHE' ,'002092.XSHE' ,'002106.XSHE' ,'002122.XSHE' ,'002128.XSHE' ,'002129.XSHE' ,'002131.XSHE' ,'002142.XSHE' ,'002146.XSHE' ,'002152.XSHE' ,'002153.XSHE' ,'002155.XSHE' ,'002174.XSHE' ,'002183.XSHE' ,'002195.XSHE' ,'002202.XSHE' ,'002230.XSHE' ,'002236.XSHE' ,'002241.XSHE' ,'002244.XSHE' ,'002252.XSHE' ,'002269.XSHE' ,'002292.XSHE' ,'002294.XSHE' ,'002299.XSHE' ,'002304.XSHE' ,'002310.XSHE' ,'002344.XSHE' ,'002352.XSHE' ,'002353.XSHE' ,'002375.XSHE' ,'002378.XSHE' ,'002385.XSHE' ,'002399.XSHE' ,'002400.XSHE' ,'002405.XSHE' ,'002410.XSHE' ,'002411.XSHE' ,'002415.XSHE' ,'002416.XSHE' ,'002422.XSHE' ,'002424.XSHE' ,'002426.XSHE' ,'002429.XSHE' ,'002431.XSHE' ,'002450.XSHE' ,'002456.XSHE' ,'002460.XSHE' ,'002465.XSHE' ,'002466.XSHE' ,'002468.XSHE' ,'002470.XSHE' ,'002475.XSHE' ,'002493.XSHE' ,'002498.XSHE' ,'002500.XSHE' ,'002508.XSHE' ,'002555.XSHE' ,'002558.XSHE' ,'002568.XSHE' ,'002570.XSHE' ,'002572.XSHE' ,'002594.XSHE' ,'002601.XSHE' ,'002602.XSHE' ,'002603.XSHE' ,'002608.XSHE' ,'002624.XSHE' ,'002653.XSHE' ,'002673.XSHE' ,'002714.XSHE' ,'002736.XSHE' ,'002739.XSHE' ,'002797.XSHE' ,'002831.XSHE' ,'002839.XSHE' ,'002841.XSHE' ,'300002.XSHE' ,'300003.XSHE' ,'300015.XSHE' ,'300017.XSHE' ,'300024.XSHE' ,'300027.XSHE' ,'300033.XSHE' ,'300058.XSHE' ,'300059.XSHE' ,'300070.XSHE' ,'300072.XSHE' ,'300085.XSHE' ,'300104.XSHE' ,'300122.XSHE' ,'300124.XSHE' ,'300133.XSHE' ,'300136.XSHE' ,'300144.XSHE' ,'300146.XSHE' ,'300168.XSHE' ,'300182.XSHE' ,'300251.XSHE' ,'300315.XSHE' ,'600000.XSHG' ,'600004.XSHG' ,'600005.XSHG' ,'600008.XSHG' ,'600009.XSHG' ,'600010.XSHG' ,'600011.XSHG' ,'600015.XSHG' ,'600016.XSHG' ,'600018.XSHG' ,'600019.XSHG' ,'600021.XSHG' ,'600022.XSHG' ,'600023.XSHG' ,'600026.XSHG' ,'600027.XSHG' ,'600028.XSHG' ,'600029.XSHG' ,'600030.XSHG' ,'600031.XSHG' ,'600036.XSHG' ,'600037.XSHG' ,'600038.XSHG' ,'600048.XSHG' ,'600050.XSHG' ,'600058.XSHG' ,'600060.XSHG' ,'600061.XSHG' ,'600062.XSHG' ,'600066.XSHG' ,'600068.XSHG' ,'600074.XSHG' ,'600079.XSHG' ,'600085.XSHG' ,'600089.XSHG' ,'600096.XSHG' ,'600098.XSHG' ,'600100.XSHG' ,'600104.XSHG' ,'600108.XSHG' ,'600109.XSHG' ,'600111.XSHG' ,'600115.XSHG' ,'600118.XSHG' ,'600123.XSHG' ,'600125.XSHG' ,'600132.XSHG' ,'600143.XSHG' ,'600150.XSHG' ,'600151.XSHG' ,'600153.XSHG' ,'600157.XSHG' ,'600160.XSHG' ,'600161.XSHG' ,'600166.XSHG' ,'600169.XSHG' ,'600170.XSHG' ,'600177.XSHG' ,'600183.XSHG' ,'600188.XSHG' ,'600196.XSHG' ,'600208.XSHG' ,'600216.XSHG' ,'600219.XSHG' ,'600220.XSHG' ,'600221.XSHG' ,'600233.XSHG' ,'600239.XSHG' ,'600252.XSHG' ,'600256.XSHG' ,'600259.XSHG' ,'600266.XSHG' ,'600267.XSHG' ,'600269.XSHG' ,'600271.XSHG' ,'600276.XSHG' ,'600277.XSHG' ,'600297.XSHG' ,'600300.XSHG' ,'600307.XSHG' ,'600309.XSHG' ,'600312.XSHG' ,'600315.XSHG' ,'600316.XSHG' ,'600317.XSHG' ,'600320.XSHG' ,'600331.XSHG' ,'600332.XSHG' ,'600340.XSHG' ,'600348.XSHG' ,'600350.XSHG' ,'600352.XSHG' ,'600362.XSHG' ,'600369.XSHG' ,'600372.XSHG' ,'600373.XSHG' ,'600376.XSHG' ,'600380.XSHG' ,'600383.XSHG' ,'600390.XSHG' ,'600395.XSHG' ,'600398.XSHG' ,'600403.XSHG' ,'600406.XSHG' ,'600415.XSHG' ,'600418.XSHG' ,'600428.XSHG' ,'600432.XSHG' ,'600436.XSHG' ,'600446.XSHG' ,'600456.XSHG' ,'600481.XSHG' ,'600482.XSHG' ,'600485.XSHG' ,'600489.XSHG' ,'600497.XSHG' ,'600498.XSHG' ,'600500.XSHG' ,'600508.XSHG' ,'600516.XSHG' ,'600517.XSHG' ,'600518.XSHG' ,'600519.XSHG' ,'600522.XSHG' ,'600528.XSHG' ,'600535.XSHG' ,'600546.XSHG' ,'600547.XSHG' ,'600549.XSHG' ,'600550.XSHG' ,'600570.XSHG' ,'600578.XSHG' ,'600582.XSHG' ,'600583.XSHG' ,'600585.XSHG' ,'600588.XSHG' ,'600595.XSHG' ,'600597.XSHG' ,'600598.XSHG' ,'600600.XSHG' ,'600606.XSHG' ,'600633.XSHG' ,'600635.XSHG' ,'600637.XSHG' ,'600642.XSHG' ,'600648.XSHG' ,'600649.XSHG' ,'600654.XSHG' ,'600655.XSHG' ,'600660.XSHG' ,'600663.XSHG' ,'600664.XSHG' ,'600666.XSHG' ,'600674.XSHG' ,'600675.XSHG' ,'600682.XSHG' ,'600685.XSHG' ,'600688.XSHG' ,'600690.XSHG' ,'600694.XSHG' ,'600703.XSHG' ,'600704.XSHG' ,'600705.XSHG' ,'600717.XSHG' ,'600718.XSHG' ,'600737.XSHG' ,'600739.XSHG' ,'600741.XSHG' ,'600754.XSHG' ,'600770.XSHG' ,'600779.XSHG' ,'600783.XSHG' ,'600795.XSHG' ,'600804.XSHG' ,'600808.XSHG' ,'600809.XSHG' ,'600811.XSHG' ,'600812.XSHG' ,'600816.XSHG' ,'600820.XSHG' ,'600823.XSHG' ,'600827.XSHG' ,'600832.XSHG' ,'600837.XSHG' ,'600839.XSHG' ,'600859.XSHG' ,'600863.XSHG' ,'600867.XSHG' ,'600871.XSHG' ,'600873.XSHG' ,'600875.XSHG' ,'600879.XSHG' ,'600880.XSHG' ,'600881.XSHG' ,'600886.XSHG' ,'600887.XSHG' ,'600893.XSHG' ,'600895.XSHG' ,'600900.XSHG' ,'600909.XSHG' ,'600919.XSHG' ,'600926.XSHG' ,'600958.XSHG' ,'600959.XSHG' ,'600970.XSHG' ,'600971.XSHG' ,'600977.XSHG' ,'600997.XSHG' ,'600998.XSHG' ,'600999.XSHG' ,'601001.XSHG' ,'601006.XSHG' ,'601009.XSHG' ,'601012.XSHG' ,'601016.XSHG' ,'601018.XSHG' ,'601021.XSHG' ,'601088.XSHG' ,'601098.XSHG' ,'601099.XSHG' ,'601101.XSHG' ,'601106.XSHG' ,'601107.XSHG' ,'601111.XSHG' ,'601117.XSHG' ,'601118.XSHG' ,'601127.XSHG' ,'601139.XSHG' ,'601155.XSHG' ,'601158.XSHG' ,'601163.XSHG' ,'601166.XSHG' ,'601168.XSHG' ,'601169.XSHG' ,'601179.XSHG' ,'601186.XSHG' ,'601198.XSHG' ,'601211.XSHG' ,'601212.XSHG' ,'601216.XSHG' ,'601225.XSHG' ,'601228.XSHG' ,'601229.XSHG' ,'601231.XSHG' ,'601233.XSHG' ,'601238.XSHG' ,'601258.XSHG' ,'601268.XSHG' ,'601288.XSHG' ,'601299.XSHG' ,'601318.XSHG' ,'601328.XSHG' ,'601333.XSHG' ,'601336.XSHG' ,'601369.XSHG' ,'601375.XSHG' ,'601377.XSHG' ,'601390.XSHG' ,'601398.XSHG' ,'601519.XSHG' ,'601555.XSHG' ,'601558.XSHG' ,'601566.XSHG' ,'601600.XSHG' ,'601601.XSHG' ,'601607.XSHG' ,'601608.XSHG' ,'601611.XSHG' ,'601618.XSHG' ,'601628.XSHG' ,'601633.XSHG' ,'601666.XSHG' ,'601668.XSHG' ,'601669.XSHG' ,'601688.XSHG' ,'601699.XSHG' ,'601717.XSHG' ,'601718.XSHG' ,'601727.XSHG' ,'601766.XSHG' ,'601788.XSHG' ,'601800.XSHG' ,'601808.XSHG' ,'601818.XSHG' ,'601857.XSHG' ,'601866.XSHG' ,'601872.XSHG' ,'601877.XSHG' ,'601878.XSHG' ,'601881.XSHG' ,'601888.XSHG' ,'601898.XSHG' ,'601899.XSHG' ,'601901.XSHG' ,'601918.XSHG' ,'601919.XSHG' ,'601928.XSHG' ,'601929.XSHG' ,'601933.XSHG' ,'601939.XSHG' ,'601958.XSHG' ,'601966.XSHG' ,'601969.XSHG' ,'601985.XSHG' ,'601988.XSHG' ,'601989.XSHG' ,'601991.XSHG' ,'601992.XSHG' ,'601997.XSHG' ,'601998.XSHG' ,'603000.XSHG' ,'603160.XSHG' ,'603288.XSHG' ,'603699.XSHG' ,'603799.XSHG' ,'603833.XSHG' ,'603858.XSHG' ,'603885.XSHG' ,'603993.XSHG' ]
print (temp_universe)
universe = [x for x in temp_universe if x[0] in '036']   # 股票池，去除B股

start_time = time.time()
data = getMarketValueAll(universe, begin_date, end_date, file_name='MarketValues_FullA.csv')
finish_time = time.time()

print (str(finish_time-start_time) + ' seconds elapsed in total.')

begin_date = '20140101'  # 开始日期
end_date = '20180130'    # 结束日期
temp_universe = ['000001.XSHE' ,'000002.XSHE' ,'000008.XSHE' ,'000009.XSHE' ,'000012.XSHE' ,'000021.XSHE' ,'000024.XSHE' ,'000027.XSHE' ,'000031.XSHE' ,'000039.XSHE' ,'000046.XSHE' ,'000059.XSHE' ,'000060.XSHE' ,'000061.XSHE' ,'000063.XSHE' ,'000069.XSHE' ,'000100.XSHE' ,'000156.XSHE' ,'000157.XSHE' ,'000166.XSHE' ,'000333.XSHE' ,'000338.XSHE' ,'000400.XSHE' ,'000401.XSHE' ,'000402.XSHE' ,'000413.XSHE' ,'000415.XSHE' ,'000422.XSHE' ,'000423.XSHE' ,'000425.XSHE' ,'000503.XSHE' ,'000527.XSHE' ,'000528.XSHE' ,'000536.XSHE' ,'000538.XSHE' ,'000539.XSHE' ,'000540.XSHE' ,'000555.XSHE' ,'000559.XSHE' ,'000562.XSHE' ,'000568.XSHE' ,'000581.XSHE' ,'000596.XSHE' ,'000598.XSHE' ,'000612.XSHE' ,'000623.XSHE' ,'000625.XSHE' ,'000627.XSHE' ,'000629.XSHE' ,'000630.XSHE' ,'000651.XSHE' ,'000656.XSHE' ,'000671.XSHE' ,'000680.XSHE' ,'000685.XSHE' ,'000686.XSHE' ,'000690.XSHE' ,'000703.XSHE' ,'000709.XSHE' ,'000712.XSHE' ,'000718.XSHE' ,'000723.XSHE' ,'000725.XSHE' ,'000728.XSHE' ,'000729.XSHE' ,'000738.XSHE' ,'000750.XSHE' ,'000758.XSHE' ,'000768.XSHE' ,'000776.XSHE' ,'000778.XSHE' ,'000780.XSHE' ,'000783.XSHE' ,'000792.XSHE' ,'000793.XSHE' ,'000800.XSHE' ,'000807.XSHE' ,'000825.XSHE' ,'000826.XSHE' ,'000831.XSHE' ,'000839.XSHE' ,'000858.XSHE' ,'000869.XSHE' ,'000876.XSHE' ,'000878.XSHE' ,'000883.XSHE' ,'000895.XSHE' ,'000898.XSHE' ,'000917.XSHE' ,'000927.XSHE' ,'000933.XSHE' ,'000937.XSHE' ,'000938.XSHE' ,'000951.XSHE' ,'000959.XSHE' ,'000960.XSHE' ,'000961.XSHE' ,'000963.XSHE' ,'000968.XSHE' ,'000969.XSHE' ,'000970.XSHE' ,'000977.XSHE' ,'000983.XSHE' ,'000999.XSHE' ,'001979.XSHE' ,'002001.XSHE' ,'002007.XSHE' ,'002008.XSHE' ,'002024.XSHE' ,'002027.XSHE' ,'002028.XSHE' ,'002038.XSHE' ,'002044.XSHE' ,'002049.XSHE' ,'002051.XSHE' ,'002065.XSHE' ,'002069.XSHE' ,'002073.XSHE' ,'002074.XSHE' ,'002081.XSHE' ,'002085.XSHE' ,'002092.XSHE' ,'002106.XSHE' ,'002122.XSHE' ,'002128.XSHE' ,'002129.XSHE' ,'002131.XSHE' ,'002142.XSHE' ,'002146.XSHE' ,'002152.XSHE' ,'002153.XSHE' ,'002155.XSHE' ,'002174.XSHE' ,'002183.XSHE' ,'002195.XSHE' ,'002202.XSHE' ,'002230.XSHE' ,'002236.XSHE' ,'002241.XSHE' ,'002244.XSHE' ,'002252.XSHE' ,'002269.XSHE' ,'002292.XSHE' ,'002294.XSHE' ,'002299.XSHE' ,'002304.XSHE' ,'002310.XSHE' ,'002344.XSHE' ,'002352.XSHE' ,'002353.XSHE' ,'002375.XSHE' ,'002378.XSHE' ,'002385.XSHE' ,'002399.XSHE' ,'002400.XSHE' ,'002405.XSHE' ,'002410.XSHE' ,'002411.XSHE' ,'002415.XSHE' ,'002416.XSHE' ,'002422.XSHE' ,'002424.XSHE' ,'002426.XSHE' ,'002429.XSHE' ,'002431.XSHE' ,'002450.XSHE' ,'002456.XSHE' ,'002460.XSHE' ,'002465.XSHE' ,'002466.XSHE' ,'002468.XSHE' ,'002470.XSHE' ,'002475.XSHE' ,'002493.XSHE' ,'002498.XSHE' ,'002500.XSHE' ,'002508.XSHE' ,'002555.XSHE' ,'002558.XSHE' ,'002568.XSHE' ,'002570.XSHE' ,'002572.XSHE' ,'002594.XSHE' ,'002601.XSHE' ,'002602.XSHE' ,'002603.XSHE' ,'002608.XSHE' ,'002624.XSHE' ,'002653.XSHE' ,'002673.XSHE' ,'002714.XSHE' ,'002736.XSHE' ,'002739.XSHE' ,'002797.XSHE' ,'002831.XSHE' ,'002839.XSHE' ,'002841.XSHE' ,'300002.XSHE' ,'300003.XSHE' ,'300015.XSHE' ,'300017.XSHE' ,'300024.XSHE' ,'300027.XSHE' ,'300033.XSHE' ,'300058.XSHE' ,'300059.XSHE' ,'300070.XSHE' ,'300072.XSHE' ,'300085.XSHE' ,'300104.XSHE' ,'300122.XSHE' ,'300124.XSHE' ,'300133.XSHE' ,'300136.XSHE' ,'300144.XSHE' ,'300146.XSHE' ,'300168.XSHE' ,'300182.XSHE' ,'300251.XSHE' ,'300315.XSHE' ,'600000.XSHG' ,'600004.XSHG' ,'600005.XSHG' ,'600008.XSHG' ,'600009.XSHG' ,'600010.XSHG' ,'600011.XSHG' ,'600015.XSHG' ,'600016.XSHG' ,'600018.XSHG' ,'600019.XSHG' ,'600021.XSHG' ,'600022.XSHG' ,'600023.XSHG' ,'600026.XSHG' ,'600027.XSHG' ,'600028.XSHG' ,'600029.XSHG' ,'600030.XSHG' ,'600031.XSHG' ,'600036.XSHG' ,'600037.XSHG' ,'600038.XSHG' ,'600048.XSHG' ,'600050.XSHG' ,'600058.XSHG' ,'600060.XSHG' ,'600061.XSHG' ,'600062.XSHG' ,'600066.XSHG' ,'600068.XSHG' ,'600074.XSHG' ,'600079.XSHG' ,'600085.XSHG' ,'600089.XSHG' ,'600096.XSHG' ,'600098.XSHG' ,'600100.XSHG' ,'600104.XSHG' ,'600108.XSHG' ,'600109.XSHG' ,'600111.XSHG' ,'600115.XSHG' ,'600118.XSHG' ,'600123.XSHG' ,'600125.XSHG' ,'600132.XSHG' ,'600143.XSHG' ,'600150.XSHG' ,'600151.XSHG' ,'600153.XSHG' ,'600157.XSHG' ,'600160.XSHG' ,'600161.XSHG' ,'600166.XSHG' ,'600169.XSHG' ,'600170.XSHG' ,'600177.XSHG' ,'600183.XSHG' ,'600188.XSHG' ,'600196.XSHG' ,'600208.XSHG' ,'600216.XSHG' ,'600219.XSHG' ,'600220.XSHG' ,'600221.XSHG' ,'600233.XSHG' ,'600239.XSHG' ,'600252.XSHG' ,'600256.XSHG' ,'600259.XSHG' ,'600266.XSHG' ,'600267.XSHG' ,'600269.XSHG' ,'600271.XSHG' ,'600276.XSHG' ,'600277.XSHG' ,'600297.XSHG' ,'600300.XSHG' ,'600307.XSHG' ,'600309.XSHG' ,'600312.XSHG' ,'600315.XSHG' ,'600316.XSHG' ,'600317.XSHG' ,'600320.XSHG' ,'600331.XSHG' ,'600332.XSHG' ,'600340.XSHG' ,'600348.XSHG' ,'600350.XSHG' ,'600352.XSHG' ,'600362.XSHG' ,'600369.XSHG' ,'600372.XSHG' ,'600373.XSHG' ,'600376.XSHG' ,'600380.XSHG' ,'600383.XSHG' ,'600390.XSHG' ,'600395.XSHG' ,'600398.XSHG' ,'600403.XSHG' ,'600406.XSHG' ,'600415.XSHG' ,'600418.XSHG' ,'600428.XSHG' ,'600432.XSHG' ,'600436.XSHG' ,'600446.XSHG' ,'600456.XSHG' ,'600481.XSHG' ,'600482.XSHG' ,'600485.XSHG' ,'600489.XSHG' ,'600497.XSHG' ,'600498.XSHG' ,'600500.XSHG' ,'600508.XSHG' ,'600516.XSHG' ,'600517.XSHG' ,'600518.XSHG' ,'600519.XSHG' ,'600522.XSHG' ,'600528.XSHG' ,'600535.XSHG' ,'600546.XSHG' ,'600547.XSHG' ,'600549.XSHG' ,'600550.XSHG' ,'600570.XSHG' ,'600578.XSHG' ,'600582.XSHG' ,'600583.XSHG' ,'600585.XSHG' ,'600588.XSHG' ,'600595.XSHG' ,'600597.XSHG' ,'600598.XSHG' ,'600600.XSHG' ,'600606.XSHG' ,'600633.XSHG' ,'600635.XSHG' ,'600637.XSHG' ,'600642.XSHG' ,'600648.XSHG' ,'600649.XSHG' ,'600654.XSHG' ,'600655.XSHG' ,'600660.XSHG' ,'600663.XSHG' ,'600664.XSHG' ,'600666.XSHG' ,'600674.XSHG' ,'600675.XSHG' ,'600682.XSHG' ,'600685.XSHG' ,'600688.XSHG' ,'600690.XSHG' ,'600694.XSHG' ,'600703.XSHG' ,'600704.XSHG' ,'600705.XSHG' ,'600717.XSHG' ,'600718.XSHG' ,'600737.XSHG' ,'600739.XSHG' ,'600741.XSHG' ,'600754.XSHG' ,'600770.XSHG' ,'600779.XSHG' ,'600783.XSHG' ,'600795.XSHG' ,'600804.XSHG' ,'600808.XSHG' ,'600809.XSHG' ,'600811.XSHG' ,'600812.XSHG' ,'600816.XSHG' ,'600820.XSHG' ,'600823.XSHG' ,'600827.XSHG' ,'600832.XSHG' ,'600837.XSHG' ,'600839.XSHG' ,'600859.XSHG' ,'600863.XSHG' ,'600867.XSHG' ,'600871.XSHG' ,'600873.XSHG' ,'600875.XSHG' ,'600879.XSHG' ,'600880.XSHG' ,'600881.XSHG' ,'600886.XSHG' ,'600887.XSHG' ,'600893.XSHG' ,'600895.XSHG' ,'600900.XSHG' ,'600909.XSHG' ,'600919.XSHG' ,'600926.XSHG' ,'600958.XSHG' ,'600959.XSHG' ,'600970.XSHG' ,'600971.XSHG' ,'600977.XSHG' ,'600997.XSHG' ,'600998.XSHG' ,'600999.XSHG' ,'601001.XSHG' ,'601006.XSHG' ,'601009.XSHG' ,'601012.XSHG' ,'601016.XSHG' ,'601018.XSHG' ,'601021.XSHG' ,'601088.XSHG' ,'601098.XSHG' ,'601099.XSHG' ,'601101.XSHG' ,'601106.XSHG' ,'601107.XSHG' ,'601111.XSHG' ,'601117.XSHG' ,'601118.XSHG' ,'601127.XSHG' ,'601139.XSHG' ,'601155.XSHG' ,'601158.XSHG' ,'601163.XSHG' ,'601166.XSHG' ,'601168.XSHG' ,'601169.XSHG' ,'601179.XSHG' ,'601186.XSHG' ,'601198.XSHG' ,'601211.XSHG' ,'601212.XSHG' ,'601216.XSHG' ,'601225.XSHG' ,'601228.XSHG' ,'601229.XSHG' ,'601231.XSHG' ,'601233.XSHG' ,'601238.XSHG' ,'601258.XSHG' ,'601268.XSHG' ,'601288.XSHG' ,'601299.XSHG' ,'601318.XSHG' ,'601328.XSHG' ,'601333.XSHG' ,'601336.XSHG' ,'601369.XSHG' ,'601375.XSHG' ,'601377.XSHG' ,'601390.XSHG' ,'601398.XSHG' ,'601519.XSHG' ,'601555.XSHG' ,'601558.XSHG' ,'601566.XSHG' ,'601600.XSHG' ,'601601.XSHG' ,'601607.XSHG' ,'601608.XSHG' ,'601611.XSHG' ,'601618.XSHG' ,'601628.XSHG' ,'601633.XSHG' ,'601666.XSHG' ,'601668.XSHG' ,'601669.XSHG' ,'601688.XSHG' ,'601699.XSHG' ,'601717.XSHG' ,'601718.XSHG' ,'601727.XSHG' ,'601766.XSHG' ,'601788.XSHG' ,'601800.XSHG' ,'601808.XSHG' ,'601818.XSHG' ,'601857.XSHG' ,'601866.XSHG' ,'601872.XSHG' ,'601877.XSHG' ,'601878.XSHG' ,'601881.XSHG' ,'601888.XSHG' ,'601898.XSHG' ,'601899.XSHG' ,'601901.XSHG' ,'601918.XSHG' ,'601919.XSHG' ,'601928.XSHG' ,'601929.XSHG' ,'601933.XSHG' ,'601939.XSHG' ,'601958.XSHG' ,'601966.XSHG' ,'601969.XSHG' ,'601985.XSHG' ,'601988.XSHG' ,'601989.XSHG' ,'601991.XSHG' ,'601992.XSHG' ,'601997.XSHG' ,'601998.XSHG' ,'603000.XSHG' ,'603160.XSHG' ,'603288.XSHG' ,'603699.XSHG' ,'603799.XSHG' ,'603833.XSHG' ,'603858.XSHG' ,'603885.XSHG' ,'603993.XSHG' ]

start_time = time.time()
data = getCallAuctionRatioAll(temp_universe, begin_date, end_date, file_name='CallAuctionRatio_FullA.csv')
finish_time = time.time()

print (str(finish_time-start_time) + ' seconds elapsed in total.')


data1 = pd.read_csv('CallAuctionRatio_FullA.csv')
data1 = data1[data1.columns[:]].set_index('tradeDate')
s1 = data1.unstack().unstack().T
WINDOW_LENGTH = 20
s_ma = pd.rolling_mean(s1, window=WINDOW_LENGTH)
s_ma.to_csv('CallAuctionRatioMA%s_FullA.csv' % WINDOW_LENGTH)

# 提取数据
factor_data = pd.read_csv('CallAuctionRatioMA20_FullA.csv')    # 选股因子
mkt_value_data = pd.read_csv('MarketValues_FullA.csv')                    # 市值数据

factor_data['tradeDate'] = map(Date.toDateTime, map(DateTime.parseISO, factor_data['tradeDate']))
mkt_value_data['tradeDate'] = map(Date.toDateTime, map(DateTime.parseISO, mkt_value_data['tradeDate']))

factor_data = factor_data[factor_data.columns[:]].set_index('tradeDate')
mkt_value_data = mkt_value_data[mkt_value_data.columns[1:]].set_index('tradeDate')

factor_data[factor_data.columns[0:15]].tail()


# 因子历史表现

n_quantile = 10
# 统计十分位数
cols_mean = ['meanQ'+str(i+1) for i in range(n_quantile)]
cols = cols_mean
corr_means = pd.DataFrame(index=factor_data.index, columns=cols)

# 计算相关系数分组平均值
for dt in corr_means.index:
    qt_mean_results = []

    # 相关系数去掉nan和绝对值大于1的
    tmp_factor = factor_data.ix[dt].dropna()
    tmp_factor = tmp_factor[(tmp_factor<=1.0) & (tmp_factor>=-1.0)]
    
    pct_quantiles = 1.0/n_quantile
    for i in range(n_quantile):
        down = tmp_factor.quantile(pct_quantiles*i)
        up = tmp_factor.quantile(pct_quantiles*(i+1))
        mean_tmp = tmp_factor[(tmp_factor<=up) & (tmp_factor>=down)].mean()
        qt_mean_results.append(mean_tmp)
    corr_means.ix[dt] = qt_mean_results


# ------------- 因子历史表现作图 ------------------------

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)

lns1 = ax1.plot(corr_means.index, corr_means.meanQ1, label='Q1')
lns2 = ax1.plot(corr_means.index, corr_means.meanQ5, label='Q5')
lns3 = ax1.plot(corr_means.index, corr_means.meanQ10, label='Q10')

lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=[0.5, 0.1], loc='', ncol=3, mode="", borderaxespad=0., fontsize=12)
ax1.set_ylabel(u'因子', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'日期', fontproperties=font, fontsize=16)
ax1.set_title(u"因子历史表现", fontproperties=font, fontsize=16)
ax1.grid()


# 计算因子分组的市值分位数平均值
def quantile_mkt_values(signal_df, mkt_df):
    n_quantile = 10
    # 统计十分位数
    cols_mean = [i+1 for i in range(n_quantile)]
    cols = cols_mean

    mkt_value_means = pd.DataFrame(index=signal_df.index, columns=cols)

    # 计算相关系数分组的市值分位数平均值
    for dt in mkt_value_means.index:
        qt_mean_results = []

        # 相关系数去掉nan和绝对值大于0.97的
        tmp_factor = signal_df.ix[dt].dropna()
        tmp_factor = tmp_factor[(tmp_factor<=0.97) & (tmp_factor>=-0.97)]
        tmp_mkt_value = mkt_df.ix[dt].dropna()
        tmp_mkt_value = tmp_mkt_value.rank()/len(tmp_mkt_value)

        pct_quantiles = 1.0/n_quantile
        for i in range(n_quantile):
            down = tmp_factor.quantile(pct_quantiles*i)
            up = tmp_factor.quantile(pct_quantiles*(i+1))
            i_quantile_index = tmp_factor[(tmp_factor<=up) & (tmp_factor>=down)].index
            mean_tmp = tmp_mkt_value[i_quantile_index].mean()
            qt_mean_results.append(mean_tmp)
        mkt_value_means.ix[dt] = qt_mean_results
    mkt_value_means.dropna(inplace=True)
    return mkt_value_means.mean()
    
# 计算因子分组的市值分位数平均值
origin_mkt_means = quantile_mkt_values(factor_data, mkt_value_data)


# 因子分组的市值分位数平均值作图
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)

width = 0.3
lns1 = ax1.bar(origin_mkt_means.index, origin_mkt_means.values, align='center', width=width)

ax1.set_ylim(0.3,0.6)
ax1.set_xlim(left=0.5, right=len(origin_mkt_means)+0.5)
ax1.set_ylabel(u'市值百分位数', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'十分位分组', fontproperties=font, fontsize=16)
ax1.set_xticks(origin_mkt_means.index)
ax1.set_xticklabels([int(x) for x in ax1.get_xticks()], fontproperties=font, fontsize=14)
ax1.set_yticklabels([str(x*100)+'0%' for x in ax1.get_yticks()], fontproperties=font, fontsize=14)
ax1.set_title(u"因子分组市值分布特征", fontproperties=font, fontsize=16)
ax1.grid()


'''

start = '2014-02-01'                       # 回测起始时间
end = '2018-01-30'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = set_universe('HS300')               # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                           # 调仓频率，表示执行handle_data的时间间隔

factor_data = pd.read_csv('CallAuctionRatioMA20_FullA.csv')     # 读取因子数据
factor_data = factor_data[factor_data.columns[:]].set_index('tradeDate')
factor_dates = factor_data.index.values

quantile_ten = 1                           # 选取股票的因子十分位数，1表示选取股票池中因子最小的10%的股票
commission = Commission(0.0002,0.0002)     # 交易费率设为双边万分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    
    # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
    q = factor_data.ix[pre_date].dropna()
    q = q[q>0]
    q_min = q.quantile((quantile_ten-1)*0.1)
    q_max = q.quantile(quantile_ten*0.1)
    my_univ = q[q>=q_min][q<q_max].index.values
    
    # 调仓逻辑
    univ = [x for x in my_univ if x in account.universe]
    
    # 不在股票池中的，清仓
    for stk in account.security_position:
        if stk not in univ:
            order_to(stk, 0)
    # 在目标股票池中的，等权买入
    for stk in univ:
        order_pct_to(stk, 1.1/len(univ))
        
'''

fig = plt.figure(figsize=(12,5))
fig.set_tight_layout(True)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.grid()

bt_quantile_ten = bt
data = bt_quantile_ten[[u'tradeDate',u'portfolio_value',u'benchmark_return']]
data['portfolio_return'] = data.portfolio_value/data.portfolio_value.shift(1) - 1.0
data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/	10000000.0 - 1.0
data['excess_return'] = data.portfolio_return - data.benchmark_return
data['excess'] = data.excess_return + 1.0
data['excess'] = data.excess.cumprod()
data['portfolio'] = data.portfolio_return + 1.0
data['portfolio'] = data.portfolio.cumprod()
data['benchmark'] = data.benchmark_return + 1.0
data['benchmark'] = data.benchmark.cumprod()
# ax.plot(data[['portfolio','benchmark','excess']], label=str(qt))
ax1.plot(data['tradeDate'], data[['portfolio']], label='portfolio(left)')
ax1.plot(data['tradeDate'], data[['benchmark']], label='benchmark(left)')
ax2.plot(data['tradeDate'], data[['excess']], label='hedged(right)', color='r')

ax1.legend(loc=2)
ax2.legend(loc=0)
# ax2.set_ylim(bottom=0.5, top=2.5)
ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
ax2.set_ylabel(u"对冲指数净值", fontproperties=font, fontsize=16)
ax2.set_ylabel(u"对冲指数净值", fontproperties=font, fontsize=16)
ax1.set_title(u"因子最小的10%股票月度调仓走势", fontproperties=font, fontsize=16)


plot_pure_alpha(bt, 10000000)

'''

start = '2014-02-10'                       # 回测起始时间
end = '2018-01-30'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = set_universe('HS300')               # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = Monthly(1)                           # 调仓频率，表示执行handle_data的时间间隔

factor_data = pd.read_csv('CallAuctionRatioMA20_FullA.csv')     # 读取因子数据
factor_data = factor_data[factor_data.columns[:]].set_index('tradeDate')
factor_dates = factor_data.index.values

quantile_ten = 1                           # 选取股票的因子十分位数，1表示选取股票池中因子最小的10%的股票
commission = Commission(0.0002,0.0002)     # 交易费率设为双边万分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    
    # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
    q = factor_data.ix[pre_date].dropna()
    q = q[q>0]
    q_min = q.quantile((quantile_ten-1)*0.1)
    q_max = q.quantile(quantile_ten*0.1)
    my_univ = q[q>=q_min][q<q_max].index.values
    
    # 调仓逻辑
    univ = [x for x in my_univ if x in account.universe]
    
    # 不在股票池中的，清仓
    for stk in account.security_position:
        if stk not in univ:
            order_to(stk, 0)
    # 在目标股票池中的，等权买入
    for stk in univ:
        order_pct_to(stk, 1.1/len(univ))
        
'''
plot_pure_alpha(bt, 10000000)