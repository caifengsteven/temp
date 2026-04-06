clear
t1 = '2017-01-13';
t2 = '2020-01-13';
t1_num = datenum(t1);
t2_num = datenum(t2);
%分钟行情数据
info = '分钟行情数据';
tref = yq_methods.get_tradingdate(t1,t2);

x = fetchmysql('show tables from ycz_min_history',2);
x = cellstr(datestr(datenum(x,'yyyymmdd'),'yyyy-mm-dd'));

temp = setdiff(tref,x);
if isempty(temp)
    OK = true;
end
report_check_info(OK,info)
%日期行情
info = '日期行情';
tref = yq_methods.get_tradingdate(t1,t2);
OK = false;
if ~isempty(tref)
    if strcmp(tref(1),t1) && strcmp(tref(end),t2)
        OK = true;
    end
end
report_check_info(OK,info)
%基金日行情
info = '基金日行情';
sql_str = 'select min(tradedate),max(tradedate) from yuqerdata.MktFunddGet';
tref = fetchmysql(sql_str,2);
OK = false;
if ~isempty(tref)
    tref_num = datenum(tref);
    if min(tref_num)<=t1_num && max(tref_num)>=t2_num
        OK = true;
    end
end
report_check_info(OK,info)
%期货主力合约行情
info = '期货主力合约行情';
sql_str = 'select min(tradedate),max(tradedate) from yuqerdata.MktMFutdGet';
tref = fetchmysql(sql_str,2);
OK = false;
if ~isempty(tref)
    tref_num = datenum(tref);
    if min(tref_num)<=t1_num && max(tref_num)>=t2_num
        OK = true;
    end
end
report_check_info(OK,info)
%ETF基金申赎清单成分券信息
info = 'ETF基金申赎清单成分券信息';
sql_str = 'select min(tradedate),max(tradedate) from S31.FundETFConsGet';
tref = fetchmysql(sql_str,2);
OK = false;
if ~isempty(tref)
    tref_num = datenum(tref);
    if min(tref_num)<=t1_num && max(tref_num)>=t2_num
        OK = true;
    end
end
report_check_info(OK,info)
%ETF基金申赎清单基本信息
info = 'ETF基金申赎清单基本信息';
sql_str = 'select min(tradedate),max(tradedate) from S31.FundETFPRListGet';
tref = fetchmysql(sql_str,2);
OK = false;
if ~isempty(tref)
    tref_num = datenum(tref);
    if min(tref_num)<=t1_num && max(tref_num)>=t2_num
        OK = true;
    end
end
report_check_info(OK,info)
%ETF分钟数据
code_name_pool = {'etf50_min','etf300_min','etf500_min'};
info = 'ETF分钟数据-%s';
for i = 1:length(code_name_pool)
    sub_x = fetchmysql(sprintf('select * from S31.%s limit 1',code_name_pool{i}),2);
    OK = ~isempty(sub_x);
    report_check_info(OK,sprintf(info,code_name_pool{i}))
end