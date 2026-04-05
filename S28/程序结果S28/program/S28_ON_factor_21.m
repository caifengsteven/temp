%{
多头策略：收盘价低于结算价做多，持有至次日上午 10 点平仓。
多空策略：收盘价低于结算价做多，高于结算价做空，持有至次日上午 10 点平仓
update
增加了连续合约换约判断，换约当日隔夜收益率不计入
连续合约代码数据来源yuqer
数据写入数据据


#分钟数据需要想下办法

%}
clear

f_type = 1;
tn_f = 'S28.comfactors';
var_info = {'symbol','tradingdate','f_type','f_val'};

sql_str  =['select tradingdate,t_hour*100+t_minute,price from pytdx_data.%s_tdx_min ',...
    'where price is not null and price>0 and tradingdate>''%s'' order by tradingdate,t_hour,t_minute'];
sql_str_2 = ['SELECT tradeDate,ticker,closeprice-settlePrice FROM ',...
    'yuqerdata.yq_MktMFutdGet where contractObject=''%s'' and mainCon=1 and tradeDate >''%s'' order by tradedate'];
cut_time = [930,1000];
cut_str = {'多头净值','多空净值'};

dns = {'IF','IH','IC'};
T_dns = length(dns);
%初始时间
sql_str1 = ['select tradingdate from S28.comfactors where f_type=%d and ',...
    'symbol = ''%s'' order by tradingdate desc limit 1'];

for i0 = 1:T_dns
    sub_str = dns{i0};
    t0 = fetchmysql(sprintf(sql_str1,f_type,sub_str),2);
    if isempty(t0)
        t0 = '1999-01-01';
    else
        t0 = t0{1};
    end
    x = fetchmysql(sprintf(sql_str,sub_str,t0),2);
    sub_code = fetchmysql(sprintf(sql_str_2,sub_str,t0),2);
    if isempty(x) || isempty(sub_code)
        continue
    end
    tref_all = x(:,1);
    tref = unique(tref_all);
    [~,~,ia] = intersect(tref,sub_code(:,1),'stable');
    if ~eq(length(ia),length(tref))
        continue
    end
    sub_signal = cell2mat(sub_code(ia,3));
    
    F = [tref,tref,tref,num2cell(sub_signal)];
    F(:,1) = {sub_str};
    F(:,3) = {f_type};
    %save(sprintf('F21_%s.mat',sub_str),'F')
    datainsert_adair(tn_f,var_info,F)
end