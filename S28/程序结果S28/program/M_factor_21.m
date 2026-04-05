%{
多头策略：收盘价低于结算价做多，持有至次日上午 10 点平仓。
多空策略：收盘价低于结算价做多，高于结算价做空，持有至次日上午 10 点平仓
update
增加了连续合约换约判断，换约当日隔夜收益率不计入
连续合约代码数据来源yuqer
%}
clear

close all
sql_str  ='select date(tradingdate),hour(tradingdate)*100+minute(tradingdate),pct_chg from S28.wind_%s where pct_chg is not null';
sql_str_2 = 'SELECT tradeDate,ticker,closeprice-settlePrice FROM futuredata.yuqer_fusharedata where contractObject=''%s'' and mainCon=1 order by tradedate';
cut_time = [930,1000];
cut_str = {'多头净值','多空净值'};

dns = {'IF','IH','IC'};
T_dns = length(dns);
for i0 = 1:T_dns
    sub_str = dns{i0};
    sub_str_l = length(sub_str);
    x = fetchmysql(sprintf(sql_str,sub_str),2);
    sub_code = fetchmysql(sprintf(sql_str_2,sub_str),2);
    tref_all = x(:,1);
    tref = unique(tref_all);
    [~,~,ia] = intersect(tref,sub_code(:,1),'stable');
    if ~eq(length(ia),length(tref))
        continue
    end
    sub_signal = cell2mat(sub_code(ia,3));
    
    F = [tref,num2cell(sub_signal)];
    save(sprintf('F21_%s.mat',sub_str),'F')

end