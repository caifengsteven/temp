%按照文献说明计算开仓、平仓价格
%以收盘前 30 秒平均卖一价作为买入开仓价，以次日上午 9:59:30-10:00:30 一分
%钟内平均买一价作为卖出平仓价

clear

t1 = 1459.30; %开仓
tt = 1500;

t3 = 959.30; %平仓
t4 = 1000.30;

sql_str_f1 = ['SELECT tradeDate,ticker FROM futuredata.yuqer_fusharedata ',...
    'where contractObject=''%s'' and mainCon=1 and tradeDate>= ''2016-12-29'' ',...
    'and tradeDate <= ''2019-10-31'' order by tradedate'];
%我们构建了指标 小时×100+分钟+秒/100，用于后续计算时间节点
sql_str_f2 = ['select hour(tradingdate)*100+minute(tradingdate)+second(tradingdate)/100,',...
    'Buy1,Sail1 from Future_tick.y%s where date(tradingdate)=''%s'' and ',...
    'ticker = ''%s'' and Buy1 is not null and Sail1 is not null order by tradingdate'];

dns = {'IF','IH','IC'};
T_dns = length(dns);
for i = 1:T_dns
    sub_str = dns{i};
    sub_code = fetchmysql(sprintf(sql_str_f1,sub_str),2);
    
    T = size(sub_code,1);
    F = cell(T,1);
    parfor j = 1:T
        sub_sub_tref = sub_code{j,1};
        sub_sub_code = sub_code{j,2};
        
        sub_sql_str = sprintf(sql_str_f2,sub_sub_tref([1:4,6:7]),sub_sub_tref,sub_sub_code);
        sub_x = fetchmysql(sub_sql_str);
        sub_indB = sub_x(:,1)>=t1 & sub_x(:,1)<=tt;
        sub_indS = sub_x(:,1)>=t3 & sub_x(:,1)<=t4;
        sub_B = mean(sub_x(sub_indB,2));
        sub_S = mean(sub_x(sub_indS,3));
        
        F(j) = {[sub_B;sub_S]};
        sprintf('%s:%d-%d',sub_str,j,T)
    end
    
    F = [F{:}]';
    temp = nan(size(F(:,1)));
    temp(2:end) = F(1:end-1,1);
    F = [F,temp];    
    F = [sub_code(:,1),num2cell(F)];
    save(sprintf('data_update_%s.mat',sub_str),'F')
    

end