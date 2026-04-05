%2.2 买卖单不平衡度因子
% B 和 S 分别表示收盘前 N 分钟区间内买一和卖一委托量的平均值

clear

t1 = 1430;
tt = 1500;

sql_str_f1 = ['SELECT tradeDate,ticker FROM futuredata.yuqer_fusharedata ',...
    'where contractObject=''%s'' and mainCon=1 and tradeDate>= ''2016-12-29'' ',...
    'and tradeDate <= ''2019-10-31'' order by tradedate'];

sql_str_f2 = ['select BV1,SV1 from Future_tick.y%s where date(tradingdate)=''%s'' and ',...
    'ticker = ''%s'' and hour(tradingdate)*100+minute(tradingdate)>=1430 and ',...
    ' hour(tradingdate)*100+minute(tradingdate)<=1500'];
%sql_str_f2 读入买一，卖一委托量，我们对时间进行了限制，只读取14：30～15：00之间数据
%关键步骤
dns = {'IF','IH','IC'};
T_dns = length(dns);
for i = 2:T_dns
    sub_str = dns{i};
    sub_code = fetchmysql(sprintf(sql_str_f1,sub_str),2);
    
    T = size(sub_code,1);
    F = zeros(T,1);
    parfor j = 1:T
        sub_sub_tref = sub_code{j,1};
        sub_sub_code = sub_code{j,2};
        
        sub_sql_str = sprintf(sql_str_f2,sub_sub_tref([1:4,6:7]),sub_sub_tref,sub_sub_code);
        sub_x = fetchmysql(sub_sql_str);
        
        sub_B = mean(sub_x(:,1));
        sub_S = mean(sub_x(:,2));
        
        sub_sub_f = (sub_S-sub_B)/((sub_S+sub_B)/2);
        
        F(j) = sub_sub_f;
        sprintf('%s:%d-%d',sub_str,j,T)
    end
    F = [sub_code(:,1),num2cell(F)];
    save(sprintf('F22_%s.mat',sub_str),'F')
    

end