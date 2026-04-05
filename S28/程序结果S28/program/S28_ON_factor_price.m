%2.2 买卖单不平衡度因子
% B 和 S 分别表示收盘前 N 分钟区间内买一和卖一委托量的平均值

clear
tn = 'S28.bac_price';   
var_info = {'symbol','tradingdate','p1','p2','p3'};

t1 = 1459.30; %buy
tt = 1500;

t3 = 959.30; %sail
t4 = 1000.30;

t_terminal = fetchmysql('show tables from Future_tick',2);
num = cellfun(@(x) str2double(x(2:end)),t_terminal);
[~,ia] = sort(-num);
t_terminal = t_terminal{ia(1)};
t_terminal = fetchmysql(sprintf(['select date(tradingdate) from ',...
    'Future_tick.%s order by tradingdate desc limit 1'],t_terminal),2);
t_terminal = t_terminal{1};

sql_str_f1 = ['SELECT tradeDate,ticker FROM yuqerdata.yq_MktMFutdGet ',...
    'where contractObject=''%s'' and mainCon=1 and tradeDate>= ''%s'' ',...
    'and tradeDate <= ''%s'' order by tradedate'];

sql_str_f2 = ['select hour(tradingdate)*100+minute(tradingdate)+second(tradingdate)/100,',...
    'Buy1,Sail1 from Future_tick.y%s where date(tradingdate)=''%s'' and ',...
    'ticker = ''%s'' and Buy1 is not null and Sail1 is not null order by tradingdate'];

dns = {'IF','IH','IC'};
T_dns = length(dns);
sql_str1 = ['select tradingdate from S28.bac_price where ',...
    'symbol = ''%s'' order by tradingdate desc limit 1'];

for i = 1:T_dns
        
    sub_str = dns{i};
    t0 = fetchmysql(sprintf(sql_str1,sub_str),2);
    if isempty(t0)
        t0 = '2016-12-29';
    else
        t0 = t0{1};
    end    
    sub_code = fetchmysql(sprintf(sql_str_f1,sub_str,t0,t_terminal),2);    
    T = size(sub_code,1);
    F = cell(T,1);
    parfor j = 1:T
        sub_sub_tref = sub_code{j,1};
        sub_sub_code = sub_code{j,2};
        
        sub_sql_str = sprintf(sql_str_f2,sub_sub_tref([1:4,6:7]),sub_sub_tref,sub_sub_code);
        sub_x = fetchmysql(sub_sql_str);
        if size(sub_x,1)>0
            sub_indB = sub_x(:,1)>=t1 & sub_x(:,1)<=tt;
            sub_indS = sub_x(:,1)>=t3 & sub_x(:,1)<=t4;
            sub_B = mean(sub_x(sub_indB,2));
            sub_S = mean(sub_x(sub_indS,3));
        else
            sub_B = 0;
            sub_S = 0;
        end
        F(j) = {[sub_B;sub_S]};
        sprintf('%s:%d-%d',sub_str,j,T)
    end
    
    F = [F{:}]';
    temp = nan(size(F(:,1)));
    temp(2:end) = F(1:end-1,1);
    F = [F,temp];    
    F(isnan(F)) = 0;
    F = [sub_code(:,[1,1]),num2cell(F)];
    F(:,1) = {sub_str};
    F = F(2:end,:);
    datainsert_adair(tn,var_info,F);
end