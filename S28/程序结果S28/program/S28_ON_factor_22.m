%2.2 买卖单不平衡度因子
% B 和 S 分别表示收盘前 N 分钟区间内买一和卖一委托量的平均值
%使用pytdx的分笔成交数据代替
clear
key_str = 'S28:合成因子2';
f_type = 2;
tn_f = 'S28.comfactors';
var_info = {'symbol','tradingdate','f_type','f_val'};

t1 = 1430;
tt = 1500;

t_terminal = fetchmysql('show tables from Future_tick',2);
num = cellfun(@(x) str2double(x(2:end)),t_terminal);
[~,ia] = sort(-num);
t_terminal = t_terminal{ia(1)};
t_terminal = fetchmysql(sprintf(['select date(tradingdate) from ',...
    'Future_tick.%s order by tradingdate desc limit 1'],t_terminal),2);
t_terminal = t_terminal{1};

sql_str_f1 = ['SELECT tradeDate,ticker FROM yuqerdata.yq_MktMFutdGet ',...
    'where contractObject=''%s'' and mainCon=1 and tradeDate>''%s'' ',...
    'and tradeDate <= ''%s'' order by tradedate'];

sql_str_f2 = ['select BV1,SV1 from Future_tick.y%s where date(tradingdate)=''%s'' and ',...
    'ticker = ''%s'' and hour(tradingdate)*100+minute(tradingdate)>=1430 and ',...
    ' hour(tradingdate)*100+minute(tradingdate)<=1500'];

sql_str1 = ['select tradingdate from S28.comfactors where f_type=%d and ',...
    'symbol = ''%s'' order by tradingdate desc limit 1'];

dns = {'IF','IH','IC'};
T_dns = length(dns);
for i = 1:T_dns
    sub_str = dns{i};
    t0 = fetchmysql(sprintf(sql_str1,f_type,sub_str),2);
    if isempty(t0)
        t0 = '2016-12-28';
    else
        t0 = t0{1};
    end
    
    sub_code = fetchmysql(sprintf(sql_str_f1,sub_str,t0,t_terminal),2);
    if isempty(sub_code)
        continue
    end
    T = size(sub_code,1);
    F = zeros(T,1);
    ef_ind = zeros(T,1);
    
    parfor j = 1:T
        sub_sub_tref = sub_code{j,1};
        sub_sub_code = sub_code{j,2};
        
        sub_sql_str = sprintf(sql_str_f2,sub_sub_tref([1:4,6:7]),sub_sub_tref,sub_sub_code);
        sub_x = fetchmysql(sub_sql_str);
        if size(sub_x,1)>0
            sub_B = mean(sub_x(:,1));
            sub_S = mean(sub_x(:,2));
            sub_sub_f = (sub_S-sub_B)/((sub_S+sub_B)/2);
            F(j) = sub_sub_f;
        else
            ef_ind(j) = 1;
            F(j) = 0;
        end
        sprintf('%s-%s:%d-%d',key_str,sub_str,j,T)
    end
    F = [sub_code(:,1),sub_code(:,1),sub_code(:,1),num2cell(F)];
    F(:,1) = {sub_str};
    F(:,3) = {f_type};
    datainsert_adair(tn_f,var_info,F)
    
end