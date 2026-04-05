%M_static1
%均值 标准差 偏度 峰度 J-B检验 J-B检验临界值
warning('off');
clear
t0 = datenum(2007,6,9);
tt = datenum(2017,6,9);

sub_data_info = {'上证指数','深证成指'};
sub_data_info = sub_data_info{1};
sql_str = ['select tradingdate,open,close from futuredata.indicator_data ',10,...
    'where symbolname = ''%s'' and tradingdate>= ''%s'' and tradingdate<= ''%s'' order by tradingdate'];
x = fetchmysql(sprintf(sql_str,sub_data_info,datestr(t0,'yyyy-mm-dd'),datestr(tt,'yyyy-mm-dd')),2);

tref = datenum(x(:,1));
open_price = cell2mat(x(:,2));
close_price = cell2mat(x(:,3));
[tref_w,open_price_w,close_price_w] = get_week_data(tref,open_price,close_price);

r1 = close_price(2:end)./close_price(1:end-1)-1;
r1_w = close_price_w(2:end)./close_price_w(1:end-1)-1;

%
n = 10;
A = floor(length(r1)/n);
R = r1(end-A*n+1:end);%截取数据

R = reshape(R,n,A); %按照列分组
%均值处理
r_g_mx = mean(r_g);
r_g = bsxfun(@minus,r_g,r_g_mx);
%累计离差计算
X_Na = cumsum(r_g);
%计算单个区间极差
R_a = range(X_Na);
%计算重标极差
S_a = std(R,1);










%获取周数据
function [tref_w,p_open_w,p_close_w] = get_week_data(tref,p_open,p_close)
week_num = weeknum(tref);
ind = find(diff(week_num));
ind = [0;ind;length(tref)];

ind = [ind(1:end-1)+1,ind(2:end)];
p_open_w = p_open(ind(:,1));
p_close_w = p_close(ind(:,2));
tref_w = tref(ind(:,2));
end

