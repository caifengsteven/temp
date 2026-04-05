%M_static2
%1/2法则


warning('off');
clear
t0 = datenum(1007,6,9);
tt = datenum(2008,1,23);

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
T = length(r1);
sub_x = zeros(T,1);
sub_y = sub_x;
for i = 5:T
    sub_y(i) = std(r1(1:i));
    sub_x(i) = i;
end
figure
plot(log(sub_x(5:end)),log(sub_y(5:end)),'+');



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