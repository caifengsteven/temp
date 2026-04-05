%M_static1
%均值 标准差 偏度 峰度 J-B检验 J-B检验临界值
clear
t0 = datenum(2006,6,9);
tt = datenum(2017,6,9);

sub_data_info = 'sh000001';
sql_str = ['select datadate,openP,closeP from research..sec_index ',10,...
    'where index_code = ''%s'' and datadate>= ''%s'' and datadate<= ''%s'' order by datadate'];
x = fetchsqlserver(sprintf(sql_str,sub_data_info,datestr(t0,'yyyy-mm-dd'),datestr(tt,'yyyy-mm-dd')),2);

tref = datenum(x(:,1));

open_price = cellfun(@str2double,x(:,2));
close_price = cellfun(@str2double,x(:,3));

r1 = close_price(2:end)./close_price(1:end-1)-1;
%r1 = close_price./open_price-1;
mean(r1)
