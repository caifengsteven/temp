%M_static3
%自相关性
clear
t0 = datenum(2006,6,9);
tt = datenum(2017,6,9);

sub_data_info = 'sz399300';
sql_str = ['select datadate,openP,closeP from research..sec_index ',10,...
    'where index_code = ''%s'' and datadate>= ''%s'' and datadate<= ''%s'' order by datadate'];
x = fetchsqlserver(sprintf(sql_str,sub_data_info,datestr(t0,'yyyy-mm-dd'),datestr(tt,'yyyy-mm-dd')),2);

tref = datenum(x(:,1));

open_price = cellfun(@str2double,x(:,2));
close_price = cellfun(@str2double,x(:,3));

r1 = close_price(2:end)./close_price(1:end-1)-1;


autocorr(r1,'NumLags',100);
