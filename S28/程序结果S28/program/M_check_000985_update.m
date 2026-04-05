clear
sql_str = 'SELECT tradeDate,CHGpct FROM yuqerdata.yq_index where symbol = ''000985'' and month(tradeDate) = 2 order by tradeDate;';

x = fetchmysql(sql_str,2);
tref_num = datenum(x(:,1));
y = cell2mat(x(:,2));

year_num = year(tref_num);
ind = find(diff(year_num));
if ~eq(ind(end),length(year_num))
    ind = [ind;length(year_num)];
end
year_ind = [0;ind];
year_ind = [year_ind(1:end-1)+1,year_ind(2:end)];

T = size(year_ind,1);
sub_y = zeros(T,1);
for i = 1:T
    sub_ind = year_ind(i,1):year_ind(i,2);
    sub_r = cumprod(1+y(sub_ind))-1;
    sub_y(i)=sub_r(end);
end

sub_x = year_num(year_ind(:,1));

bar(sub_x,sub_y*100)

setpixelposition(gcf,[430,368,1008,420]);
movegui(gcf,'center');