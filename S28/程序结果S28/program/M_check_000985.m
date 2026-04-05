clear
sql_str = 'select endDate,chgPct from yuqerdata.yq_index_month where symbol = ''000985'' order by endDate';

x = fetchmysql(sql_str,2);
tref_num = datenum(x(:,1));
y = cell2mat(x(:,2));

month_num = month(tref_num);
ind = eq(month_num,2);
sub_y = y(ind);
sub_x = tref_num(ind);

bar(sub_x,sub_y)
datetick('x','yyyy');
