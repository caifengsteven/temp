clear

sql_str =[ 'select tradedate,pe1 from yuqerdata.yq_dayprice where ',...
    'symbol = ''600519'' and pe1 is not null order by tradedate '];
x = fetchmysql(sql_str,2);

tref = x(:,1);
%找到月底最后一天
tref_num = datenum(tref);
month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
month_cut_date1 = tref(month_cut(:,1));
month_cut_date2 = tref(month_cut(:,2));

y = cell2mat(x(month_cut(:,2),2));

y1 = smooth_s30(y,0.1);

plot([y,y1']);