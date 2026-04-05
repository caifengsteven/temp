%IC ICIR t_value 
%got date and cal IC
clear
%parameters
dn = 'S23';
tn = 'fenbifactor1_month';
tn_fullname = sprintf('%s.%s',dn,tn);

tn_1month_return = 'yuqerdata.future_return_1m';

dn_yq = 'yuqerdata';
tn_yq = 'yq_dayprice';
tn_yq_fullname = sprintf('%s.%s',dn_yq,tn_yq);
%month_cut
t0 = '2011-02-01';
tt = '2017-01-01';
sql_str = 'select distinct tradedate from %s order by tradedate';
tref = fetchmysql(sprintf(sql_str,tn_yq_fullname),2);
if istable(tref)
    tref = table2cell(tref);
end
tref_num = datenum(tref);
ind_cut = tref_num>=datenum(t0)&tref_num<=datenum(tt);
tref_num = tref_num(ind_cut);
tref = tref(ind_cut);

month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
T_month_cut = size(month_cut,1);
month_cut_date = tref(month_cut(:,2));


sql_str_f = 'select symbol,f_val from %s where tradingdate = ''%s''';
ic = zeros(T_month_cut,1);
p = ic;
Y = cell(T_month_cut,1);
for i = 15:T_month_cut
    %factor data
    x = fetchmysql(sprintf(sql_str_f,tn_fullname,month_cut_date{i}),2);
    if istable(x)
        x = table2cell(x);
    end
    %return 
    y = fetchmysql(sprintf(sql_str_f,tn_1month_return,month_cut_date{i}),2);
    if istable(y)
        y = table2cell(y);
    end
    
    [~,ia,ib] = intersect(x(:,1),y(:,1));
    x_v = cell2mat(x(ia,2));
    y_v = cell2mat(y(ib,2));
    
    [ic(i),p(i)] = corr(x_v,y_v,'Type','Spearman');
    
    [~,ia] = sort(x_v);
    y_v = y_v(ia);
    ind_cut = floor(linspace(0,length(y_v),11));
    temp = zeros(10,1);
    for j = 1:length(ind_cut)-1
        temp(j) = mean(y_v((ind_cut(j)+1):ind_cut(j+1)));
    end
    Y{i} = temp;
        
    sprintf('%d-%d',i,T_month_cut)
    
end
Y=[Y{:}]';

y_curve = cumprod(1+Y);
nh_all = zeros(10,1);
for i = 1:10
    [v,v_str,sta_val] = curve_static_month(y_curve(:,i));
    nh_all(i) = sta_val.nh*100;
end
r_month = sum(Y(:,6:end)-Y(:,1:5),2)/2;
y_curve_end = cumprod(1+r_month);

leg_str = cell(10,1);
for i = 1:10
    leg_str{i} = sprintf('S%d',i);
end
figure
subplot(2,1,1)
bar(nh_all)
ylabel('�껯����%')
subplot(2,1,2)
plot(y_curve,'LineWidth',2)
legend(leg_str,'NumColumns',2)
ylabel('��������')

figure
subplot(2,1,1)
bar(r_month*100)
ylabel('�������¶�����%')
set(gca,'XTickLabelRotation',90);
set(gca,'XTick',floor(linspace(1,length(month_cut_date),30)),'xlim',[1,length(month_cut_date)]);
set(gca,'XTickLabel',month_cut_date(floor(linspace(1,length(month_cut_date),30))));
subplot(2,1,2)
bpcure_plot_updateV2(month_cut_date,y_curve_end)


