clear

print_sel = true;

tref = yq_methods.get_tradingdate('2013-05-01','2020-01-13');
tref_num = datenum(tref);
%获取月底日期
%last day for the month
month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
month_cut_date1 = tref(month_cut(:,1));
month_cut_date2 = tref(month_cut(:,2));

sql_str1 = 'select symbol,f_val from S32.factor_delta where tradingdate = ''%s''';

sql_str2 = 'select symbol,f_val from S32.ret20d_update where tradingdate = ''%s''';


T = length(month_cut_date2);
r = zeros(T,1);
r2 = r;
for i = 1:T

    x = fetchmysql(sprintf(sql_str1,month_cut_date2{i}),2);
    y = fetchmysql(sprintf(sql_str2,month_cut_date2{i}),2);

    inds = suscc_intersect({x(:,1),y(:,1)});
    x = cell2mat(x(inds(:,1),2));
    y = cell2mat(y(inds(:,2),2));
    r(i) = corr(x,y);
    r2(i) = corr(x,y,'type','Spearman');
    if print_sel
        sprintf('%d-%d',i,T)
    end
end

t_str = month_cut_date2;
T=length(t_str);
figure
subplot(2,1,1)
bar(r)
set(gca,'xlim',[0.5,T+0.5]);
set(gca,'XTick',floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)  
subplot(2,1,2)
bar(r2)
set(gca,'xlim',[0.5,T+0.5]);
set(gca,'XTick',floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)  
setpixelposition(gcf,[223,365,1345,420*2]);
movegui(gcf,'center');

