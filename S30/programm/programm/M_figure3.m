%gta data
clear

[~,~,x] = xlsread('PE1TTM.csv');
x = x(2:end,:);

tref = x(:,1);
%找到月底最后一天
tref_num = datenum(tref);
[tref_num,ia] = sort(tref_num);
tref = tref(ia);
x = x(ia,:);

sel_ind = tref_num>=datenum(2003,1,1);
tref_num = tref_num(sel_ind);
tref = tref(sel_ind);
x = x(sel_ind,:);

month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
month_cut_date1 = tref(month_cut(:,1));
month_cut_date2 = tref(month_cut(:,2));

y = cell2mat(x(month_cut(:,2),2));

y1 = smooth_s30(y,0.1);

y2 = whitsm(y,129600);

T = length(y);
figure
plot([y,y-y2,y2],'LineWidth',3)

set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
t_str = month_cut_date2(floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str);
set(gca,'XTickLabelRotation',90)
setpixelposition(gcf,[223,365,1345,420]);
legend({'PETTM','PE-cycle','PE-trend'},'NumColumns',3,'Location','best');
box off
title('贵州茅台PE的HP滤波分解-验证')
