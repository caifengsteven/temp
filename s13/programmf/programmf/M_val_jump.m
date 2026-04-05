%验证指数跳价收益
clear
close all
index_name_pool = {'沪深300','上证指数','上证50','中证500','深证成指',...
    '创业板指','中小板指','中证1000',...
        '深次新股','中证流通'};
sub_index_name1 = index_name_pool{2};
[y_year,y_month,u_ind_month,u_ind_year] = get_data(sub_index_name1);
sub_index_name2 = index_name_pool{5};
[y_year2,y_month2,u_ind_month2,u_ind_year2] = get_data(sub_index_name2);

figure
bar(u_ind_year,[y_year,y_year2]);
legend({sub_index_name1,sub_index_name2})
set(gca,'XTickLabelRotation',90);
set(gca,'XTick',u_ind_year,'xlim',u_ind_year([1,end])+[-1;1]);
set(gca,'fontsize',12);
val = get(gca,'YTick');
labels = num2str(val'*100,'%5.1f%%');
% Adjust labels on plot
set(gca,'yticklabel',labels)
setpixelposition(gcf,[416,397,961,420]);

figure
obj=plot(u_ind_month,[y_month,y_month2],'LineWidth',2);
hold on
plot(u_ind_month([1,end])+[-1;1],[0;0],'k-','LineWidth',2);
legend(obj,{sub_index_name1,sub_index_name2},'NumColumns',2,'location','best')
set(gca,'XTickLabelRotation',90);
set(gca,'XTick',u_ind_month(floor(linspace(1,length(u_ind_month),length(y_year)))),'xlim',u_ind_month([1,end])+[-1;1]);
datetick('x','yyyymmdd','keepticks');

set(gca,'fontsize',12);
val = get(gca,'YTick');
labels = num2str(val'*100,'%5.1f%%');
% Adjust labels on plot
set(gca,'yticklabel',labels)
setpixelposition(gcf,[416,397,961,420]);
box off

function [y_year,y_month,u_ind_month,u_ind_year] = get_data(sub_index_name)
%指数数据
t0 = '1995-01-01';
tt = '2019-03-29';
index_data1 = get_index_data_ycz(sub_index_name,t0);
index_data2 = get_index_data_JJ(sub_index_name,index_data1{end,1},tt);
index_data = [index_data1;index_data2(2:end,:)];

tref_str = index_data(:,1);
tref = datenum(tref_str);
o_c_price = cell2mat(index_data(:,2:3));
open_price = o_c_price(:,1);
close_price = o_c_price(:,2);
%g_cum; g_jump g_inner %累计收益，跳价收益，日内收益
%几何收益率
g_cum = [0;log(close_price(2:end)./close_price(1:end-1))];
g_jump = [0;log(open_price(2:end)./close_price(1:end-1))];
g_inner = log(close_price./open_price);
g_inner(1) = 0;

% g_info = {'累计对数收益','累计日内收益','累计跳价收益'};
g_data_geo = [g_cum,g_inner,g_jump];
% %算术收益率
% g_cum_m = [0;close_price(2:end)./close_price(1:end-1)-1];
% g_jump_m = [0;open_price(2:end)./close_price(1:end-1)-1];
% g_inner_m = close_price./open_price-1;
% g_inner_m(1) = 0;
% g_data_math = [g_cum_m,g_inner_m,g_jump_m];
% 
% obj = zeros(3,1);
% colors = [0.64,0.78,0.18;0.93,0.69,0.13;ones(1,3)*0.65];
% obj(1)= plot(tref,cumsum(g_cum),'-','linewidth',2,'color','r');
% hold on
% obj(2)= plot(tref,cumsum(g_inner),'-','linewidth',2,'color',colors(2,:));
% obj(3)= plot(tref,cumsum(g_jump),'-','linewidth',2,'color',colors(3,:));
% %obj(1)= plot(tref,cumsum([g_cum,g_inner,g_jump]),'linewidth',2);
% plot(tref([1,end]),[0,0],'k-','LineWidth',2);
% 
% set(gca,'XTickLabelRotation',90);
% set(gca,'XTick',tref(floor(linspace(1,length(tref),40))),'xlim',tref([1,end]));
% datetick('x','yyyymmdd','keepticks');
% set(gca,'fontsize',12);
% 
% box off
% set(gca,'linewidth',1.5);
% legend(obj,g_info,'Location','northwest',...
%     'NumColumns',length(obj),'location','best')
% legend('boxoff')
% 
% val = get(gca,'YTick');
% labels = num2str(val'*100,'%5.1f%%');
% % Adjust labels on plot
% set(gca,'yticklabel',labels)
% title(sub_index_name)
% setpixelposition(gcf,[416,397,961,420]);
% 
% %[v,v_str] = curve_static(-cumprod(1+g_jump_m))
% V = zeros(3,3);
% for i = 1:3
%     %sprintf('%s',g_info{i})
%     [V(:,i),v_str] = cal_para_geo(cumsum(g_data_geo(:,i)),tref(end)-tref(1)+1);
% end
% 
% f_str = '%s结果\n\t\t\t 年化收益率\t\t Sharp值 \t\t年化波动率\n';
% for i = 1:3
%     f_str = [f_str,g_info{i},'\t%0.3f \t\t %0.3f \t\t %0.3f \n'];
% end
% 
% sprintf(f_str,sub_index_name,V)

ind_year = year(tref);
ind_month = ind_year*100+month(tref);
u_ind_year = unique(ind_year);
y_year = zeros(size(u_ind_year));
for i = 1:length(u_ind_year)
    temp = g_jump(eq(ind_year,u_ind_year(i)));
    y_year(i) = sum(temp);
end
[u_ind_month,ia] = unique(ind_month);
y_month = zeros(size(u_ind_month));
for i = 1:length(u_ind_month)
    temp = g_jump(eq(ind_month,u_ind_month(i)));
    y_month(i) = sum(temp);
end
u_ind_month = tref(ia);
end
