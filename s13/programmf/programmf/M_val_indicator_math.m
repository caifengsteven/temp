
%使用算术收益率计算
clear
index_name_pool = {'沪深300','创业板指','中小板指','中证1000',...
        '上证50','深次新股','中证流通','深证成指'};
sub_index_name = index_name_pool{1};
%指数数据
t0 = '2005-01-01';
index_data1 = get_index_data_ycz(sub_index_name,t0);
index_data2 = get_index_data_JJ(sub_index_name,index_data1{end,1});
index_data = [index_data1;index_data2];

tref_str = index_data(:,1);
tref = datenum(tref_str);
o_c_price = cell2mat(index_data(:,2:3));
open_price = o_c_price(:,1);
close_price = o_c_price(:,2);
%g_cum; g_jump g_inner %累计收益，跳价收益，日内收益
g_cum = [0;close_price(2:end)./close_price(1:end-1)-1];
g_jump = [0;open_price(2:end)./close_price(1:end-1)-1];
g_inner = close_price./open_price-1;
g_inner(1) = 0;
plot(cumprod(1+[g_cum,g_jump,g_inner]))


