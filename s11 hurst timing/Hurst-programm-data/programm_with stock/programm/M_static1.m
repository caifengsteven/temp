%M_static1
%均值 标准差 偏度 峰度 J-B检验 J-B检验临界值
warning('off');
clear
t0 = datenum(2007,6,9);
tt = datenum(2017,6,9);

sub_data_info = {'上证指数','深证成指'};
sub_data_info = sub_data_info{2};
sql_str = ['select tradingdate,open,close from futuredata.indicator_data ',10,...
    'where symbolname = ''%s'' and tradingdate>= ''%s'' and tradingdate<= ''%s'' order by tradingdate'];
x = fetchmysql(sprintf(sql_str,sub_data_info,datestr(t0,'yyyy-mm-dd'),datestr(tt,'yyyy-mm-dd')),2);

tref = datenum(x(:,1));
open_price = cell2mat(x(:,2));
close_price = cell2mat(x(:,3));
[tref_w,open_price_w,close_price_w] = get_week_data(tref,open_price,close_price);

r1 = close_price(2:end)./close_price(1:end-1)-1;
r1_w = close_price_w(2:end)./close_price_w(1:end-1)-1;
%r1(abs(r1)*100>9) = [];
%r1 = close_price./open_price-1;
sta_re = get_sta_values(r1,sub_data_info);

sta_re_w = get_sta_values(r1_w,[sub_data_info,'-周']);


ah1=subplot(1,2,1);
m_histfigure(r1,ah1);

ah2=subplot(1,2,2);
m_histfigure(r1_w,ah2);

% %
% T = length(r1);
% sub_x = zeros(T,1);
% sub_y = sub_x;
% for i = 5:T
%     sub_y(i) = std(r1(1:i));
%     sub_x(i) = i;
% end
% figure
% plot(log(sub_x(5:end)),log(sub_y(5:end)),'+');

%获取统计值
function sta_re = get_sta_values(r1,sub_data_info)
[~,~,jbstat,critval] = jbtest(r1);
sta_re = [mean(r1),std(r1),skewness(r1),kurtosis(r1),jbstat,critval];
fprintf('%s \t均值 \t\t 标准差 \t 偏度 \t 峰度 \t J-B检验 \t J-B检验临界值\n','指数名称')
fprintf('%s \t%0.4e \t%0.4f \t%0.4f \t%0.4f \t%0.4f \t%0.4f \n',sub_data_info,sta_re)
end


%获取周数据
function [tref_w,p_open_w,p_close_w] = get_week_data(tref,p_open,p_close)
week_num = weeknum(tref);
ind = find(diff(week_num));
ind = [0;ind;length(tref)];

ind = [ind(1:end-1)+1,ind(2:end)];
p_open_w = p_open(ind(:,1));
p_close_w = p_close(ind(:,2));
tref_w = tref(ind(:,2));
end

%作图

function m_histfigure(r1,ah)
histogram(r1,'Normalization','pdf','parent',ah);

[mu,sigma] = normfit(r1);
d=pdf('norm',r1,mu,sigma);
hold on
[a,ia] = sort(r1);
e = d(ia);
plot(ah,a,e,'b','LineWidth',2)
[f,xi] = ksdensity(r1);
plot(ah,xi,f,'r','LineWidth',2);
end