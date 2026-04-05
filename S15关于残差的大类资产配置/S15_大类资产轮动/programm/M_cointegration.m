%{
协整研究
%}

clear
close all

mod =2;%1价格 2log价格 3收益率
window = 120; %设置计算窗口
load dataset1.mat
t1 = datenum(2002,1,1);
t2 = datenum(2016,6,30);
t3 = datenum(2014,1,1);
t4 = datenum(2005,2,1);
t5 = datenum(2010,2,1);
%日收益率
yield_v = zeros(size(x));
yield_v(2:end,:) = x(2:end,:)./x(1:end-1,:)-1;

T = size(x,1);
y = zeros(T,1);
if eq(mod,1)
    x_sel = x;
elseif eq(mod,2)
    x_sel = log(x);
elseif eq(mod,3)
    x_sel = yield_v;
else
    keyboard;
end
parfor i = window:T
    sub_x = x_sel(i-window+1:i,:);
    [~,pValue] = egcitest(sub_x);
    y(i) = pValue;
end

ind = find(tref_num>=t5&tref_num<=t2);
ind2 = find(y(ind)<=0.05);
subplot(2,1,1)
plot(y(ind));
hold on 
plot(ind2,y(ind(ind2)),'+');
sub_y = y(ind);
sub_x = tref(ind);
sub_ind = floor(linspace(1,length(sub_y),20));
set(gca,'xtick',sub_ind);
set(gca,'xlim',[1,length(sub_y)]);
set(gca,'XTickLabel',sub_x(sub_ind));
set(gca,'XTickLabelRotation',45);

subplot(2,1,2)
plot(sort(y(ind)));
