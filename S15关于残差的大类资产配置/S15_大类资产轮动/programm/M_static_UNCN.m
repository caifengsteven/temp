%{
非协整情况
%}
clear
close all
load dataset1.mat
t1 = datenum(2002,1,1);
t2 = datenum(2016,6,30);
t3 = datenum(2014,1,1);
t4 = datenum(2011,4,15);

%日收益率
yield_v = zeros(size(x));
yield_v(2:end,:) = x(2:end,:)./x(1:end-1,:)-1;

ind = tref_num>=t4&tref_num<t2;

x = yield_v(ind,:);
x = x(:,1)-x(:,2);

% plot(cumprod(1+x)-1);
% sub_y = x;
% sub_x = tref(ind);
% sub_ind = floor(linspace(1,length(sub_y),20));
% set(gca,'xtick',sub_ind);
% set(gca,'xlim',[1,length(sub_y)]);
% set(gca,'XTickLabel',sub_x(sub_ind));
% set(gca,'XTickLabelRotation',45);

x1 = yield_v;
window = 120;
T = size(x1,1);
y = zeros(T,1);
for i = window:T
    sub_wid = i-window+1:i;
    sub_x = x1(sub_wid,:);
    %rab = sub_x;
    rab= cumprod(1+sub_x)-1;
    sub_p = polyfit(sub_x(:,2),sub_x(:,1),1);
    y(i) = (rab(end,1)-rab(end,2)*sub_p(1))/(std(sub_x(:,1)-sub_x(:,2)));
end
y_ma = movmean(y,[45,0]);
yyaxis left
plot(cumprod(1+x)-1,'r','LineWidth',2);
yyaxis right
plot(y(ind),'b','LineWidth',2);
hold on
plot(y_ma(ind),'-','LineWidth',2,'color',[0.470588235294118,0.670588235294118,0.188235294117647]);
sub_y = x;
sub_x = tref(ind);
sub_ind = floor(linspace(1,length(sub_y),20));
set(gca,'xtick',sub_ind);
set(gca,'xlim',[1,length(sub_y)]);
set(gca,'XTickLabel',sub_x(sub_ind));
set(gca,'XTickLabelRotation',45);