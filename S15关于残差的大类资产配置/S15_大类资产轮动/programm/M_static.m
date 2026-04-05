%{
M_Static
%}
clear
load dataset1.mat
t1 = datenum(2002,1,1);
t2 = datenum(2016,6,30);
t3 = datenum(2014,1,1);


%图3 日收益率相关性
yield_v = zeros(size(x));
yield_v(2:end,:) = x(2:end,:)./x(1:end-1,:)-1;

window = 40;
T = size(x,1);
r_v = zeros(T,1);

for i = window+1:T
    r_v(i) = corr(yield_v(i-window:i,1),yield_v(i-window:i,2));
end

ind = tref_num>=t1&tref_num<t2;


sub_y = r_v(ind);
sub_x = tref(ind);
sub_ind = floor(linspace(1,length(sub_y),20));
subplot(1,2,1)
plot(sub_y,'r','LineWidth',2);
set(gca,'xtick',sub_ind);
set(gca,'xlim',[1,length(sub_y)]);
set(gca,'XTickLabel',sub_x(sub_ind));
set(gca,'XTickLabelRotation',45);
subplot(1,2,2)
%价格走势
yyaxis left
plot(x(ind,1),'LineWidth',2)
yyaxis right
plot(x(ind,2),'LineWidth',2);

set(gca,'xtick',sub_ind);
set(gca,'xlim',[1,length(sub_y)]);
set(gca,'XTickLabel',sub_x(sub_ind));
set(gca,'XTickLabelRotation',45);

ind2 = tref_num>=datenum(2013,1,1)&tref_num<t2;
figure
subplot(1,2,1)
plot(x(ind2,1)./x(ind2,2),'LineWidth',2)
sub_y = x(ind2,1);
sub_x = tref(ind2);
sub_ind = floor(linspace(1,length(sub_y),20));
set(gca,'xtick',sub_ind);
set(gca,'xlim',[1,length(sub_y)]);
set(gca,'XTickLabel',sub_x(sub_ind));
set(gca,'XTickLabelRotation',45);
ind3 = tref_num>=datenum(2014,7,1)&tref_num<t2;
%ind3 = ind2;
subplot(1,2,2)
plot(yield_v(ind3,1),yield_v(ind3,2),'+')
lims = axis;
temp_v1 = max(abs(lims(1:2)));
temp_v2 = max(abs(lims(3:4)));
hold on
plot([0,0],[-temp_v2,temp_v2])
plot([-temp_v1,temp_v1],[0,0])





