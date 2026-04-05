clear

%载入曲线数据
load bac_curve.mat
%无风险收益率按照年华5%计
r_bond = exp(log(1.05)/244)-1;

%读入信号数据
[~,~,x1] = xlsread('tenYear_bond.xls');%yuqer数据
[~,~,x2] = xlsread('股息2008.xlsx');%wind数据
x2 = x2(2:end,1:2);
x1 = x1(2:end,1:2);
[~,ia,ib] = intersect(x1(:,1),x2(:,1));
X = [x1(ia,:),x2(ib,end)];
tref_num_signal = datenum(X(:,1));
[tref_num_signal,ia ] = sort(tref_num_signal);
X = X(ia,:);
tref_signal = X(:,1);
tref_signal = cellstr(datestr(datenum(tref_signal),'yyyy-mm-dd'));
X = cell2mat(X(:,2:end));

%信号平滑
X20 = movmean(X,[20,0]);
X10 = movmean(X,[10,0]);
X5 = movmean(X,[5,0]);
%{
subplot(3,1,1);plot(X5);
subplot(3,1,2);plot(X10);
subplot(3,1,3);plot(X20);
%}
%可视化
figure;
yyaxis right
sub_obj0=area(tref_num_signal,X10(:,2)-X10(:,1),'FaceAlpha',0.5,'EdgeColor','none','FaceColor',[0.7,0.7,0.7]);

yyaxis left
sub_obj=plot(tref_num_signal,X10,'-','LineWidth',2);
sub_obj(1).Color='r';

datetick('x')
legend({'十年期国债利率','中证红利指数股息率','风险溢价'},'NumColumns',3)
%signal > 0  bond or 
%计算信号1
signal_value = X10(:,2)-X10(:,1);
%按照日期对齐
[~,ia,ib] = intersect(tref,tref_signal);
tref_num = tref_num(ia);
y_c = y_c(ia,1);
y_r = [0;y_c(2:end)./y_c(1:end-1)-1];
%信号发出的下一日执行转换
signal_index = zeros(size(y_c));
temp = find(signal_value>0)+1;
temp(temp>length(y_c)) = [];
signal_index(temp) = 1;

%合成曲线
r = zeros(size(signal_index));
r(eq(signal_index,1)) = y_r(eq(signal_index,1));
r(eq(signal_index,0)) = r_bond;
figure;plot(tref_num,cumprod(1+r),'LineWidth',2)
hold on
plot(tref_num,y_c/y_c(1),'LineWidth',2)
datetick('x','yyyy')

%统计参数
[v,v_str,sta_val] = curve_static0(cumprod(1+r))


