clear
key_str = '修正风险溢价因子';
%计算修正步骤数据
[tref_add,Y_m] = Impliedvolatility_update();
%load sig_add.mat tref_add Y_m
tref_add_num = datenum(tref_add);
%载入曲线数据
sql_str = 'select tradingdate,f_val1,f_val2 from S24.return_data order by tradingdate';
tref = fetchmysql(sql_str,2);
tref_num = datenum(tref(:,1));
y_c = cumprod(1+cell2mat(tref(:,2:3)));
tref = tref(:,1);
%无风险收益率按照年华5%计
r_bond = exp(log(1.05)/244)-1;

%读入信号数据
x1 = fetchmysql(['select tradingdate,10y from aksharedata.bond_china_yield ',...
    'where 10y is not null and symbol=''中债国债收益率曲线'' order by tradingdate'],2);
x2 = fetchmysql(['select tradeDate,divYield*100 from yuqerdata.yq_MktIdxdEvalGet ',...
    'where ticker = ''000922'' and PEType=1 order by tradeDate'],2); 
[~,ia,ib] = intersect(x1(:,1),x2(:,1),'stable');
X = [x1(ia,:),x2(ib,end)];
tref_num_signal = datenum(X(:,1));
[~,ia,ib] = intersect(tref_num_signal,tref_add_num);
X = X(ia,:);
tref_num_signal = tref_num_signal(ia);
tref_add_num = tref_add_num(ib);
Y_m = Y_m(ib);

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
%修正中证红利股息率
X10(:,2) = X10(:,2).*Y_m;

%可视化
figure;
yyaxis right
sub_obj0=area(tref_num_signal,X10(:,2)-X10(:,1),'FaceAlpha',0.5,'EdgeColor','none','FaceColor',[0.7,0.7,0.7]);

yyaxis left
sub_obj=plot(tref_num_signal,X10,'-','LineWidth',2);
sub_obj(1).Color='r';

datetick('x')
legend({'十年期国债利率','修正中证红利指数股息率','修正风险溢价'},'NumColumns',3)
%signal > 0  bond or 
%计算信号1
signal_value = (X10(:,2)-X10(:,1));
%按照日期对齐
[tref,ia,ib] = intersect(tref,tref_signal);
tref_num = tref_num(ia);
y_c = y_c(ia,1);
y_r = [0;y_c(2:end)./y_c(1:end-1)-1];
%信号发出的下一日执行转换
signal_index = zeros(size(y_c)); %保存信号
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
[v,v_str,sta_val] = curve_static0(cumprod(1+r));

f_str = containers.Map([0,1],{'空','持有可转债'});
sprintf('%s-%s:%s',key_str,tref{end},f_str(signal_index(end)))
