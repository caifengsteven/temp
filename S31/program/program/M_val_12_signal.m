%{
具体地，若当日指数上涨，且成交量相比于前一日放大，则发出次日看涨信号。
%}
clear

p1 = 0.01;p2 = 1.1;
symbol = {'000016','399300','000905'};
symbol_info = {'上证50','沪深300','中正500'};
code_pool = {'510050','510300','510500'};
code_name_pool = {'etf50_min','etf300_min','etf500_min'};
code_indicator = {'IH','IF','IC'};

code_id = 3;
code_sel = code_pool{code_id};
code_indicator_sel = code_indicator{code_id};
symbol_sel = symbol{code_id};
% sql_str = ['select tradedate,openIndex,turnoverVol from yuqerdata.yq_index where ',...
%     'symbol = ''%s'' and tradedate>=''2014-05-05'' and tradedate<=''2019-12-31'' order by tradedate '];

sql_str = ['select tradedate,closeIndex/openIndex-1,turnoverVol from yuqerdata.yq_index where ',...
    'symbol = ''%s'' and tradedate>=''2014-05-05'' and tradedate<=''2019-12-31'' order by tradedate '];

% sql_str = ['select tradedate,closeIndex/openIndex-1,turnoverVol,chgpct from yuqerdata.yq_index where ',...
%     'symbol = ''%s'' and tradedate>=''2017-01-13'' and tradedate<=''2020-01-13'' order by tradedate '];

T_index = length(symbol);
%for i = 1:T_index

x = fetchmysql(sprintf(sql_str,symbol{code_id}),2);
%temp = cell2mat(x(:,2));
%x = [x(2:end,1),num2cell(temp(2:end)./temp(1:end-1)-1),x(2:end,3)];    
y = cell2mat(x(:,2:end));
vol_chg = y(2:end,2)./y(1:end-1,2);
y = y(2:end,:);
y(:,2) = vol_chg;
x = x(2:end,:);

ind = find(y(:,1)>=p1 & y(:,2)>=p2);
ind(ind>=size(y,1)) = [];
ind = ind + 1;
signal_val2 = zeros(size(x(:,1)));
signal_val2(ind) = 1;
tref = x(:,1);
%end
%backtest
%sql_str = 'select tradedate,openprice from yuqerdata.MktMFutdGet where contractObject = ''%s'' and mainCon=1 order by tradedate';
sql_str = 'select tradedate,openIndex from yuqerdata.yq_index where symbol = ''%s'' order by tradedate';
%r = fetchmysql(sprintf(sql_str,code_indicator_sel),2);
r = fetchmysql(sprintf(sql_str,symbol_sel),2);
r_v = cell2mat(r(:,2));
r = [r(2:end,1),num2cell(r_v(2:end)./r_v(1:end-1)-1)];

y_r = bac_testS31_indexfuture(tref,signal_val2,r);



y_c = cumprod(1+y_r);
figure
plot(y_c,'LineWidth',3)

t_str = tref;
T=length(t_str);
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));

set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)    
setpixelposition(gcf,[223,365,1345,420]);
legend(code_sel,'Location','best');
box off

[v,v_str,sta_val] = curve_static(y_c');

temp = [signal_val2(1:end-1),y_r(2:end)];
temp = temp(:,1).*temp(:,2);
y2 = zeros(size(temp));
y2(temp>0) = 1;
y2(temp<0) = -1;

figure
plot(cumsum(y2),'LineWidth',3);

