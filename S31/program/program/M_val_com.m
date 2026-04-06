%综合信号
clear

p1 = 0.01;p2 = 1.1;
window_p1 = 28;
%window_p1 = 50;
%window_p1 = 10;
print_sel = false;

sql_str1 = ['select (closeprice-iopv)/iopv,volume from ',...
    'S31.adj_data where date(tradingdate) = ''%s'' and symbol = ''%s'' order by tradingdate'];

tN = 'S31.adj_data';
symbol = {'000016','399300','000905'};
code_pool = {'510050','510300','510500'};
code_name_pool = {'etf50_min','etf300_min','etf500_min'};
code_indicator = {'IH','IF','IC'};

code_id = 2;
%for code_id = 1%1:3
tref = yq_methods.get_tradingdate('2017-01-13','2020-01-13');
tref_num = datenum(tref);
tref_week = weekday(tref_num);
T_tref = length(tref);
code_sel = code_pool{code_id};
code_indicator_sel = code_indicator{code_id};
symbol_sel = symbol{code_id};
%sub_t = zeros(245,1);
%y = nan(245,T_tref);
signal_val1 = zeros(T_tref,1);

sql_str_check =[ 'select ticker from yuqerdata.MktMFutdGet ',...
    'where contractObject = ''%s'' and mainCon=1 and tradedate>=''%s'' and tradedate<=''%s'' ',...
    'order by tradedate';];
sql_str_check2 =[ 'select tradedate,ticker from yuqerdata.MktMFutdGet ',...
    'where contractObject = ''%s'' and mainCon=1 order by tradedate';];
tickers = fetchmysql(sprintf(sql_str_check2,code_indicator_sel),2);

parfor i = 1:T_tref-2
    %第二天是不是股指切换日期
    %sub_ticker = fetchmysql(sprintf(sql_str_check,code_indicator_sel,tref{i+1},tref{i+2}),2);
    sub_id = find(strcmp(tickers(:,1),tref(i)));
    sub_ticker = tickers(sub_id-1:sub_id,2);
    if ~strcmp(sub_ticker(1),sub_ticker(2))
        signal_val1(i+1) = 0;
        continue
    end

    sub_x = fetchmysql(sprintf(sql_str1,tref{i},code_sel));

    sub_x_open1 = mean(sub_x(:,1));
    sub_x_open2 = mean(sub_x(1:window_p1,1));
    sub_x_close = mean(sub_x(end-window_p1+1:end,1));

    sub_x1 = sub_x(1:end-1,:);
    sub_x2 = sub_x(2:end,:);

    sub_v1 = sum(sub_x2(sub_x2(:,1)>sub_x1(:,1),2));
    sub_v2 = sum(sub_x2(sub_x2(:,1)<sub_x1(:,1),2));

    if sub_v1>sub_v2 && sub_x_close>max(sub_x_open1,sub_x_open2)
        signal_val1(i+1) = 1;
    elseif sub_v1<sub_v2 && sub_x_close<min(sub_x_open1,sub_x_open2)
        signal_val1(i+1) = -1;
    else
        signal_val1(i+1) = 0;
    end
    if print_sel
        sprintf('%d-%d',i,T_tref)
    end
end
%signal 2
sql_str = ['select tradedate,closeIndex/openIndex-1,turnoverVol,chgpct from yuqerdata.yq_index where ',...
    'symbol = ''%s'' and tradedate>=''2017-01-13'' and tradedate<=''2020-01-13'' order by tradedate '];
x = fetchmysql(sprintf(sql_str,symbol{code_id}),2);
%temp = cell2mat(x(:,2));
%x = [x(2:end,1),num2cell(temp(2:end)./temp(1:end-1)-1),x(2:end,3)];    
y = cell2mat(x(:,2:end));
vol_chg = [0;y(2:end,2)./y(1:end-1,2)];
y(:,2) = vol_chg;

ind = find(y(:,1)>=p1 & y(:,2)>=p2);
ind(ind>=size(y,1)) = [];
ind = ind + 1;
signal_val2 = zeros(size(x(:,1)));
signal_val2(ind) = 1;
%signal 3
signal_val3 = zeros(size(x(:,1)));
signal_val3(eq(tref_week,6)) = 1;
signal_val3(eq(tref_week,2)) = -1;

temp = [signal_val1,signal_val2,signal_val3];
signal_val = zeros(size(signal_val1));
% signal_val(sum(abs(temp),2)>1 & sum(temp,2)>0) = 1;
% signal_val(sum(abs(temp),2)>1 & sum(temp,2)<0) = -1;

% signal_val(sum(temp,2)>0) = 1;
% signal_val(sum(temp,2)<0) = -1;

signal_val(sum(temp,2)>=2) = 1;
signal_val(sum(temp,2)<=-2) = -1;

%backtest
sql_str = 'select tradedate,openprice from yuqerdata.MktMFutdGet where contractObject = ''%s'' and mainCon=1 order by tradedate';
%sql_str = 'select tradedate,openIndex from yuqerdata.yq_index where symbol = ''%s'' order by tradedate';
r = fetchmysql(sprintf(sql_str,code_indicator_sel),2);
%r = fetchmysql(sprintf(sql_str,symbol_sel),2);
r_v = cell2mat(r(:,2));
r = [r(2:end,1),num2cell(r_v(2:end)./r_v(1:end-1)-1)];

y_r = bac_testS31_indexfuture(tref,signal_val,r);
y_r1 = bac_testS31_indexfuture(tref,signal_val1,r);

y_c = cumprod(1+y_r);
y_c1 = cumprod(1+y_r1);
figure
plot([y_c,y_c1],'LineWidth',3)

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