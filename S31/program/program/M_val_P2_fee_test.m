%{
综合择时
具体方法为：当三个子
信号同时看多时，综合信号看多；三个子信号同时看空时，综合信号看空；否则无信号
%}
clear

sta_re = [];
fee_pool = [0,1,2,3,5,8,10]/10000;
fee_title = {'无','万一','万二','万三','万五','万八','千一'};

window_p1 = 28;
print_sel = false;


sql_str = ['select (closeprice-iopv)/iopv,volume from ',...
    'S31.adj_data where date(tradingdate) = ''%s'' and symbol = ''%s'' order by tradingdate'];

tN = 'S31.adj_data';
code_pool = {'510050','510300','510500'};
code_name_pool = {'etf50_min','etf300_min','etf500_min'};
code_indicator = {'IH','IF','IC'};

tref = yq_methods.get_tradingdate('2017-01-13','2020-01-13');
T_tref = length(tref);

code_sel = code_pool{1};
code_indicator_sel = code_indicator{1};
%sub_t = zeros(245,1);
%y = nan(245,T_tref);
signal_val = zeros(T_tref,1);
sql_str_check =[ 'select ticker from yuqerdata.MktMFutdGet ',...
    'where contractObject = ''%s'' and mainCon=1 and tradedate>=''%s'' and tradedate<=''%s'' ',...
    'order by tradedate';];

parfor i = 1:T_tref-2
    %第二天是不是股指切换日期
    sub_ticker = fetchmysql(sprintf(sql_str_check,code_indicator_sel,tref{i+1},tref{i+2}),2);
    if ~strcmp(sub_ticker(1),sub_ticker(2))
        signal_val(i+1) = 0;
        continue
    end
    
    sub_x = fetchmysql(sprintf(sql_str,tref{i},code_sel));
    
    sub_x_open1 = mean(sub_x(:,1));
    sub_x_open2 = mean(sub_x(1:window_p1,1));
    sub_x_close = mean(sub_x(end-window_p1+1:end,1));
    
    sub_x1 = sub_x(1:end-1,:);
    sub_x2 = sub_x(2:end,:);
    
    sub_v1 = sum(sub_x2(sub_x2(:,1)>sub_x1(:,1),2));
    sub_v2 = sum(sub_x2(sub_x2(:,1)<sub_x1(:,1),2));

    if sub_v1>sub_v2 && sub_x_close>max(sub_x_open1,sub_x_open2)
        signal_val(i+1) = 1;
    elseif sub_v1<sub_v2 && sub_x_close<min(sub_x_open1,sub_x_open2)
        signal_val(i+1) = -1;
    else
        signal_val(i+1) = 0;
    end
    if print_sel
        sprintf('%d-%d',i,T_tref)
    end
end
%backtest
sql_str = 'select tradedate,openprice from yuqerdata.MktMFutdGet where contractObject = ''%s'' and mainCon=1  order by tradedate';
r = fetchmysql(sprintf(sql_str,code_indicator_sel),2);
r_v = cell2mat(r(:,2));
r = [r(2:end,1),num2cell(r_v(2:end)./r_v(1:end-1)-1)];

y = zeros(length(tref),length(fee_pool));
for i = 1:length(fee_pool)
    sub_fee = fee_pool(i);
    %y_r = bac_testS31_etf(tref,signal_val,r,sub_fee);
    y_r = bac_testS31_indexfuture(tref,signal_val,r,sub_fee);
    y_c =cumprod(1+y_r);
    [v,v_str,sta_val] = curve_static(y_c',[],false);
    y(:,i) = y_c;
    if eq(i,1)
        sta_re = v_str;
    end
    sta_re = cat(1,sta_re,num2cell(v));
end

%%%%%%%%%%%
plot(y,'LineWidth',2)
t_str = tref;
T=length(t_str);
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));

set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)    
setpixelposition(gcf,[223,365,1345,420]);
legend(fee_title,'Location','best');
box off

sta_re = [[{'' };fee_title'],sta_re];

