%{
综合择时
具体方法为：当三个子
信号同时看多时，综合信号看多；三个子信号同时看空时，综合信号看空；否则无信号
%}
clear

window_p1 = 10;
%window_p1 = 50;
%window_p1 = 10;
print_sel = false;

sql_str1 = ['select (closeprice-iopv)/iopv,volume from ',...
    'S31.adj_data where date(tradingdate) = ''%s'' and symbol = ''%s'' order by tradingdate'];

tN = 'S31.adj_data';
symbol = {'000016','399300','000905'};
symbol_info = {'上证50','沪深300','中正500'};
code_pool = {'510050','510300','510500'};
code_name_pool = {'etf50_min','etf300_min','etf500_min'};
code_indicator = {'IH','IF','IC'};

for code_id = 1:3
    tref = yq_methods.get_tradingdate('2017-01-13','2020-01-13');
    T_tref = length(tref);
    code_sel = code_pool{code_id};
    code_indicator_sel = code_indicator{code_id};
    symbol_sel = symbol{code_id};
    %sub_t = zeros(245,1);
    %y = nan(245,T_tref);
    signal_val = zeros(T_tref,1);
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
            signal_val(i+1) = 0;
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
    %signal_val = -signal_val;
    %backtest
    %sql_str = 'select tradedate,closeprice/openprice-1 from yuqerdata.MktMFutdGet where contractObject = ''%s'' and mainCon=1 order by tradedate';
    %r = fetchmysql(sprintf(sql_str,code_indicator_sel),2);
    
    sql_str = 'select tradedate,closeIndex/openIndex-1 from yuqerdata.yq_index where symbol = ''%s'' order by tradedate';
    r = fetchmysql(sprintf(sql_str,symbol_sel),2);
    
    [~,ia,ib] = intersect(tref,r(:,1));
    temp = [signal_val(ia),cell2mat(r(ib,2))];
    temp = temp(:,1).*temp(:,2);
    y = zeros(size(temp));
    y(temp>0) = 1;
    y(temp<0) = -1;
    plot(cumsum(y),'LineWidth',2);
    if eq(code_id,1)
        hold on
    end    
end 

T=length(tref);
t_str = tref;
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)    
setpixelposition(gcf,[223,365,1345,420]);
legend(symbol_info,'NumColumns',3,'Location','best');
