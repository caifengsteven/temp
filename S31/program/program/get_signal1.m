function [tref,signal_val] = get_signal1(code_id)
window_p1 = 28;
%window_p1 = 50;
%window_p1 = 10;
print_sel = false;

sql_str1 = ['select (closeprice-iopv)/iopv,volume from ',...
    'S31.adj_data where date(tradingdate) = ''%s'' and symbol = ''%s'' order by tradingdate'];

tN = 'S31.adj_data';
code_pool = {'510050','510300','510500'};
code_name_pool = {'etf50_min','etf300_min','etf500_min'};
code_indicator = {'IH','IF','IC'};


tref = yq_methods.get_tradingdate('2017-01-13','2020-01-13');
T_tref = length(tref);
code_sel = code_pool{code_id};
code_indicator_sel = code_indicator{code_id};
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
