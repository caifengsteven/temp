%参数敏感性检验
clear
code_id = 1;

re = zeros(90,6);
for window_p1 = 20:30
    print_sel = false;
    sql_str = ['select (closeprice-iopv)/iopv,volume from ',...
    'S31.adj_data where date(tradingdate) = ''%s'' and symbol = ''%s'' order by tradingdate'];

    tN = 'S31.adj_data';
    code_pool = {'510050','510300','510500'};
    code_name_pool = {'etf50_min','etf300_min','etf500_min'};
    code_indicator = {'IH','IF','IC'};

    tref = yq_methods.get_tradingdate('2017-01-13','2020-01-13');
    T_tref = length(tref);
    code_id = code_id;
    
    code_sel = code_pool{code_id};
    code_indicator_sel = code_indicator{code_id};
    %sub_t = zeros(245,1);
    %y = nan(245,T_tref);
    signal_val = zeros(T_tref,1);
    signal_val2 = signal_val;
    signal_val3 = signal_val;
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

        sub_x = fetchmysql(sprintf(sql_str,tref{i},code_sel));

        sub_x_open1 = mean(sub_x(:,1));
        sub_x_open2 = mean(sub_x(1:window_p1,1));
        sub_x_close = mean(sub_x(end-window_p1+1:end,1));

        sub_x1 = sub_x(1:end-1,:);
        sub_x2 = sub_x(2:end,:);

        sub_v1 = sum(sub_x2(sub_x2(:,1)>sub_x1(:,1),2));
        sub_v2 = sum(sub_x2(sub_x2(:,1)<sub_x1(:,1),2));

        if sub_x_close>sub_x_open1
            signal_val(i+1) = 1;
        elseif sub_x_close<sub_x_open1
            signal_val(i+1) = -1;
        else
            signal_val(i+1) = 0;
        end
        
        if sub_x_close>sub_x_open2
            signal_val2(i+1) = 1;
        elseif sub_x_close<sub_x_open2
            signal_val2(i+1) = -1;
        else
            signal_val2(i+1) = 0;
        end
        
        if sub_v1>sub_v2 && sub_x_close>max(sub_x_open1,sub_x_open2)
            signal_val3(i+1) = 1;
        elseif sub_v1<sub_v2 && sub_x_close<min(sub_x_open1,sub_x_open2)
            signal_val3(i+1) = -1;
        else
            signal_val3(i+1) = 0;
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


    y_r = bac_testS31_indexfuture(tref,signal_val,r);
    y_r2 = bac_testS31_indexfuture(tref,signal_val2,r);
    y_r3 = bac_testS31_indexfuture(tref,signal_val3,r);


    y_c = cumprod(1+y_r);
    y_c2 = cumprod(1+y_r2);
    y_c3 = cumprod(1+y_r3);

    [v,v_str,sta_val] = curve_static(y_c',[],false);
    [v2,v_str2,sta_val2] = curve_static(y_c2',[],false);
    [v3,v_str3,sta_val3] = curve_static(y_c3',[],false);
    
    sub_re1 = sta_val.nh/sta_val.std_return;
    sub_re2 = win_rate(signal_val,y_r);
    
    sub_re3 = sta_val2.nh/sta_val2.std_return;
    sub_re4 = win_rate(signal_val2,y_r2);
    
    sub_re5 = sta_val3.nh/sta_val3.std_return;    
    sub_re6 = win_rate(signal_val3,y_r3);
    
    re(window_p1,:) = [sub_re1,sub_re2,sub_re3,sub_re4,sub_re5,sub_re6];
    if eq(mod(window_p1,10),0)
        sprintf('%d-%d',window_p1,90)
    end
end

%%%%%%%%%%%
subplot(3,1,1)
yyaxis left
plot(re(:,1),'LineWidth',3);
set(gca,'YLim',[0,5])
yyaxis right 
plot(re(:,2),'LineWidth',3);
set(gca,'YLim',[0.4,0.6])
legend({'收益波动比','胜率'})
title('收盘-开盘信号')
subplot(3,1,2)
yyaxis left
plot(re(:,3),'LineWidth',3);
set(gca,'YLim',[0,5])
yyaxis right 
plot(re(:,4),'LineWidth',3);
set(gca,'YLim',[0.4,0.6])
legend({'收益波动比','胜率'})
title('收盘-日内平均信号')
subplot(3,1,3)
yyaxis left
plot(re(:,5),'LineWidth',3);
set(gca,'YLim',[0,5])
yyaxis right 
plot(re(:,6),'LineWidth',3);
set(gca,'YLim',[0.4,0.6])
legend({'收益波动比','胜率'})
title('综合信号')