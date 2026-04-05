clear
code_id = 1;
print_sel = false;
re = zeros(90,4);
for window_p1 = 1:90
    print_sel = false;
    sql_str = ['select (closeprice-iopv)/iopv from ',...
        'S31.adj_data where date(tradingdate) = ''%s'' and symbol = ''%s'' order by tradingdate'];

    tN = 'S31.adj_data';
    code_pool = {'510050','510300','510500'};
    code_name_pool = {'etf50_min','etf300_min','etf500_min'};
    var_info = {'symbol','tradingdate','iopv','openprice','closeprice','volume'};

    tref = yq_methods.get_tradingdate('2017-01-13','2020-01-13');
    T_tref = length(tref);

    code_sel = code_pool{code_id};
    %sub_t = zeros(245,1);
    %y = nan(245,T_tref);
    signal_val = zeros(T_tref,1);
    signal_val2 = signal_val;
    parfor i = 1:T_tref-1

        sub_x = fetchmysql(sprintf(sql_str,tref{i},code_sel));
        %sub_x = [sub_x;1531,0]
        sub_x_open = mean(sub_x(1:window_p1));
        sub_x_open2 = mean(sub_x);
        sub_x_close = mean(sub_x(end-window_p1+1:end));
        if sub_x_close>sub_x_open
            signal_val(i+1) = 1;
        else
            signal_val(i+1) = 0;
        end
        if sub_x_close>sub_x_open2
            signal_val2(i+1) = 1;
        else
            signal_val2(i+1) = 0;
        end
        if print_sel
            sprintf('%d-%d',i,T_tref)
        end
    end
    %backtest
    % sql_str = 'select tradedate,closeprice/openprice-1,CHGPct from yuqerdata.MktFunddGet where ticker = ''%s''';
    % r = fetchmysql(sprintf(sql_str,code_sel),2);
    sql_str = 'select tradedate,openprice*accumAdjFactor from yuqerdata.MktFunddGet where ticker = ''%s''  order by tradedate';
    r = fetchmysql(sprintf(sql_str,code_sel),2);
    temp_r_v = cell2mat(r(:,2:end));
    r = [r(2:end,1),num2cell(temp_r_v(2:end,1)./temp_r_v(1:end-1,1)-1)];

    y_r = bac_testS31_etf(tref,signal_val,r);
    y_r2 = bac_testS31_etf(tref,signal_val2,r);

    y_c = cumprod(1+y_r);
    y_c2 = cumprod(1+y_r2);

    [v,v_str,sta_val] = curve_static(y_c',[],false);
    [v2,v_str2,sta_val2] = curve_static(y_c2',[],false);
    
    sub_re1 = sta_val.nh/sta_val.std_return;
    sub_re2 = sum(y_r>0)/sum(signal_val);
    sub_re3 = sta_val2.nh/sta_val2.std_return;
    sub_re4 = sum(y_r2>0)/sum(signal_val2);
    
    re(window_p1,:) = [sub_re1,sub_re2,sub_re3,sub_re4];
    
    if eq(mod(window_p1,10),0)
        sprintf('%d-%d',window_p1,90)
    end
end

%%%%%%%%%%%
subplot(1,2,1)
yyaxis left
plot(re(:,1),'LineWidth',3);
set(gca,'YLim',[0,5])
yyaxis right 
plot(re(:,2),'LineWidth',3);
set(gca,'YLim',[0.35,0.6])
legend({'收益波动比','胜率'})
title('收盘-开盘信号')
subplot(1,2,2)
yyaxis left
plot(re(:,3),'LineWidth',3);
set(gca,'YLim',[0,5])
yyaxis right 
plot(re(:,4),'LineWidth',3);
set(gca,'YLim',[0.35,0.6])
legend({'收益波动比','胜率'})
title('收盘-日内平均信号')