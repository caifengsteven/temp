%{
综合择时
特色指标与价量指标后，我们将两者进行结合，构建综合指标，具体方法为：
当两个指标同时发出看多信号时，综合指标发出看多信号，否则综合指标无信号。其中，
特色指标以“收盘归宿与日内平均”为例
%}
clear

window_p1 = 38;
print_sel = false;

sql_str1 = ['select (closeprice-iopv)/iopv,volume from ',...
    'S31.adj_data where date(tradingdate) = ''%s'' and symbol = ''%s'' order by tradingdate'];

tN = 'S31.adj_data';
code_pool = {'510050','510300','510500'};
code_name_pool = {'etf50_min','etf300_min','etf500_min'};
var_info = {'symbol','tradingdate','iopv','openprice','closeprice','volume'};

tref = yq_methods.get_tradingdate('2017-01-13','2020-01-13');
T_tref = length(tref);

for code_id = 1:3
    code_sel = code_pool{code_id};
    %sub_t = zeros(245,1);
    %y = nan(245,T_tref);
    signal_val = zeros(T_tref,1);
    parfor i = 1:T_tref-1

        sub_x = fetchmysql(sprintf(sql_str1,tref{i},code_sel));

        sub_x_open = mean(sub_x(:,1));
        sub_x_close = mean(sub_x(end-window_p1+1:end,1));

        sub_x1 = sub_x(1:end-1,:);
        sub_x2 = sub_x(2:end,:);

        sub_v1 = sum(sub_x2(sub_x2(:,1)>sub_x1(:,1),2));
        sub_v2 = sum(sub_x2(sub_x2(:,1)<sub_x1(:,1),2));

        if sub_v1>sub_v2 && sub_x_close>sub_x_open
            signal_val(i+1) = 1;
        else
            signal_val(i+1) = 0;
        end
        if print_sel
            sprintf('%d-%d',i,T_tref)
        end
    end
    %backtest
    sql_str = 'select tradedate,openprice*accumAdjFactor from yuqerdata.MktFunddGet where ticker = ''%s'' order by tradedate';
    r = fetchmysql(sprintf(sql_str,code_sel),2);
    temp_r_v = cell2mat(r(:,2:end));
    r = [r(2:end,1),num2cell(temp_r_v(2:end,1)./temp_r_v(1:end-1,1)-1)];

    y_r = bac_testS31_etf(tref,signal_val,r);

    y_c =cumprod(1+y_r);
    
    subplot(3,1,code_id)
    plot(y_c,'LineWidth',3)

    t_str = tref;
    T=length(t_str);
    set(gca,'xlim',[0,T]);
    set(gca,'XTick',floor(linspace(1,T,15)));

    set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
    set(gca,'XTickLabelRotation',90)    
    %setpixelposition(gcf,[223,365,1345,420]);
    legend(code_sel,'Location','best');
    box off

    [v,v_str,sta_val] = curve_static(y_c');
end
