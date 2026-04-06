%{
综合择时
具体方法为：当三个子
信号同时看多时，综合信号看多；三个子信号同时看空时，综合信号看空；否则无信号
%}
clear

tn = 'S31.S31_signal';
var_info = {'signal_type','symbol','tradingdate','f_val'};
signal_type = 'P2_5';

signal_str = containers.Map([-1,0,1],{'做空','平仓','做多'});
window_p1 = 28;
%window_p1 = 50;
%window_p1 = 10;
print_sel = true;

sql_str1 = ['select (closeprice-iopv)/iopv,volume from ',...
    'S31.adj_data where date(tradingdate) = ''%s'' and symbol = ''%s'' order by tradingdate'];

tN = 'S31.adj_data';
code_pool = {'510050','510300','510500'};
code_indicator = {'IH','IF','IC'};

h = figure('Units','normalized','Position',[0,0,1,1],'Name','股指期货三因子择时');
movegui(h,'center');

sql_str_a1 =[ 'select tradingdate,f_val from S31.S31_signal where signal_type=''%s'' ',...
    'and symbol =''%s'' order by tradingdate'];

for code_id = 1:3
    tref = yq_methods.get_tradingdate('2017-01-13',datestr(now,'yyyy-mm-dd'));
    tref_f = yq_methods.get_tradingdate_future(tref{end});
    tref_f = [tref;tref_f(2)];
    T_tref = length(tref);
    code_sel = code_pool{code_id};
    code_indicator_sel = code_indicator{code_id};
    %sub_t = zeros(245,1);
    %y = nan(245,T_tref);
    signal_val = zeros(T_tref+1,1);
    
    x0 = fetchmysql(sprintf(sql_str_a1,signal_type,code_indicator_sel),2);
    if isempty(x0)
        t0 = datenum(1990,1,1);
        num0 = 1;
    else
        t0 = datenum(x0(end,1));
        num0 = find(eq(datenum(tref_f),t0));
        signal_val(1:num0) = cell2mat(x0(:,2));
    end
    
    sql_str_check =[ 'select ticker from yuqerdata.yq_MktMFutdGet ',...
        'where contractObject = ''%s'' and mainCon=1 and tradedate>=''%s'' and tradedate<=''%s'' ',...
        'order by tradedate';];
    sql_str_check2 =[ 'select tradedate,ticker from yuqerdata.yq_MktMFutdGet ',...
        'where contractObject = ''%s'' and mainCon=1 order by tradedate';];
    tickers = fetchmysql(sprintf(sql_str_check2,code_indicator_sel),2);       
    
    parfor i = num0:T_tref
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
    
    signal_re = [tref_f,tref_f,tref_f,num2cell(signal_val)];
    signal_re(:,1) = {signal_type};
    signal_re(:,2) = {code_indicator_sel};
    signal_re = signal_re(datenum(signal_re(:,3))>t0,:);
    if ~isempty(signal_re)
        datainsert_adair(tn,var_info,signal_re);
    end
    %signal_val = -signal_val;
    %backtest
    sub_info = signal_str(signal_val(end));
    sql_str = 'select tradedate,openprice from yuqerdata.yq_MktMFutdGet where contractObject = ''%s'' and mainCon=1 order by tradedate';
    r = fetchmysql(sprintf(sql_str,code_indicator_sel),2);
    r_v = cell2mat(r(:,2));
    r = [r(2:end,1),num2cell(r_v(2:end)./r_v(1:end-1)-1)];

    y_r = bac_testS31_indexfuture(tref,signal_val,r);

    y_c = cumprod(1+y_r);
    subplot(3,1,code_id)
    plot(y_c,'LineWidth',3)

    t_str = tref;
    T=length(t_str);
    set(gca,'xlim',[0,T]);
    set(gca,'XTick',floor(linspace(1,T,15)));

    set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
    set(gca,'XTickLabelRotation',90)    
    %setpixelposition(gcf,[223,365,1345,420]);
    legend(code_indicator_sel,'Location','best');
    box off
    title(sprintf('%s-%s:%s',tref{end},code_indicator_sel,sub_info))

    [v,v_str,sta_val] = curve_static(y_c');
end 
