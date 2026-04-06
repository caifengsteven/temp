%三方择时结果
clear

signal_str = containers.Map([-1,0,1],{'做空','平仓','做多'});

p1 = 0.01;p2 = 1.1;
window_p1 = 28;
%window_p1 = 50;
%window_p1 = 10;
print_sel = false;

sql_str1 = ['select (closeprice-iopv)/iopv,volume from ',...
    'S31.adj_data where date(tradingdate) = ''%s'' and symbol = ''%s'' order by tradingdate'];

tN = 'S31.adj_data';
symbol = {'000016','399300','000905'};
symbol_info = {'上证50','沪深300','中正500'};
code_pool = {'510050','510300','510500'};
%code_name_pool = {'etf50_min','etf300_min','etf500_min'};
code_indicator = {'IH','IF','IC'};

%code_id = 3;
re_sta = [];
y_c = cell(3,1);

h = figure('Units','normalized','Position',[0,0,1,1],'Name','S31easy择时三方信号综合回测结果');
movegui(h,'center');

for code_id = 1:3
    tref = yq_methods.get_tradingdate('2014-01-13',datestr(now,'yyyy-mm-dd'));
    tref_f = yq_methods.get_tradingdate_future(tref{end});
    tref_f = [tref;tref_f(2)];
    signal_final = zeros(length(tref_f),3);
    
    tref_num = datenum(tref_f);
    tref_week = weekday(tref_num);
    code_sel = code_pool{code_id};
    code_indicator_sel = code_indicator{code_id};
    symbol_sel = symbol{code_id};
    
    [tref1,signal_val1,tref_f1] = get_signal1_update(code_id);
    [~,ia,ib] = intersect(tref_f,tref_f1);
    signal_final(ia,1) = signal_val1(ib);
    %signal 2
    sql_str = ['select tradedate,closeIndex/openIndex-1,turnoverVol,chgpct from yuqerdata.yq_index where ',...
        'symbol = ''%s'' and tradedate>=''2014-01-13''   order by tradedate '];
    x = fetchmysql(sprintf(sql_str,symbol{code_id}),2);
    tref_f2 = yq_methods.get_tradingdate_future(x{1,end});
    tref_f2 = [x(:,1);tref_f2(2)];
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
    
    ind = find(y(:,1)>=p1 & y(:,2)>=p2);
    ind = ind + 1;
    sub_signal_val2 = zeros(size(x,1)+1,1);
    sub_signal_val2(ind) = 1;
    [~,ia,ib] = intersect(tref_f,tref_f2);
    signal_final(ia,2) = sub_signal_val2(ib);
    
    %signal 3
    signal_val3 = zeros(size(x,1)+1,1);
    signal_val3(eq(tref_week,2)) = 1;
    signal_val3(eq(tref_week,5)) = -1;
    [~,ia,ib] = intersect(tref_f,tref_f2);
    signal_final(ia,3) = signal_val3(ib);
    
    signal_val3 = signal_val3(1:end-1);
    temp = [signal_val2,signal_val3];
    
    temp1 = zeros(size(temp(:,1)));
    [~,ia,ib] = intersect(tref,tref1);
    temp1(ia) = signal_val1(ib);
    temp = [temp,temp1];
    
    signal_val = zeros(size(signal_val2));
    % signal_val(sum(abs(temp),2)>1 & sum(temp,2)>0) = 1;
    % signal_val(sum(abs(temp),2)>1 & sum(temp,2)<0) = -1;

    %signal_val = signal_val3;
    signal_val(sum(temp,2)>0) = 1;
    signal_val(sum(temp,2)<0) = -1;
    
    signal_valf = zeros(size(signal_final));
    signal_valf(sum(signal_final,2)>0) = 1;
    signal_valf(sum(signal_final,2)<0) = -1;
    sub_info = signal_str(signal_valf(end));
    %backtest
    %sql_str = 'select tradedate,openprice from yuqerdata.MktMFutdGet where contractObject = ''%s'' and mainCon=1 order by tradedate';
    %r = fetchmysql(sprintf(sql_str,code_indicator_sel),2);

    sql_str = 'select tradedate,closeIndex/openIndex-1 from yuqerdata.yq_index where symbol = ''%s'' order by tradedate';
    r = fetchmysql(sprintf(sql_str,symbol_sel),2);
    [~,ia,ib] = intersect(tref,r(:,1));
    
    temp = [signal_val(ia),cell2mat(r(ib,2))];
    temp = temp(:,1).*temp(:,2);
    y = zeros(size(temp));
    y(temp>0) = 1;
    y(temp<0) = -1;
    y_c{code_id} = cumprod(1+temp);
    
    temp = [signal_val2(ia),cell2mat(r(ib,2))];
    temp = temp(:,1).*temp(:,2);
    y2 = zeros(size(temp));
    y2(temp>0) = 1;
    y2(temp<0) = -1;
    
    temp = [signal_val3(ia),cell2mat(r(ib,2))];
    temp = temp(:,1).*temp(:,2);
    y3 = zeros(size(temp));
    y3(temp>0) = 1;
    y3(temp<0) = -1;
    
    [v1,v_str] = curve_static(y_c{code_id});
    %v2 = curve_static(y2,12);
    %v3 = curve_static(y3,12);
    re_sta = [re_sta;v1];
    subplot(3,1,code_id)
    plot(y_c{code_id},'LineWidth',2);
    t_str = tref;
    T=length(t_str);
    set(gca,'xlim',[0,T]);
    set(gca,'XTick',floor(linspace(1,T,15)));

    set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
    set(gca,'XTickLabelRotation',90)    
    %title(symbol_info{code_id});
    title(sprintf('%s-%s:%s',tref_f{end},symbol_info{code_id},sub_info))
    %setpixelposition(gcf,[223,365,1345,420]);
    box off
end


