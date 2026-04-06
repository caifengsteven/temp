clear
sta_re = [];
fee_pool = [0,1,2,3,5,8,10]/10000;
fee_title = {'无','万一','万二','万三','万五','万八','千一'};

window_p1 = 38;
print_sel = false;

sql_str = ['select (closeprice-iopv)/iopv,volume from ',...
    'S31.adj_data where date(tradingdate) = ''%s'' and symbol = ''%s'' order by tradingdate'];

tN = 'S31.adj_data';
code_pool = {'510050','510300','510500'};
code_name_pool = {'etf50_min','etf300_min','etf500_min'};
var_info = {'symbol','tradingdate','iopv','openprice','closeprice','volume'};

tref = yq_methods.get_tradingdate('2017-01-13','2020-01-13');
T_tref = length(tref);

code_sel = code_pool{1};
%sub_t = zeros(245,1);
%y = nan(245,T_tref);
signal_val = zeros(T_tref,1);
parfor i = 1:T_tref-1
 
    sub_x = fetchmysql(sprintf(sql_str,tref{i},code_sel));
    
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
sql_str = 'select tradedate,openprice*accumAdjFactor from yuqerdata.MktFunddGet where ticker = ''%s''  order by tradedate';
r = fetchmysql(sprintf(sql_str,code_sel),2);
temp_r_v = cell2mat(r(:,2:end));
r = [r(2:end,1),num2cell(temp_r_v(2:end,1)./temp_r_v(1:end-1,1)-1)];


y = zeros(length(tref),length(fee_pool));
for i = 1:length(fee_pool)
    sub_fee = fee_pool(i);
    y_r = bac_testS31_etf(tref,signal_val,r,sub_fee);
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