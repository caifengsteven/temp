%{
具体地，若当日指数上涨，且成交量相比于前一日放大，则发出次日看涨信号。
%}
clear

symbol = {'000016','399300','000905'};
symbol_info = {'上证50','沪深300','中正500'};
print_sel = false;
% sql_str = ['select tradedate,openIndex,turnoverVol from yuqerdata.yq_index where ',...
%     'symbol = ''%s'' and tradedate>=''2014-05-05'' and tradedate<=''2019-12-31'' order by tradedate '];

% sql_str = ['select tradedate,closeIndex/openIndex-1,turnoverVol from yuqerdata.yq_index where ',...
%     'symbol = ''%s'' and tradedate>=''2014-05-05'' and tradedate<=''2019-12-31'' order by tradedate '];

sql_str = ['select tradedate,closeIndex/openIndex-1,turnoverVol,chgpct from yuqerdata.yq_index where ',...
    'symbol = ''%s'' and tradedate>=''2014-05-05'' and tradedate<=''2019-12-31'' order by tradedate '];

T_index = length(symbol);
for i = 1:T_index
    symbol_sel = symbol{i};
    x = fetchmysql(sprintf(sql_str,symbol{i}),2);
    signal_val2 = zeros(size(x(:,1)));
    %temp = cell2mat(x(:,2));
    %x = [x(2:end,1),num2cell(temp(2:end)./temp(1:end-1)-1),x(2:end,3)];
    tref = x(:,1);
    y = cell2mat(x(:,2:end));
    
    T= size(x,1);
    yp = zeros(T,1);
    for j = 2:T-2
        if y(j,1)>0.008 && y(j,2)/y(j-1,2)>1.2
            signal_val2(j+1) = 1;
            if y(j+1,1)>0                
                yp(j+1)=1;
            elseif y(j+1,1)<0
                yp(j+1) = -1;
            end
            
        end
        if print_sel
            sprintf('%d-%d',j,T)
        end
    end
    %plot(cumsum(yp>0)./cumsum(signal_val2),'LineWidth',3)
    plot(cumsum(yp),'LineWidth',3)
    if eq(i,1)
        hold on
    end
    
end
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
t_str = x(:,1);
set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)    
setpixelposition(gcf,[223,365,1345,420]);
legend(symbol_info,'NumColumns',3,'Location','best');



%{
%backtest
%sql_str = 'select tradedate,openprice from yuqerdata.MktMFutdGet where contractObject = ''%s'' and mainCon=1 order by tradedate';
sql_str = 'select tradedate,openIndex from yuqerdata.yq_index where symbol = ''%s'' order by tradedate';
%r = fetchmysql(sprintf(sql_str,code_indicator_sel),2);
r = fetchmysql(sprintf(sql_str,symbol_sel),2);
r_v = cell2mat(r(:,2));
r = [r(2:end,1),num2cell(r_v(2:end)./r_v(1:end-1)-1)];

y_r = bac_testS31_indexfuture(tref,signal_val2,r);



y_c = cumprod(1+y_r);
figure
plot(y_c,'LineWidth',3)

t_str = tref;
T=length(t_str);
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));

set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)    
setpixelposition(gcf,[223,365,1345,420]);
legend(symbol_sel,'Location','best');
box off

[v,v_str,sta_val] = curve_static(y_c');

temp = [signal_val2(1:end-1),y_r(2:end)];
temp = temp(:,1).*temp(:,2);
y2 = zeros(size(temp));
y2(temp>0) = 1;
y2(temp<0) = -1;
plot(cumsum(y2>0)./cumsum(signal_val2(2:end)));

%}