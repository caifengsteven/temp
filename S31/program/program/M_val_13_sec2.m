%周内效应
clear
symbol = {'000016','399300','000905'};
symbol_info = {'上证50','沪深300','中正500'};

sql_str = ['select tradedate,weekday(tradedate),chgPct from yuqerdata.yq_index where ',...
    'symbol = ''%s'' and tradedate>=''2014-05-05'' and tradedate<=''2019-12-31'' order by tradedate '];

re = cell(3,1);
for i = 1:3
    sub_x = fetchmysql(sprintf(sql_str,symbol{i}),2);
    sub_y = cell2mat(sub_x(:,2:end));
    %wub_y = [weekday(datenum(sub_x(:,2))),cell2mat(sub_x(:,3))];
    sub_y0 = sub_y(eq(sub_y(:,1),0)|eq(sub_y(:,1),3),:);
    
    y =sub_y0(:,1);
    y(eq(sub_y0(:,1),0)) = 1;
    y(eq(sub_y0(:,1),3)) = -1;
    
    yp = y;
    yp(sub_y0(:,2)>0) = 1;
    yp(sub_y0(:,2)<0) = -1;
    %k = sum(eq(yp,y))
    %k = sum(~eq(yp,y))
    y2 = y;
    y2(eq(yp,y)) = 1;
    y2(~eq(yp,y)) = -1;
    plot(cumsum(y2),'LineWidth',3)
    if eq(i,1)        
        hold on
    end
    
end

t_str = sub_x(~eq(y,0),1);
T=length(t_str);
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));

set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)    
setpixelposition(gcf,[223,365,1345,420]);
legend(symbol_info,'NumColumns',3,'Location','best');
box off