%周内效应
clear
symbol = {'000016','399300','000905'};
symbol_info = {'上证50','沪深300','中正500'};

sql_str = ['select chgPct from yuqerdata.yq_index where ',...
    'weekday(tradedate)=%d and symbol = ''%s'' and tradedate>=''2014-05-05'' and tradedate<=''2019-12-31'' '];
x = zeros(5,3);
for i = 1:5
    for j = 1:3
        temp = fetchmysql(sprintf(sql_str,i-1,symbol{j}));
        x(i,j) = sum(temp>0)/length(temp);
        i
    end
end

plot(x*100,'s-','linewidth',3)

set(gca,'XLim',[0.8,5.2])
set(gca,'XTick',1:5)
x_tl = cell(5,1);
for i = 1:5
    x_tl{i} = sprintf('周%d',i);
end
hold on
lims = axis(gca);
plot(lims(1:2),[50,50])
set(gca,'XTickLabel',x_tl);
setpixelposition(gcf,[298,324,940,420]);
box(gca,'off')
legend(symbol_info,'NumColumns',3,'Location','best')