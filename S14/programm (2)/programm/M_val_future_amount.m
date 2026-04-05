%{
FICC 系列研究之二 —— 基于动量和期限结构的商品期货策略
2.1 回测品种选择
%}

clear

symbol = {'SHFE.FU','SHFE.BU'};

symbol = symbol{1};

symbol = strsplit(symbol,'.');
sql_str = 'select tradingdate,close,volume/10000 from futuredata.price_if_data where variety0=''%s'' and variety=''%s'' and tradingdate <= ''2017-03-01'' order by tradingdate';

x = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);

tref = datenum(x(:,1));
y = cell2mat(x(:,3));
x = cell2mat(x(:,2));


yyaxis left
bar(y);
yyaxis right
plot(x,'LineWidth',2);
set(gca,'XTickLabelRotation',90);
set(gca,'XTick',floor(linspace(1,length(tref),40)),'xlim',[1,length(tref)]);
set(gca,'XTickLabel',cellstr(datestr(tref(floor(linspace(1,length(tref),40))),'yyyymmdd')));
%datetick('x','yyyymmdd','keeplimits');
set(gca,'fontsize',12);