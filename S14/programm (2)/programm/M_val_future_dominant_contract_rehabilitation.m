%{
FICC 系列研究之二 —— 主力合约合成
2.2 主力合约复权
%}

clear

symbol = {'SHFE.FU','SHFE.BU','SHFE.RB'};
sy_info = {'燃油','沥青','螺纹钢'};
symbol = symbol{3};

symbol = strsplit(symbol,'.');

%获取掘金主连数据
sql_str = 'select tradingdate,close from futuredata.price_if_data where variety0=''%s'' and variety=''%s'' and tradingdate <= ''2017-03-01'' order by tradingdate';
y_jj = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);

sql_str2 = 'select tradingdate,symbol from futuredata.future_contracts_data where variety=''%s.%s'' order by tradingdate';
sub_sql_str2 = sprintf(sql_str2,symbol{1},symbol{2});
index_contracts = fetchmysql(sub_sql_str2,2);

[~,ia,ib] = intersect(y_jj(:,1),index_contracts(:,1));
y_jj = y_jj(ia,:);
y_jj_price = cell2mat(y_jj(:,end));
index_contracts = index_contracts(ib,:);

sql_str3 = 'select codename,close,volume from futuredata.price_%s_data where variety=''%s'' and tradingdate = ''%s'' order by volume desc';
rehabilitation_factor = zeros(size(y_jj(:,1)));
T = length(rehabilitation_factor);
for i = 1:T
    if eq(i,1)
        rehabilitation_factor(i) = 1;
    else
        if strcmp(index_contracts(i,2),index_contracts(i-1,2))
            rehabilitation_factor(i) = rehabilitation_factor(i-1);
        else
            %复权因子（T）= 复权因子（T-1）×旧主力合约前收盘价（T）/新主力合约前收盘价（T）
            sub_sql_str = sprintf(sql_str3,symbol{1},symbol{2},y_jj{i-1,1});
            sub_x = fetchmysql(sub_sql_str,2);
            %get data
            ia = strcmpi(sub_x(:,1),index_contracts{i-1,2}(length(symbol{1})+2:end));
            sub_x1 = sub_x(ia,:);
            ib = strcmpi(sub_x(:,1),index_contracts{i,2}(length(symbol{1})+2:end));
            sub_x2 = sub_x(ib,:);
            rehabilitation_factor(i) = rehabilitation_factor(i-1)*sub_x1{1,2}/sub_x2{1,2};
            
        end
    end
    sprintf('%d-%d',i,T)
end

%yyaxis left
obj = plot([y_jj_price,y_jj_price.*rehabilitation_factor],'-','LineWidth',2);
% yyaxis right
% plot(rehabilitation_factor,'-','LineWidth',2);

tref0 = y_jj(:,1);
set(gca,'XTickLabelRotation',90);
set(gca,'XTick',floor(linspace(1,length(tref0),40)),'xlim',[1,length(tref0)]);
set(gca,'XTickLabel',tref0(floor(linspace(1,length(tref0),40))))
set(gca,'fontsize',12);

legend(obj,{'主力合约收盘价','复权后主力合约收盘价'},'Location','northwest',...
    'NumColumns',length(obj),'location','best')

