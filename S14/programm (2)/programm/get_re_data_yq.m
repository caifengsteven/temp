function re = get_re_data_yq(symbol)

symbol = strsplit(symbol,'.');
fix_fushare_info = containers.Map({'CZCE','SHFE','DCE'},{'XZCE','XSGE','XDCE'});
symbol{1} = fix_fushare_info(symbol{1});
sql_str = ['select tradedate,ticker,openprice,closeprice from futuredata.yuqer_fusharedata ',...
        'where exchangeCD=''%s'' and contractObject=''%s''and openprice is not null and closeprice is not null ',...
        'and tradedate <= ''2017-03-01''  and tradedate>=''2005-01-01'' and mainCon=1 order by tradedate'];
y_yq = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);
y_yq_price_open = cell2mat(y_yq(:,3));
y_yq_price_close = cell2mat(y_yq(:,4));
index_contracts = y_yq(:,1:2);    

sql_str3 = ['select ticker,closeprice,turnoverVol from futuredata.yuqer_fusharedata ',...
    'where exchangeCD=''%s'' and contractObject=''%s'' and tradedate = ''%s'' order by turnoverVol desc'];
rehabilitation_factor = zeros(size(y_yq(:,1)));
T = length(rehabilitation_factor);
for i = 1:T
    if eq(i,1)
        rehabilitation_factor(i) = 1;
    else
        if strcmp(index_contracts(i,2),index_contracts(i-1,2))
            rehabilitation_factor(i) = rehabilitation_factor(i-1);
        else
            %复权因子（T）= 复权因子（T-1）×旧主力合约前收盘价（T）/新主力合约前收盘价（T）
            sub_sql_str = sprintf(sql_str3,symbol{1},symbol{2},y_yq{i-1,1});
            sub_x = fetchmysql(sub_sql_str,2);
            %get data
            temp_str1 = index_contracts{i-1,2};
            temp_str2 = index_contracts{i,2};
            if length(temp_str1)<length(sub_x{end,1})
                temp_str1 = sprintf('%s20%s',temp_str1(1:end-4),temp_str1(end-3:end));
                temp_str2 = sprintf('%s20%s',temp_str2(1:end-4),temp_str2(end-3:end));                
            elseif length(temp_str1)>length(sub_x{end,1})
                temp_str1 = sprintf('%s%s',temp_str1(1:end-6),temp_str1(end-3:end));
                temp_str2 = sprintf('%s%s',temp_str2(1:end-6),temp_str2(end-3:end)); 
            end
            ia = strcmpi(sub_x(:,1),temp_str1);
            sub_x1 = sub_x(ia,:);
            ib = strcmpi(sub_x(:,1),temp_str2);
            sub_x2 = sub_x(ib,:);
            %if eq(i,120);keyboard;end
            %补充，有的合约名称有20，有的没有20
            if isempty(sub_x2)||eq(sub_x2{1,2},0)||isempty(sub_x1)
                rehabilitation_factor(i) = rehabilitation_factor(i-1);
            else
                rehabilitation_factor(i) = rehabilitation_factor(i-1)*sub_x1{1,2}/sub_x2{1,2};
            end
            if isnan(rehabilitation_factor(i))||eq(0,rehabilitation_factor(i))
                rehabilitation_factor(i) = rehabilitation_factor(i-1);
            end
        end
    end
    sprintf('%d-%d',i,T)
end
tref0 = y_yq(:,1);
re = [tref0,tref0,num2cell([[y_yq_price_open,y_yq_price_close].*rehabilitation_factor,rehabilitation_factor])];
re(:,1) = {strjoin(symbol,'.')};


end