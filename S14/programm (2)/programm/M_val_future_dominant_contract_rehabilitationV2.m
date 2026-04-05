%{
FICC 系列研究之二 —— 主力合约合成
生成 主力合约复权
并写入数据库

将所有复权后的主力合约写入数据库
发现问题：
掘金主力合约根据自己的数据源合成的，我们的数据源是fushare，二者不匹配，有的掘金选入的
数据fushare没有；掘金合成的主力合约居然在切换的时候价格为0，会造成inf。fushare数据质量
比较差，居然有code写错的，比如Ma写成了ME


后续应对方法，更换fushare数据源。


%}

clear

%tradingdate,symbol,close_pirce,open_price,reh_factor

[~,~,x] = xlsread('future_info.xlsx','sheet3');
symbol0 = cellfun(@(x,y) [x,'.',y],x(:,1),x(:,2),'UniformOutput',false);
sy_info0 = x(:,3);
M = cell2mat(x(:,6));
T = length(symbol0);
tablename = 'futuredata.JJ_future_rehabilitation_data';
conna = database('ycz_zhubi','root','','com.mysql.jdbc.Driver','jdbc:mysql://localhost:3306/ycz_zhubi?useSSL=false&');
colnames = {'symbol','tradingdate','open_price','close_pirce','reh_factor'};
for symbol_sel = 24:T
    symbol = symbol0{symbol_sel};
    re = get_re_data(symbol);    
    datainsert(conna,tablename,colnames,re)
    
end

close(conna);

function re = get_re_data(symbol)

symbol = strsplit(symbol,'.');

%获取掘金主连数据
sql_str = 'select tradingdate,close,open from futuredata.price_if_data where variety0=''%s'' and variety=''%s''   order by tradingdate';
y_jj = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);

sql_str2 = 'select tradingdate,symbol from futuredata.future_contracts_data where variety=''%s.%s'' order by tradingdate';
sub_sql_str2 = sprintf(sql_str2,symbol{1},symbol{2});
index_contracts = fetchmysql(sub_sql_str2,2);

sql_str3 = 'select tradingdate from futuredata.price_%s_data where variety=''%s'' order by tradingdate';
tref_detail = fetchmysql(sprintf(sql_str3,symbol{1},symbol{2}),2);
[~,ia] = intersect(index_contracts(:,1),tref_detail);
index_contracts = index_contracts(ia,:);

[~,ia,ib] = intersect(y_jj(:,1),index_contracts(:,1));
y_jj = y_jj(ia,:);
y_jj_price = cell2mat(y_jj(:,2));
y_jj_price_open = cell2mat(y_jj(:,3));
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
            temp_str1 = index_contracts{i-1,2}(length(symbol{1})+2:end);
            temp_str2 = index_contracts{i,2}(length(symbol{1})+2:end);
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
        end
    end
    sprintf('%d-%d',i,T)
end
tref0 = y_jj(:,1);
re = [tref0,tref0,num2cell([[y_jj_price_open,y_jj_price].*rehabilitation_factor,rehabilitation_factor])];
re(:,1) = {strjoin(symbol,'.')};


end

