%{
FICC 系列研究之二 —— 主力合约合成
生成 主力合约复权
并写入数据库

将所有复权后的主力合约写入数据库
发现问题：
后续应对方法，使用yqer数据计算（数据很全）
替换所有数据，格式要正确
%}

clear
key_str = 'S14合成后复权数据';
%获取品种
sql_str = 'select distinct(CONCAT(exchangeCD,"-",contractObject)) from yuqerdata.yq_MktMFutdGet where exchangeCD in ("XDCE","XSGE","XZCE");';
x = fetchmysql(sql_str,2);
x=cellfun(@(x) strsplit(x,'-')',x,'UniformOutput',false);
x = [x{:}]';

T = size(x,1);
tablename = 'futuredata.YQ_future_rehabilitation_data';

%conna = database('futuredata','root','liudehua','com.mysql.jdbc.Driver','jdbc:mysql://localhost:3306/futuredata?useSSL=false&');
colnames = {'symbol','tradingdate','open_price','close_price','reh_factor'};
sql_str1 = 'select * from %s where symbol = ''%s'' order by tradingdate desc limit 1';
re = cell(T,1);
parfor i = 1:T
    symbol = x(i,:);
    sub_x0 = fetchmysql(sprintf(sql_str1,tablename,strjoin(symbol,'.')),2);
    sub_re = get_re_data_yq_update(symbol,sub_x0);    
    re{i} = sub_re';
    sprintf('%s:%d-%d',key_str,i,T)
end
%close(conna);
re = [re{:}]';
datainsert_adair(tablename,colnames,re)


