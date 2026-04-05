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

%tradingdate,symbol,close_pirce,open_price,reh_factor

[~,~,x] = xlsread('future_info.xlsx','sheet3');
symbol0 = cellfun(@(x,y) [x,'.',y],x(:,1),x(:,2),'UniformOutput',false);
sy_info0 = x(:,3);
M = cell2mat(x(:,6));
T = length(symbol0);
tablename = 'futuredata.YQ_future_rehabilitation_data';

conna = database('futuredata','root','liudehua','com.mysql.jdbc.Driver','jdbc:mysql://localhost:3306/futuredata?useSSL=false&');
colnames = {'symbol','tradingdate','open_price','close_pirce','reh_factor'};
for symbol_sel = 24:T
    symbol = symbol0{symbol_sel};
    re = get_re_data_yq(symbol);    
    
    datainsert(conna,tablename,colnames,re) 
    
end
close(conna);




