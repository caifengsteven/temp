addpath('F:\works2019\SOME\项目\gmsdk');
%从掘金下载指数数据
set_token('294931e0418e0fa49482130b72b5ae27e58b0806')

data_type = {'indicator'};
sel = 1;

db_name = 'futuredata';
tb_name = sprintf('%s_data',data_type{sel});

var1 = {'tradingdate';'symbols';'symbolName';'open';'close';'high';'low';'amount';'volume'};

[~,~,x] = xlsread('指数.xlsx');
x=x(2:end,:);
ind = strcmp(x(:,end),'有');
x = x(ind,:);
t0 = datenum(x(:,3));
t0 = cellstr(datestr(t0,'yyyy-mm-dd'));
%t0 
T = length(t0);
%conna = database('ycz_zhubi','root','','Vendor','MySQL','Server','localhost');
conna = database('ycz_zhubi','root','','com.mysql.jdbc.Driver','jdbc:mysql://localhost:3306/ycz_zhubi?useSSL=false&');
for i = 4:T
    x1 = history(x(i,1),'1d',t0{i},'2019-05-01','ADJUST_NONE');
    temp1 = [x1.eob',num2cell([x1.open;x1.close;x1.high;x1.low;x1.amount;x1.volume]')];
    temp2 = repmat(x(i,1:2),size(temp1,1),1);
    temp1 = [temp1(:,1),temp2,temp1(:,2:end)];
    
    xlswrite([x{i,1},'.xlsx'],[var1';temp1]);
    %datainsert(conna,sprintf('%s.%s',db_name,tb_name),var1,temp1)
    
    
end



% 获取财务数据（表名，标的名，开始时间，结束时间，表字段）
% x = get_fundamentals('trading_derivative_indicator',{ 'SHSE.600000','SZSE.000001'}, '2018-04-01', '2018-08-01',  {'TCLOSE','NEGOTIABLEMV','TOTMKTCAP','TURNRATE'})



%x1 = history({'SZSE.399005'},'1d','2005-06-07','2019-05-01','ADJUST_NONE');
%var1 = {'tradingdate';'symbols';'symbolName';'open';'close';'high';'low';'amount';'volume'};


rmpath('F:\works2019\SOME\项目\gmsdk');


