%{
尾盘信号
尾盘30分钟内触发信号后开仓、收盘平仓、开盘平仓
阈值包括1%、2%、3%、4%、5%
%}
clear

t1 = '2013-01-01';
t2 = '2017-01-01';

cut_value = (1:5)/100;
max_d = 10;
var1 = {'symbol','tradingdate','precoloseprice','closeprice','r1','d','r2'};
db_name = 'ycz_result';
tb_name = 'ycz_result.sta_re20190702_last_30min';
%代码，时间，前收，现收，收盘价，时间间隔，第二日开盘价
%获取交易日历
sql_str = ['select distinct tradingdate from futuredata.STK_MKT_QUOTATION ',...
    'where tradingdate >=''%s'' and tradingdate<=''%s'' and filling =0'];

tref = fetchmysql(sprintf(sql_str,t1,t2),2);
tref_num = datenum(tref);
tref_str2 = cellstr(datestr(tref_num,'yyyymmdd'));
%查看table
%sql_str = 'show tables from ycz_min_history';
%tb_all = fetchmysql(sql_str,2);
%数据未缺失
T = length(tref_str2);
sql_str1 = ['select symbol,tradingdate,close from ycz_min_history.`%s` ',...
    'where  (hour(tradingdate)>=14 and minute(tradingdate)>=30) or hour(tradingdate)=15  order by symbol,tradingdate'];

sql_str2 = 'select symbol,precloseprice from futuredata.STK_MKT_QUOTATION where tradingdate = ''%s''';
sql_str3 = ['select symbol,open from ycz_min_history.`%s` where ',...
    'hour(tradingdate)=9 and minute(tradingdate)<=31'];

re_all = cell(T,1);

parfor i = 1:T-1
    sub_x = fetchmysql(sprintf(sql_str1,tref_str2{i}),2);
    sub_y = fetchmysql(sprintf(sql_str2,tref{i}),2);
    sub_x_next = fetchmysql(sprintf(sql_str3,tref_str2{i+1}),2);
    sub_symbols = unique(sub_x(:,1));
    
    
    Q = length(sub_symbols);
    sub_re = cell(Q*20,7);
    sub_re_ind = 0;
    for j = 1:Q
        temp_v = cell2mat(sub_y(strcmp(sub_y(:,1),sub_symbols{j}(3:end)),2));
        if isempty(temp_v)
            temp_v = 0;
        end
        temp_v2 = cell2mat(sub_x_next(strcmp(sub_x_next(:,1),sub_symbols(j)),2));
        if isempty(temp_v2)
            temp_v2 = 0;
        end
        sub_sub_x_a = sub_x(strcmp(sub_x(:,1),sub_symbols(j)),:);
        temp_v3 = sub_sub_x_a{end,end};
        sub_sub_x = cell2mat(sub_sub_x_a(:,3));
        sub_sub_r = [0;sub_sub_x(2:end)./sub_sub_x(1:end-1)-1];      
        for k = 1:5
            sub_ind = find(sub_sub_r>cut_value(k),1);
            if ~isempty(sub_ind)
                %代码，时间，前收，现收，收盘价，时间间隔，第二日开盘价
                sub_sub_re = {sub_symbols{j},sub_sub_x_a{sub_ind,2},temp_v,...
                    sub_sub_x(sub_ind),temp_v3,k,temp_v2};
                sub_re(sub_re_ind+1:sub_re_ind+size(sub_sub_re,1),:) = sub_sub_re;
                sub_re_ind = sub_re_ind+size(sub_sub_re,1);
            end
        end
        
        %信号2
        for k = 1:5
            sub_ind = find(sub_sub_r<-cut_value(k),1);
            if ~isempty(sub_ind)
                %代码，时间，前收，现收，收盘价，时间间隔，第二日开盘价
                sub_sub_re = {sub_symbols{j},sub_sub_x_a{sub_ind,2},temp_v,...
                    sub_sub_x(sub_ind),temp_v3,-k,temp_v2};
                sub_re(sub_re_ind+1:sub_re_ind+size(sub_sub_re,1),:) = sub_sub_re;
                sub_re_ind = sub_re_ind+size(sub_sub_re,1);
            end
        end
        sprintf('%d-%d %d-%d',j,Q,i,T)
    end
    sub_re = sub_re(1:sub_re_ind,:);    
    conna = database('futuredata','root','liudehua','com.mysql.jdbc.Driver','jdbc:mysql://localhost:3306/futuredata?useSSL=false&');
    datainsert(conna,tb_name,var1,sub_re);
    close(conna);
    
end


