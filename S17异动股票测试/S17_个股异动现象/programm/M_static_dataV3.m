%{
盘中异动现象的短期收益观察
找到文献中亏、盈异动信号并写入数据库
找到所有信号
%}
clear

t1 = '2013-01-01';
t2 = '2017-01-01';

cut_value = 1/100;
max_d = 10;
var1 = {'symbol','tradingdate','precoloseprice','closeprice','r1','d','r2'};
db_name = 'ycz_result';
tb_name = 'ycz_result.sta_re20190702_All';
%代码，时间，前收，现收，收益率1，时间间隔，收益率2
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
sql_str1 = 'select symbol,tradingdate,close from ycz_min_history.`%s` order by symbol,tradingdate';
sql_str2 = 'select symbol,precloseprice from futuredata.STK_MKT_QUOTATION where tradingdate = ''%s''';

re_all = cell(T,1);

parfor i = 1:T
    sub_x = fetchmysql(sprintf(sql_str1,tref_str2{i}),2);
    sub_y = fetchmysql(sprintf(sql_str2,tref{i}),2);
    sub_symbols = unique(sub_x(:,1));
    
    
    Q = length(sub_symbols);
    sub_re = cell(Q*20,7);
    sub_re_ind = 0;
    for j = 1:Q
        temp_v = cell2mat(sub_y(strcmp(sub_y(:,1),sub_symbols{j}(3:end)),2));
        if isempty(temp_v)
            temp_v = 0;
        end
        sub_sub_x_a = sub_x(strcmp(sub_x(:,1),sub_symbols(j)),:);
        sub_sub_x = cell2mat(sub_sub_x_a(:,3));
        sub_sub_r = [0;sub_sub_x(2:end)./sub_sub_x(1:end-1)-1];        
        sub_inds = find(sub_sub_r>cut_value(1));
        
        for j1 = 1:length(sub_inds)
            sub_ind = sub_inds(j1);
            %股盘中首次 出现1分钟交易时间内
            k = 1;
            sub_sub_re=cell(max_d,7);
            while sub_ind+k<=length(sub_sub_r)&&k<=max_d
                sub_sub_re(k,:) = {sub_symbols{j},sub_sub_x_a{sub_ind+k,2},temp_v,...
                    sub_sub_x(sub_ind),sub_sub_r(sub_ind),k,sub_sub_x(sub_ind+k)/sub_sub_x(sub_ind)-1};
                k = k + 1;
            end
            sub_sub_re = sub_sub_re(1:k-1,:);
            %sub_re = cat(1,sub_re,sub_sub_re);
            sub_re(sub_re_ind+1:sub_re_ind+size(sub_sub_re,1),:) = sub_sub_re;
            sub_re_ind = sub_re_ind+size(sub_sub_re,1);
        end       
        
        %
        sub_inds = find(sub_sub_r<-cut_value(1));
        
        for j1 = 1:length(sub_inds)% ~isempty(sub_ind)
            sub_ind = sub_inds(j1);
            %股盘中首次 出现1分钟交易时间内
            k = 1;
            sub_sub_re=cell(max_d,7);
            while sub_ind+k<=length(sub_sub_r)&&k<=max_d
                sub_sub_re(k,:) = {sub_symbols{j},sub_sub_x_a{sub_ind+k,2},temp_v,...
                    sub_sub_x(sub_ind),sub_sub_r(sub_ind),k,sub_sub_x(sub_ind+k)/sub_sub_x(sub_ind)-1};
                k = k + 1;
            end
            sub_sub_re = sub_sub_re(1:k-1,:);
            sub_re(sub_re_ind+1:sub_re_ind+size(sub_sub_re,1),:) = sub_sub_re;
            sub_re_ind = sub_re_ind+size(sub_sub_re,1);
        end
        sprintf('%d-%d %d-%d',j,Q,i,T)
    end
    sub_re = sub_re(1:sub_re_ind,:);    
    conna = database('futuredata','root','liudehua','com.mysql.jdbc.Driver','jdbc:mysql://localhost:3306/futuredata?useSSL=false&');
    datainsert(conna,tb_name,var1,sub_re);
    close(conna);
    
end


