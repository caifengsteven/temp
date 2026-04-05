%{

合成展期收益率因子
包含以下类型：
第一种是近月和次近月 R1
第二种是近月和主力 R2
第三种是近月和最远月 R3
第四种是主力和次主力 R4

数据库相关
%create table yuqer_fushare_rollreturn
%columns: tradingdate,symbol,R1,R2,R3,R4

update
处理数据为nan现象



%}
clear

%create table
var1 = {'tradeDate','contractObject','exchangeCD','wrVOL'};

db_name = 'futuredata';
tb_name = 'yq_warehousefactor_data';

tablename = sprintf('%s.%s',db_name,tb_name);

%获取品种
sql_str_gettype = ['select  distinct contractobject, exchangeCD from futuredata.yuqer_fusharedata',...
    ' where exchangeCD in(''XDCE'',''XSGE'',''XZCE'')'];
fushare_info = fetchmysql(sql_str_gettype,2);
%取第一个品种数据示例


T_f_type = size(fushare_info,1);
for f_id = 41:T_f_type
    f_code = fushare_info{f_id,1};%期货品种代码
    f_where = fushare_info{f_id,2};%所属交易所
    sub_fn = sprintf('re_%s_%s.csv',f_code,f_where);
%     if exist(sub_fn,'file')
%         continue
%     end
    %获取日数据
    sql_str_data1 = ['select tradeDate,wrvol from futuredata.yq_warehouse_data ',...
        ' where contractobject = ''%s'' order by tradedate'];
    sub_x = fetchmysql(sprintf(sql_str_data1,f_code),2);
    if isempty(sub_x)
        continue
    end
    tref = unique(sub_x(:,1));
    T = length(tref);

    conna = database('ycz_zhubi','root','352471Cf','com.mysql.jdbc.Driver',...
        'jdbc:mysql://localhost:33006/ycz_zhubi?useSSL=false&');
    re = cell(T,4);
    eind = zeros(T,1);
    %try
        for i = 1:T
            sub_data = sub_x(strcmp(sub_x(:,1),tref(i)),2);
            
            if ~isempty(sub_data)
                temp = cell2mat(sub_data);
                %keyboard
                sub_re = [{tref{i},f_code,f_where},num2cell(sum(temp))];
                %tradingdate,symbol,exchangeCD,R1,R2,R3,R4


                sprintf('%d-%d (%d-%d)',i,T,f_id,T_f_type)
                re(i,:) = sub_re;
            end
        end
        sub_empty_ind = cellfun(@isempty,re(:,1));
        re = re(~sub_empty_ind,:);
        datainsert(conna,tablename,var1,re)
        %re1 = cell2table(re,'VariableNames',var1);
        %writetable(re1,sub_fn);
    %catch
    %    eind(i) = 1;
    %end
end

%dos('shutdown -s -t 0')
%close(conna)

