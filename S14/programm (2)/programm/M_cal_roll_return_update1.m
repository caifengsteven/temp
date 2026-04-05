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

由于收盘价很多缺失数据，使用结算价计算。

%}
clear

%create table
var1 = {'tradingdate','symbol','exchangeCD','R1','R2','R3','R4'};
var1_type = cell(size(var1));
var1_type(:) = {'float'};
var1_type(1:3) = {'date','varchar(10)','varchar(10)'};

db_name = 'futuredata';
tb_name = 'yuqer_future_rollreturn';
obj = mysqlTool();
sqlquery1=obj.createTable(db_name,tb_name,var1,var1_type);
OK1 = exemysql(sqlquery1);

tablename = sprintf('%s.%s',db_name,tb_name);

%获取品种
sql_str_gettype = ['select  distinct contractobject, exchangeCD from futuredata.yuqer_fusharedata',...
    ' where exchangeCD in(''XDCE'',''XSGE'',''XZCE'')'];
fushare_info = fetchmysql(sql_str_gettype,2);
%取第一个品种数据示例


T_f_type = size(fushare_info,1);
for f_id = 1:T_f_type
    f_code = fushare_info{f_id,1};%期货品种代码
    f_where = fushare_info{f_id,2};%所属交易所
    sub_fn = sprintf('re_%s_%s_update1.csv',f_code,f_where);
%     if exist(sub_fn,'file')
%         continue
%     end
    %获取该品种期货所有和约的交割日(使用最后交割日)
    sql_str_delidate = ['select ticker,lastdelidate from futuredata.yuqer_fushare_info ',...
        'where contractobject = ''%s'''];
    f_detail_date = fetchmysql(sprintf(sql_str_delidate,f_code),2);
    f_detail_date_num = datenum(f_detail_date(:,2));
    %获取时间
    sql_str_tref = ['select distinct(tradedate) from futuredata.yuqer_fusharedata ',...
        'where contractobject = ''%s'' order by tradedate'];
    tref = fetchmysql(sprintf(sql_str_tref,f_code),2);
    tref_num = datenum(tref);
    %获取日数据
    sql_str_data1 = ['select ticker,settleprice,contractMark,mainCon,smainCon from futuredata.yuqer_fusharedata ',...
        ' where contractobject = ''%s'' and tradedate=''%s'' order by ticker'];
    T = length(tref);

%     conna = database('ycz_zhubi','root','liudehua','com.mysql.jdbc.Driver',...
%         'jdbc:mysql://localhost:33006/ycz_zhubi?useSSL=false&');
    re = cell(T,7);
    eind = zeros(T,1);
    %try
        parfor i = 1:T
            sub_data = fetchmysql(sprintf(sql_str_data1,f_code,tref{i}),2);
            %去掉数据缺失数据
            sub_del_ind = cellfun(@isnan,sub_data(:,2));
            sub_data = sub_data(~sub_del_ind,:);
            if isempty(sub_data)
                continue
            end
            sub_code1 = sub_data(strcmpi(sub_data(:,3),'L0'),:);%当月;
            sub_code2 = sub_data(strcmpi(sub_data(:,3),'L1'),:);%次月;

            sub_code3 = sub_data(eq(cell2mat(sub_data(:,4)),1),:);%主力;
            sub_code4 = sub_data(eq(cell2mat(sub_data(:,5)),1),:);%次主力;

            sub_code5 = sub_data(end,:);%最远月


            sub_code_pair = {sub_code1,sub_code2;sub_code1,sub_code3;sub_code1,sub_code5;sub_code3,sub_code4};

            sub_r = zeros(4,1);
            for j = 1:4

                sub_code_n = sub_code_pair{j,1};%近月
                sub_code_d = sub_code_pair{j,2};%远月
                if ~isempty(sub_code_n)&&~isempty(sub_code_d)
                    sub_r(j) = call_roll_return(sub_code_n,sub_code_d,f_detail_date,f_detail_date_num,tref_num(i));
                else
                    sub_r(j) = nan;
                end
            end
            %keyboard
            sub_re = [{tref{i},f_code,f_where},num2cell(sub_r')];
            %tradingdate,symbol,exchangeCD,R1,R2,R3,R4


            sprintf('%d-%d (%d-%d)',i,T,f_id,T_f_type)
            re(i,:) = sub_re;
        end
        sub_empty_ind = cellfun(@isempty,re(:,1));
        re = re(~sub_empty_ind,:);
        %datainsert(conna,tablename,var1,re)
        re1 = cell2table(re,'VariableNames',var1);
        writetable(re1,sub_fn);
    %catch
    %    eind(i) = 1;
    %end
end

%dos('shutdown -s -t 0')
%close(conna)

