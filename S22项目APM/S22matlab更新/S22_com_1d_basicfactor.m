%1day factor
%com 1day factor
%补充，中间有重复结果
clear
%parameters
key_str = 'S22合成APB日度高频指标';
dn = 'S22';
tn = 's22_factor_apb_1d_single';
tn_fullname = sprintf('%s.%s',dn,tn);

dn_ycz = 'ycz_min_history';
tn_yczall = fetchmysql(sprintf('show tables from %s',dn_ycz),2);
tn_yczall = sort(tn_yczall);
name_length =cellfun(@length,tn_yczall);
tn_yczall = tn_yczall(eq(name_length,8));

%section 1 calculate factor
%check table
%check database

sub_sql = 'show databases';
info = fetchmysql(sub_sql,2);
if ~any(strcmpi(info,dn))
    exemysql(sprintf('create database %s',dn));
end
sub_sql = sprintf('show tables from %s',dn);
info = fetchmysql(sub_sql,2);

%create database ? table ? primary key ? MyISAM
var_info = {'symbol','tradingdate','f_val'};
if ~any(strcmpi(info,tn))
    %create table    
    var_type = cell(size(var_info));
    var_type(:) = {'float'};
    var_type(1:2) = {'varchar(6)','date'};
    obj = mysqlTool();
    sqlquery1=obj.createTable(dn,tn,var_info,var_type);
    OK1 = exemysql(sqlquery1);
    OK2 = exemysql(sprintf('alter table %s.%s engine=MyISAM;',dn,tn));
    OK3 = exemysql(sprintf('alter table %s.%s add primary key(symbol,tradingdate);',dn,tn));
end
%deal day by day
tref_complete = fetchmysql(sprintf('select distinct(tradingdate) from %s',tn_fullname),2);
tref_complete = cellfun(@(x) x([1:4 6:7 9:10]),tref_complete,'UniformOutput',false);
tn_yczall = setdiff(tn_yczall,tref_complete);
T = length(tn_yczall);
sql_str_f1 = 'select symbol,amount/volume/100, volume*100  from %s order by tradingdate';
parfor i = 1:T
    sub_t = sprintf('%s-%s-%s',tn_yczall{i}(1:4),tn_yczall{i}(5:6),tn_yczall{i}(7:8));
	sub_tn_fullname = sprintf('%s.%s',dn_ycz,tn_yczall{i});
    sub_x = fetchmysql(sprintf(sql_str_f1,sub_tn_fullname),2);
    sub_symbols = unique(sub_x(:,1));
    T_symbols = length(sub_symbols);
    sub_sub_f=zeros(T_symbols,1);
    for j = 1:T_symbols
        sub_sub_x = cell2mat(sub_x(strcmp(sub_x(:,1),sub_symbols(j)),2:3));   
        nan_ind = isnan(sum(sub_sub_x,2));
        sub_sub_x(nan_ind,:) = [];
        if ~isnan(sub_sub_x)
            sub_sub_f(j) = log(mean(sub_sub_x(:,1))/(sum(sub_sub_x(:,1).*sub_sub_x(:,2))/sum(sub_sub_x(:,2))));
        end
    end
    %write to mysql
    
    sub_re = [cellfun(@(x) x(3:end),sub_symbols,'UniformOutput',false),sub_symbols,num2cell(sub_sub_f)];
    sub_re(:,2) = {sub_t};
    %insert data to database
    conna = mysql_conn();
    %write data to mysql
    datainsert(conna,tn_fullname,var_info,sub_re)
    close(conna)
    
    sprintf('%s:%d-%d',key_str,i,T)
    
    
end
