clear
%check database
dN = 'YCZfenbi';
dN_all = fetchmysql('show databases;',2);
if istable(dN_all)
    dN_all = table2cell(dN_all);
end
if ~any(strcmpi(dN_all,dN))
    exemysql(sprintf('create database %s',dN));
end
%check tables `

year_num = 2010;

%get all tables should exists
sql_str = ['select distinct tradedate from yuqerdata.yq_dayprice where tradedate ',...
    '>= ''%d0101'' and tradedate <=''%d0101'' order by tradedate'];
tref = fetchmysql(sprintf(sql_str,year_num,year_num+1),2);
tns = cellfun(@(x) x([1:4,6:7,9:10]),tref,'UniformOutput',false);
T = length(tns);

tns_all = fetchmysql(sprintf('show tables from %s',dN),2);
var_info = {'symbol','tradingdate','price','dealAmount','turnoverValue','volume','BSsel',...
                      'BP1','BP2','BP3','BP4','BP5','BS1','BS2','BS3',...
                      'BS4','BS5','BV1','BV2','BV3','BV4','BV5','SV1',...
                      'SV2','SV3','SV4','SV5'};

for i = 1:T
    tn = tns{i};
    if ~any(strcmpi(tns_all,tn))
        %create table   
        var_type = cell(size(var_info));
        var_type(:) = {'float'};
        var_type(1:2) = {'varchar(6)','datetime'};
        var_type(7) = {'varchar(6)'};
        obj = mysqlTool();
        sqlquery1=obj.createTable(dN,tn,var_info,var_type);
        OK1 = exemysql(sqlquery1);
        OK2 = exemysql(sprintf('alter table %s.%s engine=MyISAM;',dN,tn));
        OK3 = exemysql(sprintf('alter table %s.%s add primary key(symbol,tradingdate);',dN,tn));
    end
end



