%经营活动产生的现金流量净额_TTM  经营活动产生的现金流量净额
%yuqerdata.yq_FdmtCFTTMPITGet   NCFOperateA
clear 
key_str = '合成基础因子4';
tN = 'S30.F4_season';
var_info = {'symbol','pub_date','end_date','f_val'};

t0 = fetchmysql(sprintf('select pub_date from %s order by pub_date desc limit 1',tN),2);
sql_str0 = ['select ticker,publishDate,endDate,NCFOperateA from ',...
    'yuqerdata.yq_FdmtCFTTMPITGet where publishDate>''%s'' order by endDate desc,publishDate desc' ];
X = fetchmysql(sprintf(sql_str0,t0{1}),2);
if isempty(X)
    sprintf('%s：Complete',key_str)
    return
end
y = cellfun(@(x,y) [x,y],X(:,1),X(:,3),'UniformOutput',false);
[~,ia] =unique(y);
X= X(ia,:);

nan_ind = cellfun(@isnan,X(:,end));
X(nan_ind,:) = [];
%write to mysql
if ~isempty(X)
    conna = mysql_conn();
    datainsert(conna,tN,var_info,X);
    close(conna);            
end
sprintf('%s：Complete',key_str)