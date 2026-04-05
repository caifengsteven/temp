%+非流动负债合计_最新财报 - 货币资金_最新财报
%TNCL,cashCEquiv
clear
key_str = '合成基础因子5';

tN = 'S30.F5_season';
var_info = {'symbol','pub_date','end_date','f_val'};
t0 = fetchmysql(sprintf('select pub_date from %s order by pub_date desc limit 1',tN),2);
X = yq_methods.get_HeBingZiChanFuZhai('TNCL,cashCEquiv');
ind = datenum(X(:,2))>datenum(t0);
X= X(ind,[1,2,3,end-1,end]);
if isempty(X)
    sprintf('%s：Complete',key_str)
    return
end

sub_v = cell2mat(X(:,end-1:end));
sub_v(isnan(sub_v)) = 0;
sub_v = sub_v(:,end-1)-sub_v(:,end);
del_ind = eq(sub_v,0);
X(del_ind,:) = [];
sub_v(del_ind,:) = [];
if isempty(X)
    sprintf('%s：Complete',key_str)
    return
end

X = [X(:,1:end-2),num2cell(sub_v)];
%去重
y = cellfun(@(x,y) [x,y],X(:,1),X(:,3),'UniformOutput',false);
[~,ia] =unique(y);
X= X(ia,:);
%%%%%%%%%%%
nan_ind = cellfun(@isnan,X(:,end));
X(nan_ind,:) = [];
%write to mysql
if ~isempty(X)
    conna = mysql_conn();
    datainsert(conna,tN,var_info,X);
    close(conna);            
end
sprintf('%s：Complete',key_str)