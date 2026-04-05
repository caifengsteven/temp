%股东权益合计(不含少数股东权益)_最新财报 / 总市值
%所有者权益合计-少数股东权益  37_834_FS_Combas 合并资产负债表
%优矿数据替代
%
clear
key_str = '合成基础因子1';
tN = 'S30.F1_season';
var_info = {'symbol','pub_date','end_date','f_val'};

t0 = fetchmysql(sprintf('select pub_date from %s order by pub_date desc limit 1',tN),2);
X = yq_methods.get_HeBingZiChanFuZhai('TShEquity,minorityInt');
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