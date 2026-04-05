%{
Ret20d 为前 20 个交易日的涨跌幅
升级为写入所有交易日（上一版本为只写入月底数据）
%}
clear
key_str = 'S32反转因子（日度频率）';
m_start_time = datetime;
print_sel = true;
tN2 = 'yuqerdata.yq_dayprice';
tN3 = 'S32.ret20d_update';

var_info = {'symbol','tradingdate','f_val','f_val2'};
window1 = 20;

t1 = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tN3),2);
t1 = datestr(datenum(t1)+1,'yyyy-mm-dd');%从下一个日期开始
t2 = datestr(now,'yyyy-mm-dd');%当前时间（截至时间）
tref = yq_methods.get_tradingdate(t1,t2);%没有计算过的时间

sql_str1 = ['select symbol,tradedate,negMarketValue,chgPct from %s where tradeDate>=''%s'' and tradeDate<=''%s'' ',...
    'and chgPct is not null and negMarketValue is not null order by tradedate'];


T=length(tref);
for i = 1:T
    %获取交易日期
    sub_tref = yq_methods.get_tradingdate('2000-01-01',tref{i});
    sub_tref = sub_tref(end-window1+1:end);
    
    x = fetchmysql(sprintf(sql_str1,tN2,sub_tref{1},sub_tref{end}),2);
    %记录 mv
    sub_f1 = x(strcmp(x(:,2),sub_tref(end)),[1,3]);
    sub_symbol = sub_f1(:,1);
    sub_T = length(sub_symbol);
    
    %2 reverse
    sub_re = cell(sub_T,1);
    parfor j = 1:sub_T
        sub_y = x(strcmp(x(:,1),sub_symbol(j)),4);
        if size(sub_y,1)<window1 %中间有停牌去掉
            sub_re{j} = nan;
            continue
        end
        sub_y = cell2mat(sub_y);
        sub_y(isnan(sum(sub_y,2)),:) = [];
        if size(sub_y,1)<window1/4*3
            sub_re{j} = nan;
            continue
        end
        sub_sub_f1 = cumprod(1+sub_y(:,1))-1;
        sub_sub_f1 = sub_sub_f1(end);
%         sub_sub_f2 = std(sub_y(:,1));
%         sub_sub_f3 = mean(sub_y(:,2));
        %sub_re{j} = [sub_sub_f1;sub_sub_f2;sub_sub_f3];
        sub_re{j} = sub_sub_f1;
        if print_sel
            sprintf('%s：%d-%d %d-%d',key_str,j,sub_T,i,T)
        end
    end
    
    sub_re = [sub_re{:}]';
    sub_fv = [cell2mat(sub_f1(:,2)),sub_re];
    sub_f = [sub_f1(:,1),sub_f1(:,1),num2cell(sub_fv)];
    sub_f(:,2) = sub_tref(end);
    
    del_ind = isnan(sum(sub_fv,2));
    sub_f = sub_f(~del_ind,:);    
    
    %insert to mysql
    if ~isempty(sub_f)
        conna = mysql_conn();
        datainsert(conna,tN3,var_info,sub_f);
        close(conna);            
    end
end

m_end_time = datetime;
sprintf('Time used %s',m_end_time-m_start_time)