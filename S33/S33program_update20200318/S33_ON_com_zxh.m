%原程序为逐个股票计算，计算速度快
%修改为按照日期计算，便于每日更新数据
%因子为月度数据
clear
print_sel = true;
key_str = 'S33合成中性化因子';
tN= 'S33.factor_zxh';
var_info = {'symbol','tradingdate','f_mv','f_reverse','f_std','f_change'};
window1 = 22;

%get date
t1 = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tN),2);
t1 = datestr(datenum(t1)+1,'yyyy-mm-dd');%从下一个日期开始
t2 = datestr(now,'yyyy-mm-dd');%当前时间（截至时间）
tref1 = yq_methods.get_tradingdate(t1,t2);%没有计算过的时间
tref2 = yq_methods.get_tradingdate_future(tref1{end});%从当前开始（含当前）公示的未来交易日
tref = [tref1;tref2(2)];
tref_num = datenum(tref);

%获取月底日期
%last day for the month
month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
if length(month_cut)<2
    sprintf('%s:已经是最新日期，无需更新',key_str)
    T_month = 0;
else

month_cut_date1 = tref(month_cut(:,1));
month_cut_date2 = tref(month_cut(:,2));
month_trefnum = datenum(month_cut_date2);

%
T_month = length(month_cut_date2);

sql_str =[ 'select symbol,tradeDate,negMarketValue,chgPct,turnoverRate from ',...
    'yuqerdata.yq_dayprice where tradeDate>= ''%s'' and tradeDate<=''%s'' order by tradeDate'];
end
for i = 1:T_month
    %获取交易日期
    sub_tref = yq_methods.get_tradingdate('2000-01-01',month_cut_date2{i});
    sub_tref = sub_tref(end-window1+1:end);
    %读入数据
    x = fetchmysql(sprintf(sql_str,sub_tref{1},sub_tref{end}),2);
    %记录 mv
    sub_f1 = x(strcmp(x(:,2),sub_tref(end)),[1,3]);
    sub_symbol = sub_f1(:,1);
    sub_T = length(sub_symbol);
    %2 reverse 3 std 反转 波动
    %4 换手
    sub_re = cell(sub_T,1);
    parfor j = 1:sub_T
        sub_y = x(strcmp(x(:,1),sub_symbol(j)),4:5);
        if size(sub_y,1)<window1 %中间有停牌去掉
            sub_re{j} = nan(3,1);
            continue
        end
        sub_y = cell2mat(sub_y);
        sub_y(isnan(sum(sub_y,2)),:) = [];
        if size(sub_y,1)<window1/4*3
            sub_re{j} = nan(3,1);
            continue
        end
        sub_sub_f1 = cumprod(1+sub_y(:,1))-1;
        sub_sub_f1 = sub_sub_f1(end);
        sub_sub_f2 = std(sub_y(:,1));
        sub_sub_f3 = mean(sub_y(:,2));
        sub_re{j} = [sub_sub_f1;sub_sub_f2;sub_sub_f3];
        if print_sel
            sprintf('%s：%d-%d %d-%d',key_str,j,sub_T,i,T_month)
        end
    end
    sub_re = [sub_re{:}]';
    sub_fv = [cell2mat(sub_f1(:,2)),sub_re];
    sub_f = [sub_f1(:,1),sub_f1(:,1),num2cell(sub_fv)];
    sub_f(:,2) = sub_tref(end);
    
    del_ind = isnan(sum(sub_fv,2));
    sub_f = sub_f(~del_ind,:);    
    %write to mysql
    if ~isempty(sub_f)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,sub_f);
        close(conna);            
    end

end
