%{
1.细分因子标准化。对细分价值因子进行分位数变换标准化。
2.合成价值风格因子。将标准化之后的细分因子等权合成价值风格因子。

合成价值风格因子

%}

clear
key_str = '合成价值风格因子';
tN = 'S30.F_month_final';
var_info = {'symbol','tradingdate','f_val'};
t0 = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tN),2);
%last day for the month
month_cut_date = yq_methods.get_month_data();
month_cut_date_num = datenum(month_cut_date);
ind = month_cut_date_num>datenum(t0);
month_cut_date = month_cut_date(ind);
month_cut_date_num = month_cut_date_num(ind);
T_month = length(month_cut_date);
if eq(T_month,0)
    sprintf('%s:Complete!',key_str)
    return
end

%载入ST信息数据
sql_str = 'SELECT * FROM yuqerdata.st_info order by tradedate desc';
x_st = fetchmysql(sql_str,2);
x_st(:,1) = cellfun(@str2double,x_st(:,1),'UniformOutput',false);
x_st_codenum = cell2mat(x_st(:,1));
x_st_u_codenum = unique(x_st_codenum);
x_st_data = cell(length(x_st_u_codenum),3);
for i = 1:length(x_st_u_codenum)
    sub_x_st_data=x_st(eq(x_st_codenum,x_st_u_codenum(i)),:);
    x_st_data(i,:) = {sprintf('%0.6d',x_st_u_codenum(i)),sub_x_st_data{1,2},sub_x_st_data{end,2}};
end
x_st_symbol = x_st_data(:,1);
x_st_date0 = datenum(x_st_data(:,3));
x_st_date1 = datenum(x_st_data(:,2));
%%%%%%%%%%%%%%%%%%%%%
sql_str = 'select symbol,f_val from S30.f%d_month where tradingdate=''%s'' and f_val is not null';
%sql_str_f2 = 'select symbol,f_val from S30.mv_month where tradingdate=''%s'' and f_val is not null';
%symbolpool = cell(T_month,1);
%symbolpool_f = cell(T_month,1);
X = cell(T_month);
parfor i = 1:T_month
    tref_sec = month_cut_date{i};
    tref_sec_num = datenum(tref_sec);
    %横截面数据
    for j = 1:5
        sub_x = fetchmysql(sprintf(sql_str,j,tref_sec),2);
        if eq(j,1)
            x = sub_x;
        else
            [~,ia,ib] = intersect(x(:,1),sub_x(:,1));
            x = [x(ia,:),sub_x(ib,end)];
        end
    end
    
    f = cell2mat(x(:,2:end));
    for j = 1:5
        sub_x = f(:,j);
        [~,~,sub_x] = unique(sub_x);
        f(:,j) = zscore(sub_x);
    end
    %合成价值风格因子。将标准化之后的细分因子等权合成价值风格因子。
    f = [x(:,[1,1]),num2cell(mean(f,2))];
    f(:,2) = month_cut_date(i);
    X{i} = f';
    sprintf('%s:Complete: %d-%d',key_str,i,T_month)
end
X = [X{:}]';
%to mysql
if ~isempty(X)
    conna = mysql_conn();
    datainsert(conna,tN,var_info,X);
    close(conna);            
end