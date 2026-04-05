%{
1.细分因子标准化。对细分价值因子进行分位数变换标准化。
2.合成价值风格因子。将标准化之后的细分因子等权合成价值风格因子。

合成价值风格因子

%}

clear

tN = 'S30.F_month_final';
var_info = {'symbol','tradingdate','f_val'};

%tref = fetchmysql('select distinct(tradeDate) from yuqerdata.yq_dayprice order by tradeDate',2);
load tref
tref_num = datenum(tref);

sel_ind = tref_num>=datenum(2005,1,1);
tref = tref(sel_ind);
tref_num = tref_num(sel_ind);
%last day for the month
month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
month_cut_date1 = tref(month_cut(:,1));
month_cut_date2 = tref(month_cut(:,2));

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


T = length(month_cut_date2);
sql_str = 'select symbol,f_val from S30.f%d_month where tradingdate=''%s'' and f_val is not null';
sql_str_f2 = 'select symbol,f_val from S30.mv_month where tradingdate=''%s'' and f_val is not null';
symbolpool = cell(T,1);
symbolpool_f = cell(T,1);
for i = 1:T
    tref_sec = month_cut_date2{i};
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
    f(:,2) = month_cut_date2(i);
    %to mysql
    if ~isempty(f)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,f);
        close(conna);            
    end
    sprintf('合成价值风格因子 Complete: %d-%d',i,T)
end
