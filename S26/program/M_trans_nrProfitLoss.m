%合成非经常性损益因子
%1 转换为季度  年度内数据如何处理
%2 filling
%3 写入数据库备用

clear
%paramas
key_str = 'S26转换非经常性损益因子';
tn_fullname = 'S26.f_nrProfitLoss';
var_info = {'symbol','tradingdate','f_val'};
tref_f = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tn_fullname),2);
if isempty(tref_f)
    tref_f = '2001-01-01';
else
    tref_f = tref_f{1};
    tref_f = datestr(datenum(tref_f)+1,'yyyy-mm-dd');
end

%basic data
tref = yq_methods.get_tradingdate(tref_f,datestr(now,'yyyy-mm-dd'));
tref_num = datenum(tref);

x = yq_methods.get_CaiWu_yansheng('nrProfitLoss');
symbols = unique(x(:,1));

T = length(symbols);
parfor i = 1:T
    sub_x = x(strcmp(x(:,1),symbols(i)),:);
    %转换为单个季度数值
    sub_x = flipud(sub_x);
    %去掉修正的数值
    [~,ia] = unique(sub_x(:,3),'stable');
    sub_x = sub_x(ia,:);
    %去掉missing数据
    del_ind = cellfun(@isnan,sub_x(:,end));
    sub_x(del_ind,:) = [];
    %计算时间节点
    sub_t_end = datenum(sub_x(:,3));
    sub_t_end_year = year(sub_t_end);
    u_sub_end_year = unique(sub_t_end_year);
    sub_T = length(u_sub_end_year);
    sub_y = cell(size(sub_x));
    %每个年度内，1季度，半年-一季度，前3季度-半年，全年-前3季度  转换
    for j = 1:sub_T
        sub_ind = eq(sub_t_end_year,u_sub_end_year(j));
        sub_v = cell2mat(sub_x(sub_ind,4));
        sub_v2 = [sub_v(1);diff(sub_v)];
        sub_y(sub_ind,:) = [sub_x(sub_ind,1:3),num2cell(sub_v2)];
    end
    %填充
    sub_f = nan(size(tref_num));    
    sub_t_cut = datenum(sub_y(:,2));
    sub_T2 = length(sub_t_cut);
    for j = 1:sub_T2
        if j <sub_T2
            sub_ind = tref_num>sub_t_cut(j)& tref_num<=sub_t_cut(j+1);
        else
            sub_ind = tref_num>sub_t_cut(j);
        end
        sub_f(sub_ind) = sub_y{j,end};
    end
    
    sub_ind = ~isnan(sub_f);
    sub_F = [tref(sub_ind),tref(sub_ind),num2cell(sub_f(sub_ind))];
        
    %write to table
    if ~isempty(sub_F)
        sub_F(:,1) = symbols(i);
        conna = mysql_conn();
        %write data to mysql
        datainsert(conna,tn_fullname,var_info,sub_F)
        close(conna)
    end
    
    sprintf('%s:%d-%d',key_str,i,T)
end

