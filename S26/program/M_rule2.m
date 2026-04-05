%M_rule1
%{
RULE2：净利润(Y-1)<0 AND 净利润(Y-2)<0 AND YOY(营业收入)>0.2
%}

clear
tN1 = 'S26.FdmtBSAllLatestGet';
%
window = 2;
obj_yq = yq_methods();
%净利润，营业收入
x = obj_yq.get_HeBingLiRun('NIncomeAttrp,revenue');
x = x(strcmp(x(:,5),'A'),:);

sql_str = 'SELECT ticker,publishDate FROM s26.equrestructuringget where isSucceed=1 order by publishDate';
y=fetchmysql(sql_str,2);

symbol_N = unique(x(:,1));
T = length(symbol_N);

re_rule1 = cell(T,1);
parfor i = 1:T    
    sub_x = x(strcmp(x(:,1),symbol_N(i)),:);
    [~,ia] = unique(sub_x(:,3),'stable');
    sub_x = sub_x(ia,:);
    sub_symbol = sprintf('%0.6d',str2double(symbol_N(i)));
    sub_y = y(strcmp(y(:,1),sub_symbol),:);
    if ~isempty(sub_y)
        sub_y_t = datenum(sub_y(:,2));
    else
        sub_y_t = 0;
    end
    
    sub_t = sub_x(:,2);
    sub_t_num = datenum(sub_t);
    sub_v = cell2mat(sub_x(:,6:7));
    [sub_t_num,ia] = sort(sub_t_num);
    sub_v = sub_v(ia,:);
    sub_t = sub_t(ia);
    
    sub_T = length(sub_t);
    sub_re = cell(sub_T,1);
    for j = window+1:sub_T
        sub_window_v = sub_v(j-window:j-1,1);
        if all(sub_window_v<0) && sub_v(j,2)/sub_v(j-1,2)>0.2 && sub_t_num(j) >datenum(2010,1,1) && sub_t_num(j)<datenum(2016,1,1)
            %
            %re_rule1 = cat(1,re_rule1,[sub_t(j),sub_symbol]);
            sub_re{j} = [sub_t(j),sub_symbol]';
        end    
    end
    re_rule1{i} = {[sub_re{:}]};


    sprintf('%d-%d',i,T)
        
end
re_rule1 = [re_rule1{:}]';
rule_validation_update(re_rule1)
setpixelposition(gcf,[430,368,1008,420]);