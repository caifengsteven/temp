%M_rule1
%{
RULE1：(净利润/median(净利润,3))<0.2 AND 净利润>0 AND 净利润>10e6 AND IF(资
产重组,2)=1
净利润是归属母公司的净利润（不含少数股东
权益），median(净利润,4)代表过去四个会计年度的净利润中位数，IF 是判断函数，IF(资
产重组,2)是指过去两年是否有发公告表示公司及其子公司有重组行为。
%}

clear
tN1 = 'S26.FdmtBSAllLatestGet';
%
window = 3;
obj_yq = yq_methods();
x = obj_yq.get_HeBingLiRun('NIncomeAttrp');
x = x(strcmp(x(:,5),'A'),:);

sql_str = 'SELECT ticker,publishDate FROM s26.equrestructuringget where isSucceed=1 order by publishDate';
y=fetchmysql(sql_str,2);

symbol_N = unique(x(:,1));
T = length(symbol_N);
num=0;
num1 = 0;
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
    sub_v = cell2mat(sub_x(:,6));
    [sub_t_num,ia] = sort(sub_t_num);
    sub_v = sub_v(ia,:);
    sub_t = sub_t(ia);
    
    sub_T = length(sub_t);
    sub_re = cell(sub_T,1);
    for j = window+1:sub_T
        sub_window_v = sub_v(j-window:j-1);
        if sub_v(j)/median(sub_window_v)<0.2 && sub_v(j)<10e6 && sub_v(j)>0 &&sum(sub_y_t>=sub_t_num(j)-365*2&sub_y_t<=sub_t_num(j))>0 && sub_t_num(j) >datenum(2010,1,1) && sub_t_num(j)<datenum(2016,1,1)
            %
            %re_rule1 = cat(1,re_rule1,[sub_t(j),sub_symbol]);
            sub_re{j} = [sub_t(j),sub_symbol]';
        end    
    end
    
    re_rule1(i) = {[sub_re{:}]};

    sprintf('%d-%d',i,T)
        
end
re_rule1 = [re_rule1{:}]';
%rule_validation(re_rule1)
rule_validation_update(re_rule1)
setpixelposition(gcf,[430,368,1008,420]);