%M_rule8
%{
Rule8：其他流动资产/流动资产>80% OR YOY（其他流动资产/流动资产）>10
%}

clear
%
window = 2;
obj_yq = yq_methods();
%净利润，营业收入
x = obj_yq.get_HeBingZiChanFuZhai('othCA,TCA');

symbol_N = unique(x(:,1));
T = length(symbol_N);
num=0;
num1 = 0;
re_rule1 = cell(T,1);
parfor i = 1:T    
    sub_x = x(strcmp(x(:,1),symbol_N(i)),:);
    [~,ia] = unique(sub_x(:,3),'stable');
    sub_x = sub_x(ia,:);
    sub_symbol = symbol_N(i);
    ia = sum(cell2mat(sub_x(:,6:7)),2);
    sub_x(isnan(ia),:) = [];
    if isempty(sub_x)
        continue
    end
    sub_t = sub_x(:,2);
    sub_t_num = datenum(sub_t);
    sub_v = cell2mat(sub_x(:,6:7));
    [sub_t_num,ia] = sort(sub_t_num);
    sub_v = sub_v(ia,:);
    sub_v = sub_v(:,1)./sub_v(:,2);
    sub_t = sub_t(ia);
    
    sub_T = length(sub_t);
    sub_num = 0;
    sub_re = cell(sub_T,1);
    for j = window:sub_T
        sub_test1 = sub_v(j);
        sub_test2 = sub_v(j)/sub_v(j-1)-1;
        if sub_test1>0.8 &&sub_test2>10 && sub_t_num(j) >datenum(2010,1,1) && sub_t_num(j)<datenum(2016,1,1)
            sub_re{j} = [sub_t(j),sub_symbol]';
            %sub_num=sub_num+1;
            %re_rule1 = cat(1,re_rule1,[sub_t(j),sub_symbol]);
        end    
    end
    re_rule1{i} = [sub_re{:}];
    sprintf('%d-%d',i,T)
        
end
re_rule1 = [re_rule1{:}]';

rule_validation(re_rule1)