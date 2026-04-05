%Rule9：非经常性损益>行业均值+标准差*3 AND RANK(非经常性损益)<0.03
clear
M_start_time = datetime
tref = yq_methods.get_tradingdate('2010-01-01','2017-01-01');
sql_str = ['select symbol,f_val from S26.f_nrProfitLoss ',...
    'where tradingdate=''%s'' and f_val is not null'];

T = length(tref);
re_rule1 = cell(T,1);
parfor i = 1:T
    x = fetchmysql(sprintf(sql_str,tref{i}),2);
    obj_yq = yq_methods();
    indus_code = obj_yq.get_industry_class(tref{i});
    
    [~,ia,ib] = intersect(x(:,1),indus_code(:,1),'stable');
    X = [x(ia,:),indus_code(ib,end)];
    X_v = cell2mat(X(:,2:end));
    
    indus_code_u = unique(X_v(:,2));
    sub_T= length(indus_code_u);
    indus_mx = zeros(sub_T,1);
    indus_mx_all = zeros(size(X_v(:,2)));
    indus_rank = indus_mx_all;
    for j = 1:sub_T
        temp_ind = eq(X_v(:,2),indus_code_u(j));
        indus_mx(j) = mean(X_v(temp_ind,1))+std(X_v(temp_ind,1))*3; 
        indus_mx_all(temp_ind) = indus_mx(j);
        indus_rank(temp_ind) = prctile(X_v(temp_ind,1),0.03*100);
    end
    
    %ind = X_v(:,1)>indus_mx_all & X_v(:,1)<prctile(X_v(:,1),0.03*100);
    ind = X_v(:,1)>indus_mx_all & X_v(:,1)>indus_rank;
    if any(ind)
        temp = X(ind,[1,1]);
        temp(:,1) = tref(i);
        re_rule1{i} = temp';
    end
    
    sprintf('%d-%d',i,T)
end
re_rule1 = [re_rule1{:}]';
[~,ia ] = unique(re_rule1(:,2));
re_rule1 = re_rule1(ia,:);
% re_rule1 = [];
% for i = 1:T
%     if ~isempty(re{i})
%         re_rule1 = [re_rule1;re{i}];
%         [~,ia] = unique(re_rule1(:,2),'stable');
%         re_rule1 = re_rule1(ia,:);
%     end
%     
% end

rule_validation_update(re_rule1)

M_end_time = datetime