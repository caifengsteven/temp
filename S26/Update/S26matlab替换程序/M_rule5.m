%M_rule1
%{
YOY(主营业务收入)>0 AND YOY(应收账款)>0.5 AND YOY(应收账款)/ YOY(主营业务收入)>3
找不到一个
%}

clear
tN1 = 'yuqerdata.yq_FdmtBSGet';
%
window = 2;
obj_yq = yq_methods();
%主营业务收入，应收账款
x = obj_yq.get_HeBingLiRun('revenue-othGain'); %营业收入-其他收益
%x = obj_yq.get_YeJiKuaiBao('primeOperRev');
y = obj_yq.get_HeBingZiChanFuZhai('AR'); %应收账款
x = x(strcmp(x(:,5),'A'),:);
y = y(strcmp(y(:,5),'A'),:);
%合并数据
symbol_N = unique(x(:,1));
T = length(symbol_N);
z = cell(size(x,1),size(x,2)+1);
z_num = 0;
for i = 1:T
    sub_symbol = sprintf('%0.6d',str2double(symbol_N(i)));
    sub_x = x(strcmp(x(:,1),symbol_N(i)),:);
    sub_y = y(strcmp(y(:,1),sub_symbol),:);
    sub_x(:,2) = cellfun(@(x) x(1:10),sub_x(:,2),'UniformOutput',false);
    [~,ia,ib] = intersect(sub_x(:,3),sub_y(:,3),'stable');
    sub_z = [sub_x(ia,:),sub_y(ib,end)];
    sub_ind = z_num+1:z_num+size(sub_z,1);
    z(sub_ind,:) = sub_z;
    z_num = z_num+size(sub_z,1);
    sprintf('step 1: %d-%d',i,T)
end
z = z(1:z_num,:);

num=0;
num1 = 0;
re_rule1 = [];
for i = 1:T    
    sub_x = z(strcmp(z(:,1),symbol_N(i)),:);
    [~,ia] = unique(sub_x(:,3),'stable');
    sub_x = sub_x(ia,:);
    sub_symbol = sprintf('%0.6d',str2double(symbol_N(i)));
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
    sub_t = sub_t(ia);
    
    sub_T = length(sub_t);
    sub_num = 0;
    %YOY(主营业务收入)>0 AND YOY(应收账款)>0.5 AND YOY(应收账款)/ YOY(主营业务收入)>3
    for j = window:sub_T
        sub_YOY = sub_v(j,:)./sub_v(j-1,:)-1;
        if sub_YOY(1)>0 && sub_YOY(2)>0.5 && sub_YOY(2)/sub_YOY(1)>3 && sub_t_num(j) >datenum(2010,1,1) && sub_t_num(j)<datenum(2016,1,1)
            %
            sub_num=sub_num+1;
            re_rule1 = cat(1,re_rule1,[sub_t(j),sub_symbol]);
        end    
    end
    if sub_num>0
        num = num+1;
    end
    num1 = num1+sub_num;
    sprintf('%d-%d',i,T)
        
end

rule_validation(re_rule1)