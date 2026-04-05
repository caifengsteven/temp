%{
Rule7：其他应收款/流动资产>10% AND YOY（其他应收款/流动资产）>10 AND 归母净利润>0

NIncomeAttrP	float	归属于母公司的净利润 合并现金流量表 (Point in time)

#合并资产负债表 (Point in time) 
#TCA	float	流动资产合计
#othReceiv	float	其他应收款
get 2 
%}

clear

window = 2;
obj_yq = yq_methods();
x = obj_yq.get_HeBingLiRun('NIncomeAttrP'); %归属于母公司的净利润
%x = obj_yq.get_YeJiKuaiBao('primeOperRev');
y = obj_yq.get_HeBingZiChanFuZhai('othReceiv,TCA'); %其他应收款，流动资产
x = x(strcmp(x(:,5),'A'),:);
y = y(strcmp(y(:,5),'A'),:);
%合并数据
xid = cellfun(@(x,y) [x,',',y],x(:,1),x(:,3),'UniformOutput',false);
yid = cellfun(@(x,y) [x,',',y],y(:,1),y(:,3),'UniformOutput',false);
[~,ia,ib] = intersect(xid,yid);
z = [x(ia,:),y(ib,end-1:end)];
symbol_N = unique(z(:,1));

T = length(symbol_N);
% z = cell(size(x,1),size(x,2)+2);
% z_num = 0;
% for i = 1:T
%     sub_symbol = sprintf('%0.6d',str2double(symbol_N(i)));
%     sub_x = x(strcmp(x(:,1),symbol_N(i)),:);
%     sub_y = y(strcmp(y(:,1),sub_symbol),:);
%     sub_x(:,2) = cellfun(@(x) x(1:10),sub_x(:,2),'UniformOutput',false);
%     [~,ia,ib] = intersect(sub_x(:,3),sub_y(:,3),'stable');
%     sub_z = [sub_x(ia,:),sub_y(ib,end-1:end)];
%     sub_ind = z_num+1:z_num+size(sub_z,1);
%     z(sub_ind,:) = sub_z;
%     z_num = z_num+size(sub_z,1);
%     sprintf('step 1: %d-%d',i,T)
% end
% z = z(1:z_num,:);

re_rule1 = cell(T,1);
parfor i = 1:T    
    sub_x = z(strcmp(z(:,1),symbol_N(i)),:);
    [~,ia] = unique(sub_x(:,3),'stable');
    sub_x = sub_x(ia,:);
    sub_symbol = sprintf('%0.6d',str2double(symbol_N(i)));
    ia = sum(cell2mat(sub_x(:,6:end)),2);
    sub_x(isnan(ia),:) = [];
    if isempty(sub_x)
        continue
    end
    sub_t = sub_x(:,2);
    sub_t_num = datenum(sub_t);
    sub_v = cell2mat(sub_x(:,6:end));
    [sub_t_num,ia] = sort(sub_t_num);
    sub_v = sub_v(ia,:);
    sub_t = sub_t(ia);
    
    sub_T = length(sub_t);
    sub_num = 0;
    %其他应收款/流动资产>10% AND YOY（其他应收款/流动资产）>10 AND 归母净利润>0
    %归属于母公司的净利润,其他应收款，流动资产 
    sub_re = cell(sub_T,1);
    for j = window:sub_T
        
        sub_test1 = sub_v(j,2)/sub_v(j,3);
        sub_test2 = sub_v(:,2)./sub_v(:,3);
        sub_test2 = sub_test2(end)/sub_test2(end-1)-1;        
        sub_test3 = sub_v(j,1);
        
        if sub_test1>0.1 && sub_test2>10 && sub_test3>0 && sub_t_num(j) >datenum(2010,1,1) && sub_t_num(j)<datenum(2016,1,1)
            %
            sub_re{j} = [sub_t(j),sub_symbol]';
            %sub_num=sub_num+1;
            %re_rule1 = cat(1,re_rule1,[sub_t(j),sub_symbol]);
        end    
    end
    re_rule1{i} = [sub_re{:}];
    sprintf('%d-%d',i,T)
        
end
re_rule1 = [re_rule1{:}]';
rule_validation_update(re_rule1)