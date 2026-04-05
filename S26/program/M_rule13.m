%Rule13：((YOY(主营业务收入)-YOY(净利润)) / abs(YOY(净利润)) > 10
% 主营业务收入同比    业绩快报 DataAPI.FdmtEeGet primeOperRev %全部为empty value
% 净利润同比 业绩快报 DataAPI.FdmtEeGet  NIncomeAttrP
% 更换数据
% 合并利润表 主营业务 obj_yq.get_HeBingLiRun('revenue-othGain'); %营业收入-其他收益 
% 净利润  NIncome
% 数据还是missing
% 主营业务收入使用国泰安数据

clear
window = 2;
XF1 = yq_methods.get_HeBingLiRun('NIncome');
XF1 = XF1(strcmp(XF1(:,5),'A'),:);
del_ind = cellfun(@isnan,XF1(:,end));
XF1(del_ind,:) = [];

XF2 = fetchmysql('select Stkcd,Accper,B110101 from gtadata.FAR_Finidx where B110101 is not null and B110101!=0 order by Accper,Annodt',2);
%合并数据
symbols = unique(XF1(:,1));
T = length(symbols);
re_rule1 = cell(T,1);
parfor i = 1:T
    sub_symbol = sprintf('%0.6d',str2double(symbols(i)));
    sub_xf1 = XF1(strcmp(XF1(:,1),symbols(i)),:);
    sub_xf1 = flipud(sub_xf1);
    sub_xf2 = XF2(strcmp(XF2(:,1),sub_symbol),:);
    
    [~,ia,ib] = intersect(sub_xf1(:,3),sub_xf2(:,2),'stable');
    
    sub_xf12 = [sub_xf1(ia,:),sub_xf2(ib,end)];
    
    if isempty(sub_xf12)
        continue
    end
    [~,ia] = unique(sub_xf12(:,3),'stable');
    sub_xf12 = sub_xf12(ia,:);
    sub_v = cell2mat(sub_xf12(:,end-1:end));
    sub_T = size(sub_v,1);
    sub_re = cell(sub_T,1);
    for j = window:sub_T
        sub_YOY = sub_v(j,:)./sub_v(j-1,:)-1;
        %((YOY(主营业务收入)-YOY(净利润)) / abs(YOY(净利润)) > 10
        if (sub_YOY(1)-sub_YOY(2))/abs(sub_YOY(2)) > 10
            sub_re{j} = [sub_xf12(j,2),sub_symbol]';
            %re_rule1 = cat(1,re_rule1,sub_re);
        end
    end
    re_rule1{i} = [sub_re{:}];
    sprintf('%d-%d',i,T)
end
re_rule1 = [re_rule1{:}]';
ia = datenum(re_rule1(:,1));
ia = ia>=datenum(2010,1,1) & ia<=datenum(2017,1,1);
re_rule1 = re_rule1(ia,:);

rule_validation_update(re_rule1)
setpixelposition(gcf,[430,368,1008,420]);
