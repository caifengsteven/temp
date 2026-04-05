%Rule3：实际当年利息率 - 计算利息率>1%
%太多
%计算利息率 = 冲减财务费用的利息收入/过去一年平均货币资金余额（使用过去四个季度资产负债表均值计算得到）。
%财务费用 finanExp 合并利润表 DataAPI.FdmtISGet
%货币资金 cashCEquiv 合并资产负债表 FdmtBSGet
clear
XF1 = yq_methods.get_HeBingLiRun('finanExp');
XF2 = yq_methods.get_HeBingZiChanFuZhai('cashCEquiv');

del_ind1 = cellfun(@isnan,XF1(:,end));
XF1(del_ind1,:) = [];
XF1 = XF1(strcmp(XF1(:,5),'A'),:);
del_ind2 = cellfun(@isnan,XF2(:,end));
XF2(del_ind2,:) = [];

symbols = unique(XF1(:,1));
T = length(symbols);
re = cell(T,1);
parfor i = 1:T
    sub_xf1 = XF1(strcmp(XF1(:,1),symbols(i)),:);
    sub_xf1 = flipud(sub_xf1);
    sub_symbol = sprintf('%0.6d',str2double(symbols{i}));
    sub_xf2 = XF2(strcmp(XF2(:,1),sub_symbol),:);
    sub_xf2 = flipud(sub_xf2);
    [~,ia] = unique(sub_xf2(:,3),'stable');
    sub_xf2 = sub_xf2(ia,:);
    sub_xf2_v = cell2mat(sub_xf2(:,end));
    %sub_xf2_v = diff(sub_xf2_v);
    %sub_xf2 = sub_xf2(2:end,:);
    sub_xf2_v = movmean(sub_xf2_v,[4-1,0]);
    sub_xf2(:,end) = num2cell(sub_xf2_v);
    
    [~,ia,ib] = intersect(sub_xf1(:,3),sub_xf2(:,3));
    sub_xf12 = [sub_xf1(ia,:),sub_xf2(ib,end)];
    sub_v = cell2mat(sub_xf12(:,end-1))./cell2mat(sub_xf12(:,end));
    
    ind = 1.35 - abs(sub_v)*100>1;
    if any(ind)
        sub_re = sub_xf2(ind,[2,2]);
        sub_re(:,2) = {sub_symbol};
        re{i} = sub_re';
    end
    sprintf('step1: %d-%d',i,T)
end
% for i = 1:T
%     if ~isempty(re{i})
%         re_rule1 = cat(1,re_rule1,re{i});
%     end
%     sprintf('step2: %d-%d',i,T)
% end
re_rule1 = [re{:}]';

ia = datenum(re_rule1(:,1));
ia = ia>=datenum(2010,1,1) & ia<=datenum(2017,1,1);
re_rule1 = re_rule1(ia,:);

rule_validation_update(re_rule1)
setpixelposition(gcf,[430,368,1008,420]);