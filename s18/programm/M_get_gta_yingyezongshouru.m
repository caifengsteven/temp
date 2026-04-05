%获取营业收入
clear
tN2 = 'gtadata.STK_FIN_Income';
var2 = {'B0011'};
%var3 = {'net_profit'};
%XF0 = get_com_finData(tN2,var2{1},var3{1});
XF0  = get_fin_dataV2(tN2,var2);
XF = num2cell(XF0);
XF(:,1) = cellfun(@num2str,XF(:,1),'UniformOutput',false);
XF(:,1) = cellfun(@(x) x(3:end),XF(:,1),'UniformOutput',false);
XF(:,2) = cellstr(datestr(XF0(:,2),'yyyy-mm-dd'));
XF(:,3) = cellstr(datestr(XF0(:,3),'yyyy-mm-dd'));

x = cell2table(XF,'VariableNames',{'symbol','date1','date2','ys_value'});
writetable(x,'yingyezongshouru_val.csv');