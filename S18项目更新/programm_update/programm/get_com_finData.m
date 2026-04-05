function [XF3,XF1,XF2] = get_com_finData(tN2,var2,var1,sel)
if nargin<4
    sel = [];
end
addpath('D:\OneDrive\guorongdu\adairPool')
if ~isempty(sel)
    sqlquery1 = ['select code,quarter,report_date,',var1,' from research..financeDataV2 where code = ''',sel,''' order by quarter'];
    XF1A = fetchsqlserver(sqlquery1,2);
    XF2 = get_fin_dataV2(tN2,var2,sel(3:end));
else
    sqlquery1 = ['select code,quarter,report_date,',var1,' from research..financeDataV2 order by code,quarter'];
    XF1A = fetchsqlserver(sqlquery1,2);
    XF2 = get_fin_dataV2(tN2,var2);
end
%XF1转化
[stockStr,~,ib] = unique(XF1A(:,1));
stockID = transStockID(stockStr);
[dateStr,~,ib2] = unique(XF1A(:,2));
dateID1 = datenum(dateStr,'yyyy-mm-dd');
[dateStr2,~,ib3] = unique(XF1A(:,3));
dateID2 = datenum(dateStr2,'yyyy-mm-dd');
XF1 = [stockID(ib),dateID1(ib2),dateID2(ib3),cell2mat(XF1A(:,4))];
%合并
XF3=[XF2;XF1];
[~,ia] = unique(XF3(:,1)*10^6+XF3(:,2));
XF3 = XF3(ia,:);
