%增加季度信息
%nan值赋值为0
function [fX2,sub_stockID2,t_num3] = get_fin_dataV2(tN2,var2,sel,nandeal)
if nargin <4
    nandeal = 0;
end
if nargin < 3
    sel = [];
end
if ~iscell(var2)
    var2 = {var2};
end
if length(var2)>1
    sprintf('数据错误')
    keyboard
end
var2 = {'Symbol','enddate','ActRelDate',var2{1}};
%联合查询
k1 = cellfun(@(x) ['a.',x],var2,'UniformOutput',false);
k1{3}(1) = 'b';
if strcmpi(tN2,'gtadata.STK_FIN_Balance')
    limitStr = ' and a.ReportTypeID in(1,5,6,7)';
elseif any(strcmpi(tN2,{'gtadata.STK_FIN_Income','gtadata.STK_FIN_CashFlow',...
        'gtadata.STK_FIN_CashFlowIndrect'}))
    limitStr = ' and a.ReportTypeID in(1,2,3,4)';
else
    limitStr = [];
end

if isempty(sel)
    sqlquery = ['select ',strjoin(k1,','),' from ',tN2,' a',10,...
        'inner join gtadata.STK_FIn_RelForcDate b on a.enddate = b.AccouPeri and a.InstitutionID=b.ListedCoID',10,...
        'where a.StateTypeCode = ''A'' ',limitStr,'  and b.ActRelDate is not null',10,...
        'order by a.enddate,a.Symbol'];
else
    sqlquery = ['select ',strjoin(k1,','),' from ',tN2,' a',10,...
        'inner join gtadata.STK_FIn_RelForcDate b on a.enddate = b.AccouPeri and a.InstitutionID=b.ListedCoID',10,...
        'where a.StateTypeCode = ''A'' ',limitStr,'  and b.ActRelDate is not null',10,...
        'and a.Symbol =''',sel,'''',10,...
        'order by a.enddate,a.Symbol'];
end
[X2,OK4] = fetchmysql(sqlquery,2);
if ~isempty(X2)
    fX2 = zeros(size(X2));
    [t_str3,ia3,ic3] = unique(X2(:,2));
    t_num3 = datenum(t_str3,'yyyy-mm-dd');
    fX2(:,2) = t_num3(ic3);

    [t_str3b,ia3b,ic3b] = unique(X2(:,3));
    t_num3b = datenum(t_str3b,'yyyy-mm-dd');
    fX2(:,3) = t_num3b(ic3b);

    [stock_num4,ia4,ic4] = unique(X2(:,1));
    sub_stockID2 = cellfun(@str2double,stock_num4);
    sub_stockID2  = stockNum2ID(sub_stockID2);
    fX2(:,1) = sub_stockID2(ic4);
    fX2(:,4) = cell2mat(X2(:,4));
    fX2 = fX2(fX2(:,1)<33000000,:);
    sub_stockID2 = sub_stockID2(sub_stockID2<33000000);
    if eq(nandeal,0)
        fX2(isnan(fX2(:,end))|isinf(fX2(:,end)),:) = [];
    else
        fX2(isinf(fX2(:,end)),:) = [];
    end
else
    fX2 = [];
    sub_stockID2= [];
    t_num3= [];
end