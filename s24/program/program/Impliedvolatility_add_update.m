function [tref,Y_m] = Impliedvolatility_add_update()
%wind 可转债隐含波动率合成指数 
%
tref = fetchmysql('SELECT distinct(tradingdate) FROM yuqerdata.bond_impliedvol_wind where tradingdate >=''2008-01-01'' order by tradingdate',2);

T = length(tref);
sql_str = 'select symbol,f_val from yuqerdata.bond_impliedvol_wind where tradingdate =''%s''';
sql_str1 = 'select tickerBond from yuqerdata.convertiblebond_dayprice where tradedate =''%s'' and 100/convPrice*closePriceEqu>=95 and 100/convPrice*closePriceEqu<=120';
y = zeros(T,1);
parfor i = 1:T
    sub_x = fetchmysql(sprintf(sql_str,tref{i}),2);
    sub_x2 = fetchmysql(sprintf(sql_str1,tref{i}),2);
    if ~isempty(sub_x2)
        sub_x(:,1) = cellfun(@num2str,sub_x(:,1),'UniformOutput',false);
        [~,ia] = intersect(sub_x(:,1),sub_x2(:,1));
        if ~isempty(ia)
            sub_x = cell2mat(sub_x(ia,2));
        else
            sub_x = cell2mat(sub_x(:,2));
        end
    else
        sub_x = cell2mat(sub_x(:,2));
    end
    sub_x(isnan(sub_x)) = [];
    y(i) = mean(sub_x);
    sprintf('%d-%d',i,T)
end

%y_ci = y;
%平滑
y_ci = movmean(y,[20,0])/100;

%50指数实际波动率
x = fetchmysql(['SELECT tradedate,closeIndex/precloseIndex-1 FROM yuqerdata.yq_index ',...
    'where symbol = ''000016'' and tradedate>=''2008-01-01'' order by tradedate'],2);

y2 = cell2mat(x(:,2));
y1 = movstd(y2,[15,0]);

y_50 = movmean(y1*sqrt(244),[20,0]);
%y_50 =  y1*sqrt(244);

%数据对齐
[~,ia,ib] = intersect(tref,x(:,1));
tref = tref(ia);
Y = [y_ci(ia),y_50(ib)];
%计算修正系数
Y_m = Y(:,2)./Y(:,1);
