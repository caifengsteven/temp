clear
%date 
%rate 0  select tradeDate,rate from yuqerdata.shibor_data where ticker = 'Shibor1M' and currency = 'CNY' order by tradeDate;
%
year_t = 244;
ticker1 = fetchmysql('select distinct tickerBond from yuqerdata.convertiblebond_dayprice where tradedate>=''2008-01-01''',2);

ticker2 = fetchmysql('select distinct Liscd from gtadata.BND_Ccbdinfo',2);
ticker2 = cellfun(@num2str,ticker2,'UniformOutput',false);

ticker20 = fetchmysql('select distinct symbol from yuqerdata.bond_impliedvol_wind',2);
ticker20 = cellfun(@num2str,ticker20,'UniformOutput',false);
ticker3 = intersect(ticker1,ticker2);

t_cut = fetchmysql('select Liscd,Matdt  from gtadata.BND_Ccbdinfo',2);
t_cut(:,1) = cellfun(@num2str,t_cut(:,1),'UniformOutput',false);
[~,ia,ib] = intersect(ticker3,t_cut(:,1),'stable');
t_cut = t_cut(ib,:);
t_cut_num = datenum(t_cut(:,end));

% tref = fetchmysql('select distinct tradedate from yuqerdata.convertiblebond_dayprice where tradedate>=''2008-01-01''',2);
shibor_3m = fetchmysql('select tradeDate,rate from yuqerdata.shibor_data where ticker = ''Shibor1M'' and currency = ''CNY'' order by tradeDate',2);
% [~,ia,ib] = intersect(tref,shibor_3m(:,1));
% shibor_3m_copy = zeros(size(tref));
% shibor_3m_copy(ia) = cell2mat(shibor_3m(ib,2));
% zero_ind = find(eq(shibor_3m_copy,0));
% for i = 1:length(zero_ind)
%     if ~eq(zero_ind(i),1)
%         shibor_3m_copy(zero_ind(i)) = shibor_3m_copy(zero_ind(i)-1);
%     else
%         shibor_3m_copy(zero_ind(i)) = 3/100;
%     end
%     
% end

%Êý¾ÝÈ±Ê§Ìî³ä

%setdiff(ticker1,ticker3)
T = length(ticker3);
sql_str1 = ['select tradeDate,closePriceEqu,convPrice,',...
    'closePriceBond,debtPuredebtRatio from yuqerdata.convertiblebond_dayprice ',...
    'where tickerBond=''%s'' order by tradeDate'];
sql_str2 = ['select tradingdate,f_val ',...
    'from yuqerdata.bond_debtpuredebtratio_wind ',...
    'where symbol =''%s'' order by tradingdate'];
re = cell(T,1);

for i = 1:T
    p1 = fetchmysql(sprintf(sql_str1,ticker3{i}),2);   
    p2 = fetchmysql(sprintf(sql_str2,ticker3{i}),2);
    if isempty(p1) || isempty(p2)
        continue
    end
    [~,ia,ib] = intersect(p1(:,1),p2(:,1));
    p3 = [p1(ia,:),p2(ib,end)];
    if isempty(p3)
        continue
    end
    [~,ia,ib] = intersect(p3(:,1),shibor_3m(:,1));
    sub_shibor_3m = cell2mat(shibor_3m(ib,2));
    p3 = p3(ia,:);
    if isempty(p3)
        continue
    end
    nan_ind = isnan(cell2mat(p3(:,end-1)));
    p3(nan_ind,end-1) = p3(nan_ind,end);
    
    P = cell2mat(p3(:,2:end-1));
    nan_ind = isnan(P(:,end)) | P(:,3)-P(:,4)<0;
    P(nan_ind,:) = [];
    p3(nan_ind,:) = [];
    sub_shibor_3m(nan_ind,:) = [];
    P = [P(:,1:2),P(:,3)-P(:,4)];
    dt = (t_cut_num(i)-datenum(p3(:,1)))/year_t;
    
    sub_T = length(dt);
    sub_Volatility = blsimpv(P(:,1),P(:,2), sub_shibor_3m, dt,P(:,3),[], [], [], {'call'});
    sub_re = p3(:,1:3);
    sub_re(:,2) = ticker3(i);
    sub_re(:,3) = num2cell(sub_Volatility);
    del_ind = isnan(sub_Volatility);
    sub_re(del_ind,:) = [];
    
    if ~isempty(sub_re)
        re{i} = sub_re;
    end
    sprintf('%d-%d',i,T)
    
end


