%APM rankIC
clear
print_sel = true;
tN = 'S33.factor_cvar_month';
%tN = 'S33.factor_cvar_month_v2';
%

tN2 = 'yuqerdata.MktEqumAdjAfGet';

% end

window = 60;

%读取时间
tref = yq_methods.get_tradingdate('2015-12-01','2020-01-13');
tref_num = datenum(tref);
%获取月底日期
%last day for the month
month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
month_cut_date1 = tref(month_cut(:,1));
month_cut_date2 = tref(month_cut(:,2));

tref = month_cut_date2;
% sql_str = sprintf('select distinct(tradingdate) from %s order by tradingdate',tN);
% tref = fetchmysql(sql_str,2);
%tref = tref(datenum(tref)<=datenum(2016,5,31));
T = length(tref);
sql_str1 = 'select symbol,f_val1 from %s where tradingdate = ''%s''';
sql_str2 = 'select ticker,closeprice/openprice-1 from %s where enddate = ''%s''';
sql_str3 = 'select ticker from   yuqerdata.st_info where tradedate =''%s''';
sql_str4 = ['select ticker,listDate from yuqerdata.equget where listStatusCd !=''UN''',...
    'and listDate is not null']; 
symbol_info = fetchmysql(sql_str4,2);
symbol_listdate = datenum(symbol_info(:,2));
r = zeros(T,1);
parfor i = 1:T-1
    
    x1 = fetchmysql(sprintf(sql_str1,tN,tref{i}),2);
    x2 = fetchmysql(sprintf(sql_str2,tN2,tref{i+1}),2);
    
    %st
    st = fetchmysql(sprintf(sql_str3,tref{i}),2);
    st = cellfun(@str2double,st,'UniformOutput',false);
    st = cellfun(@(x) sprintf('%0.6d',x),st,'UniformOutput',false);
    [~,ia] = intersect(x1(:,1),st);
    x1(ia,:) = [];
    %上市未满 60 日的新股
    ind = datenum(tref{i})-symbol_listdate>window;
    [~,ia] = intersect(x1(:,1),symbol_info(ind,1));
    x1 = x1(ia,:);
    
    [~,ia,ib] = intersect(x1(:,1),x2(:,1));
    x1_v = cell2mat(x1(ia,2));
    x2_v = cell2mat(x2(ib,2));
    ia = isnan(x1_v+x2_v);
    r(i+1) = corr(x1_v(~ia),x2_v(~ia),'type','Spearman');
        
    if print_sel
        sprintf('%d-%d',i,T)
    end
        
end
% f = [tref,tref,num2cell(r)];
% f(1,:) = [];
% f(:,1) = {tN_key};
% id = datenum(f(:,2))>datenum(t0);
% f = f(id,:);
% if ~isempty(f)
%     conna = mysql_conn();
%     datainsert(conna,tN_w,var_info,f);
%     close(conna);
% end

t_str = tref;
T=length(t_str);
figure
bar(r)
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));

set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)    
setpixelposition(gcf,[223,365,1345,420]);
box off

