clear
print_sel = true;
tN = 'S33.factor_cvar_month';
%tN = 'S33.factor_cvar_month_v2';

tN2 = 'yuqerdata.MktEqumAdjAfGet';

fee = 3/1000;
window = 60;
g_num = 5;
g_str = cell(g_num+1,1);
for i = 1:g_num
    g_str{i} = sprintf('第%d组',i);
end
g_str{end} = '对冲组';
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
%tref = tref(datenum(tref)<=datenum(2016,5,31));
T = length(tref);
sql_str1 = 'select symbol,f_val1 from %s where tradingdate = ''%s''';
sql_str2 = 'select ticker,closeprice/openprice-1 from %s where enddate = ''%s''';
sql_str3 = 'select ticker from   yuqerdata.st_info where tradedate =''%s''';
sql_str4 = ['select ticker,listDate from yuqerdata.equget where listStatusCd !=''UN''',...
    'and listDate is not null']; 
symbol_info = fetchmysql(sql_str4,2);
symbol_listdate = datenum(symbol_info(:,2));
r = zeros(T,g_num);
r0 = zeros(T,1);
parfor i = 1:T-1
    
    %因子值
    x1 = fetchmysql(sprintf(sql_str1,tN,tref{i}),2);
    %未来一个月收益率
    x2 = fetchmysql(sprintf(sql_str2,tN2,tref{i+1}),2);
    
%     %st
%     st = fetchmysql(sprintf(sql_str3,tref{i}),2);
%     st = cellfun(@str2double,st,'UniformOutput',false);
%     st = cellfun(@(x) sprintf('%0.6d',x),st,'UniformOutput',false);
%     [~,ia] = intersect(x1(:,1),st);
%     x1(ia,:) = [];
%     
%     %上市未满 60 日的新股
%     ind = datenum(tref{i})-symbol_listdate>window;
%     [~,ia] = intersect(x1(:,1),symbol_info(ind,1));
%     x1 = x1(ia,:);
%     
    [~,ia,ib] = intersect(x1(:,1),x2(:,1));
    x1_v = cell2mat(x1(ia,2));
    x2_v = cell2mat(x2(ib,2));
    ia = isnan(x1_v+x2_v);
    
    x1_v(ia,:) = [];
    x2_v(ia,:) = [];
    x1(ia,:) = [];
    
    [~,ia] = sort(x1_v);
    sub_t = floor(length(ia)/g_num);
    for j = 1:g_num
        if ~eq(j,g_num)
            sub_w = (j-1)*sub_t+1:j*sub_t;
        else
            sub_w = (j-1)*sub_t+1:length(ia);
        end
        r(i+1,j) = mean(x2_v(ia(sub_w)));
    end
    r0(i+1) = mean(x2_v)-fee;    
    if print_sel
        sprintf('%d-%d',i,T)
    end
        
end

%%{
t_str = tref;
T=length(t_str);
r_c = cumprod(1+r);
r_2 = cumprod(1+r(:,end)-r(:,1));
figure
yyaxis  left
obj1 = plot(r_c,'-','LineWidth',2);
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
yyaxis right
obj2=plot(r_2,'-','LineWidth',2);
obj = [obj1;obj2];
%set color
color_para =[0 0.4470 0.7410;0.8500 0.3250 0.0980;0.9290 0.6940 0.1250;...
            0.4940 0.1840 0.5560;0.4660 0.6740 0.1880;...
            0.3010 0.7450 0.9330;0.6350 0.0780 0.1840];

for i = 1:size(obj,1)
    obj(i).Color = color_para(i,:);
end
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));

set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)    
setpixelposition(gcf,[223,365,1345,420]);
box off
legend(g_str,'NumColumns',g_num+1,'Location','best');

figure
bar(mean(r)-mean(r0))
setpixelposition(gcf,[223,365,1345,420]);

figure
plot(r_c,'LineWidth',2)
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)    
setpixelposition(gcf,[223,365,1345,420]);
box off
legend(g_str(1:end-1),'NumColumns',g_num,'Location','best');
%}