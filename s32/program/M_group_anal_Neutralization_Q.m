%增加中性化处理
%月度
clear
print_sel = true;
tN = 'S32.factor_q';
tN2 = 'yuqerdata.MktEqumAdjAfGet';
fee = 3/1000;
window = 60;
g_num = 5;
g_str = cell(g_num+1,1);
for i = 1:g_num
    g_str{i} = sprintf('第%d组',i);
end
g_str{end} = '对冲组';
sql_str = sprintf('select distinct(tradingdate) from %s order by tradingdate',tN);
tref = fetchmysql(sql_str,2);
%tref = tref(datenum(tref)<=datenum(2016,5,31));
T = length(tref);
sql_str1 = 'select symbol,f_val from %s where tradingdate = ''%s''';
sql_str2 = 'select ticker,closeprice/openprice-1 from %s where enddate = ''%s''';
sql_str3 = 'select ticker from   yuqerdata.st_info where tradedate =''%s''';
sql_str4 = ['select ticker,listDate from yuqerdata.equget where listStatusCd !=''UN''',...
    'and listDate is not null']; 
sql_str5 = 'select symbol,f_val,log(f_val2) from S32.ret20d where tradingdate = ''%s''';
symbol_info = fetchmysql(sql_str4,2);
symbol_listdate = datenum(symbol_info(:,2));
r = zeros(T,g_num);
r0 = zeros(T,1);
warning('off')
for i = 1:T-1
    
    %因子值
    x1 = fetchmysql(sprintf(sql_str1,tN,tref{i}),2);
    %未来一个月收益率
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
    %中性化步骤
    x_f = fetchmysql(sprintf(sql_str5,tref{i}),2);
    x_indus = yq_methods.get_industry_class_2(tref{i});
    inds = suscc_intersect({x1(:,1),x_f(:,1),x_indus(:,1)});
    x1 = x1(inds(:,1),:);
    y = cell2mat(x1(:,2));
    f = cell2mat([x_f(inds(:,2),2:end),x_indus(inds(:,3),2:end)]);
    
    dummy_f = yq_methods.trans_dummy(f(:,end));
    
    [~,~,y] = regress(y,[ones(size(y)),f(:,1:2),dummy_f]); 
    x1(:,end) = num2cell(y);
    
    [~,ia,ib] = intersect(x1(:,1),x2(:,1));
    x1 = x1(ia,:);
    x2 = x2(ib,:);
    x1_v = cell2mat(x1(:,2));
    x2_v = cell2mat(x2(:,2));
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
        r(i,j) = mean(x2_v(ia(sub_w)));
    end
    r0(i) = mean(x2_v);    
    if print_sel
        sprintf('%d-%d',i,T)
    end
        
end
warning('on')
%%{
t_str = tref;
T=length(t_str);
r_c = cumprod(1+r);
r_2 = cumprod(1+r(:,1)-r(:,end)-fee);
figure
yyaxis  left
plot(r_c,'LineWidth',2)
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
yyaxis right
plot(r_2,'LineWidth',2)
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));

set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)    
setpixelposition(gcf,[223,365,1345,420]);
box off
legend(g_str,'NumColumns',g_num+1,'Location','best');


ind =datenum(tref)<=datenum(2016,5,31);
t_str = tref(ind);
T=length(t_str);
figure
yyaxis left
plot(r_c(ind,:),'LineWidth',2)
set(gca,'xlim',[0,T+0.5]);
set(gca,'XTick',floor(linspace(1,T,15)));
yyaxis right
plot(r_2(ind),'LineWidth',2)
set(gca,'xlim',[0,T+0.5]);
set(gca,'XTick',floor(linspace(1,T,15)));

set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)    
setpixelposition(gcf,[223,365,1345,420]);
box off
%}