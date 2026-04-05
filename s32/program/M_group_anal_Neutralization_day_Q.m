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
warning('off')
%search symbol
symbol_pool = cell(T,g_num);
for i = 1:T
    %因子值
    x1 = fetchmysql(sprintf(sql_str1,tN,tref{i}),2);   
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
   
    x1_v = cell2mat(x1(:,2));
    ia = isnan(x1_v);
    
    x1_v(ia,:) = [];
    x1(ia,:) = [];
    
    [~,ia] = sort(x1_v);
    sub_t = floor(length(ia)/g_num);
    for j = 1:g_num
        if ~eq(j,g_num)
            sub_w = (j-1)*sub_t+1:j*sub_t;
        else
            sub_w = (j-1)*sub_t+1:length(ia);
        end
        symbol_pool{i,j} = x1(ia(sub_w),1);
    end   
    if print_sel
        sprintf('中性化分组测试-选股：%d-%d',i,T)
    end
        
end
warning('on')
tref0 = yq_methods.get_tradingdate(tref{1},'2020-01-13');
tref0 = tref0(2:end);
tref0_num = datenum(tref0);
tref_num = datenum(tref);
%
T = length(tref0);
r = cell(T,1);
r2 = r;
sql_str6 = 'select symbol,chgPct from yuqerdata.yq_dayprice where tradeDate = ''%s'' and chgPct is not null';
parfor i = 1:T
    %获取symbol
    ind = find(tref_num<tref0_num(i),1,'last');
    sub_symbol = symbol_pool(ind,:);
    %获取当日数据
    x1 = fetchmysql(sprintf(sql_str6,tref0{i}),2);
    %保存
    sub_r = zeros(g_num,1);
    sub_r2 = sub_r;
    for j = 1:g_num
        temp = sub_symbol{j};
        sub_y = zeros(size(temp));
        [~,ia,ib]=intersect(temp,x1(:,1));
        sub_y(ia) = cell2mat(x1(ib,2));
        sub_r(j) = mean(sub_y);
        sub_r2(j) = mean(sub_y(ia));
    end
    r{i} = sub_r;
    r2{i} = sub_r2;
    if print_sel
        sprintf('中性化分组测试-每日统计：%d-%d',i,T)
    end
end
r = [r{:}]';
r2 = [r2{:}]';

[~,ia,ib] = intersect(tref0,tref);
ia=[0;ia];
ia = ia + 1;
fee_all = zeros(size(r(:,1)));
fee_all(ia) = fee;
%%{
t_str = tref0;
T=length(t_str);
r_c = cumprod(1+r);
r_2 = cumprod(1+r(:,1)-r(:,end)-fee_all);

r_c_2 = cumprod(1+r2);
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


ind =datenum(tref0)<=datenum(2016,5,31);
t_str = tref0(ind);
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

figure
bpcure_plot_updateV2(tref0(ind),r_2(ind))
setpixelposition(gcf,[223,365,1345,420]);
%}
sql_str = 'select tradeDate, closeIndex from yuqerdata.yq_index where symbol = ''000905'' order by tradeDate';
x = fetchmysql(sql_str,2);
[~,ia] = intersect(x(:,1),tref0,'stable');
x = cell2mat(x(ia,2));
figure
plot([r_c(:,1:2),x/x(1)],'LineWidth',2)
t_str = tref0;
T=length(t_str);
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)  
setpixelposition(gcf,[223,365,1345,420]);
legend({'SMART组合','SMART组合2','中证500'})

figure
plot([r_c(ind,1:2),x(ind)/x(1)],'LineWidth',2)
t_str = tref0(ind);
T=length(t_str);
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)  
setpixelposition(gcf,[223,365,1345,420]);
legend({'SMART组合','SMART组合2','中证500'})
grid on
%%%%%%%%%%%
r_x = zeros(size(x));
r_x(2:end) = x(2:end)./x(1:end-1)-1;
r_c2 = cumprod(1+r(:,2)-r_x);
figure
bpcure_plot_updateV2(tref0,r_c2)
setpixelposition(gcf,[223,365,1345,420]);
%%%%%%%%%%%%%%%%%%
figure
bpcure_plot_updateV2(tref0(ind),r_c2(ind))
setpixelposition(gcf,[223,365,1345,420]);