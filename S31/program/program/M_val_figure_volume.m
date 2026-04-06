clear
sql_str = ['select hour(tradingdate)*100+minute(tradingdate),volume,(closeprice-iopv)/iopv from ',...
    'S31.adj_data where date(tradingdate) = ''%s'' and symbol = ''%s'''];

tN = 'S31.adj_data';
code_pool = {'510050','510300','510500'};
code_name_pool = {'etf50_min','etf300_min','etf500_min'};
var_info = {'symbol','tradingdate','iopv','openprice','closeprice','volume'};

tref = yq_methods.get_tradingdate('2017-01-13','2020-01-13');
T_tref = length(tref);

code_sel = code_pool{1};
%sub_t = zeros(245,1);
%y = nan(245,T_tref);
re = cell(T_tref,1);
parfor i = 1:T_tref
 
    sub_x = fetchmysql(sprintf(sql_str,tref{i},code_sel));
    %sub_x = [sub_x;1531,0]
    sub_x(:,2:3) = zscore(sub_x(:,2:3));
    re{i} = sub_x;
    sprintf('%d-%d',i,T_tref)
end

for i = 1:T_tref
    if eq(i,1)
        sub_t = re{i}(:,1);
    else
        sub_t = unique([sub_t;re{i}(:,1)]);
    end
end

y = nan(length(sub_t),T_tref);
y1 = y;
for i = 1:T_tref
    [~,ia,ib] = intersect(sub_t,re{i}(:,1));
    y(ia,i) = re{i}(ib,2);
    y1(ia,i) = re{i}(ib,3);
end


yf = zeros(size(sub_t));
yf1 = yf;
for i = 1:length(yf)
    sub_x = y(i,:);
    yf(i) = mean(sub_x(~isnan(sub_x)));
    sub_x = y1(i,:);
    yf1(i) = mean(sub_x(~isnan(sub_x)));
end

T = length(yf);

subplot(1,2,1)
plot(yf,'LineWidth',3)
hold on
plot([0,T],[0,0],'r-.');
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
t_str = cellfun(@num2str,num2cell(sub_t),'UniformOutput',false);
set(gca,'XTickLabel',t_str);
set(gca,'XTickLabelRotation',90)    

subplot(1,2,2)
plot(yf1,'LineWidth',3)
hold on
plot([0,T],[0,0],'r-.');
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
t_str = cellfun(@num2str,num2cell(sub_t),'UniformOutput',false);
set(gca,'XTickLabel',t_str);
set(gca,'XTickLabelRotation',90)    

setpixelposition(gcf,[223,365,1345,420*2]);
movegui(gcf,'center')

figure;
plot(yf1,'LineWidth',3);