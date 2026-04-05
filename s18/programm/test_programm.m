%开始回测

T = length(tref_sel);
y = zeros(size(tref_num));
%初始权重
ini_v = 1;
for i = 1:T
    if i < T
        sub_t = [tref_sel(i)+1,tref_sel(i+1)];
    else
        sub_t = [tref_sel(i)+1,tref_num(end)];
    end
    sub_symbol = symbol_all{i};
    
    t_ind_sel = tref_num>=sub_t(1)&tref_num<=sub_t(2);
    sub_tref = tref(t_ind_sel);
    sub_y  = get_single_GTA_bac_return(sub_symbol,sub_t,sub_tref);
    
    sub_y_a = ini_v/length(sub_symbol)*sum(cumprod(1+sub_y),2);    
    ini_v = sub_y_a(end);
    
    y(t_ind_sel) = sub_y_a;
    
    sprintf('%d-%d',i,T)
end

ind = tref_num<=datenum(2017,5,31);
sub_t = tref_num(ind);
sub_tref = tref(ind);
sub_y = y(ind);
sub_y(1) = 1;

x_ref = fetchmysql(['SELECT tradingdate,close FROM futuredata.indicator_data ',...
    'where symbols=''SHSE.000905'' order by tradingdate;'],2);

[~,ia,ib] = intersect(x_ref(:,1),sub_tref);
sub_y_ref = cell2mat(x_ref(ia,2));
sub_y = sub_y(ib);

sub_y_tref = sub_y_ref/sub_y_ref(1);
figure
plot([sub_y,sub_y_tref,sub_y-sub_y_tref+1],'LineWidth',2);

sub_x = sub_tref(ib);
sub_ind = floor(linspace(1,length(sub_y),20));
set(gca,'xtick',sub_ind);
set(gca,'xlim',[1,length(sub_y)]);
set(gca,'XTickLabel',sub_x(sub_ind));
set(gca,'XTickLabelRotation',45);
leg_strs = {'策略','中证500*持仓','相对强弱'};
legend(leg_strs,'location','northwest','NumColumns',length(leg_strs))



function sub_y  = get_single_GTA_bac_return(sub_symbol,sub_t,sub_tref)
    fee1 = 1/1000;
    fee2 = 2/1000;
    sql_str_m1 = ['select symbol,date(tradingdate),closeprice/precloseprice-1 from futuredata.STK_MKT_BWARDQUOTATION where symbol in(%s) ',...
    'and tradingdate>=''%s'' and tradingdate<=''%s'' and filling = 0 order by tradingdate;'];
    
    sub_str1 = sprintf('''%s''',strjoin(sub_symbol,''','''));
    sub_str2 = datestr(sub_t(1),'yyyy-mm-dd');
    sub_str3 = datestr(sub_t(2),'yyyy-mm-dd');
    
    sub_x = fetchmysql(sprintf(sql_str_m1,sub_str1,sub_str2,sub_str3),2);
    
    sub_y = zeros(length(sub_tref),length(sub_symbol));
    for j = 1:length(sub_symbol)
        sub_sub_x = sub_x(strcmp(sub_x(:,1),sub_symbol(j)),:);
        [~,ia] = intersect(sub_tref,sub_sub_x(:,2),'stable');
        sub_y(ia,j) = cell2mat(sub_sub_x(:,3));
    end
    sub_y(1,:) = sub_y(1,:)-fee1;%买入
    sub_y(end,:) = sub_y(end,:) - fee2;%卖出
end

