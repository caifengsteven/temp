%IC ICIR t_value 
%got date and cal IC
%���Ի�
%�޶���Ʊ��
clear
close all
%parameters

dn = 'S23';
tns = {'zhubifactor_volumebig'};
tn = tns{1};

%�Ƿ����Ի� 0��1��
neutralization_sel = 1;
%��Ʊ��ѡ��
symbol_pool_all = {[],'000300','000905','000985','000852'};
symbol_pool_info = {'ȫ�г�','300��Ʊ��','500��Ʊ��','��֤ȫָ��Ʊ��','��֤1000��Ʊ��'};
symbol_pool=symbol_pool_all{1};
%������
fee_pool = [0,1/1000+2/10000];
fee = fee_pool(1);
%��������

paretol_pool = {'BB1','BS1','minus_BS1','sum_BS1',...
    'BB2','BS2','minus_BS2','sum_BS2','focus_B','focus_S','minus_fBS','sum_fBS'};
f_sel_ind = 1;
sql_str_f = ['select symbol,',paretol_pool{f_sel_ind},' from %s where tradingdate = ''%s'''];

tn_fullname = sprintf('%s.%s',dn,tn);

tn_1month_return = 'yuqerdata.future_return_1m';

dn_yq = 'yuqerdata';
tn_yq = 'yq_dayprice';
tn_yq_fullname = sprintf('%s.%s',dn_yq,tn_yq);
%month_cut
t0 = '2013-02-01';
tt = '2019-01-01';
sql_str = 'select distinct tradedate from %s order by tradedate';
tref = fetchmysql(sprintf(sql_str,tn_yq_fullname),2);
tref_num = datenum(tref);
ind_cut = tref_num>=datenum(t0)&tref_num<=datenum(tt);
tref_num = tref_num(ind_cut);
tref = tref(ind_cut);

month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
T_month_cut = size(month_cut,1);
month_cut_date = tref(month_cut(:,2));

sql_str_fr = 'select symbol,f_val from %s where tradingdate = ''%s''';
ic = zeros(T_month_cut,1);
p = ic;
Y = cell(T_month_cut,1);
parfor i = 1:T_month_cut
    %factor data
    x = fetchmysql(sprintf(sql_str_f,tn_fullname,month_cut_date{i}),2);        
    %��Ʊ���޶�
    if ~isempty(symbol_pool)
        
        sub_t = fetchmysql(sprintf(...
            'select tradingdate from s22.index_com where tradingdate < ''%s'' and ticker = ''%s'' order by tradingdate limit 1',...
            month_cut_date{i},symbol_pool),2);
        if isempty(sub_t)
            sub_t = fetchmysql(sprintf(...
            'select tradingdate from s22.index_com where tradingdate >= ''%s'' and ticker = ''%s''  order by tradingdate limit 1',...
            month_cut_date{i},symbol_pool),2);
        end
        sub_symbol_pool = fetchmysql(sprintf('select symbol from s22.index_com where tradingdate = ''%s'' and ticker = ''%s''',sub_t{1},symbol_pool),2);
        [~,ia] = intersect(x(:,1),sub_symbol_pool);
        x = x(ia,:);
    end
    %return 
    y = fetchmysql(sprintf(sql_str_fr,tn_1month_return,month_cut_date{i}),2);
    if neutralization_sel>0
        %industry
        sub_indus_code = yq_methods.get_industry_class(month_cut_date{i});
        %��ֵ
        sub_mv = yq_methods.get_market_value(month_cut_date{i});
        %����
        inds = suscc_intersect({x(:,1),y(:,1),sub_indus_code(:,1),sub_mv(:,1)});
        x = x(inds(:,1),:);
        y = y(inds(:,2),:);
        sub_indus_code = sub_indus_code(inds(:,3),:);
        sub_mv = sub_mv(inds(:,4),:);
        x_v = cell2mat(x(:,2));
        y_v = cell2mat(y(:,2));
        sub_mv = cell2mat(sub_mv(:,2));
        sub_indus_code = cell2mat(sub_indus_code(:,2));
        %���Ի�
        %�Ʊ�������
        u_sub_sub_x1 = unique(sub_indus_code);
        sub_sub_x1_yb = zeros(length(sub_indus_code),length(u_sub_sub_x1));
        for j = 1:length(u_sub_sub_x1)
            sub_sub_x1_yb(eq(sub_indus_code,u_sub_sub_x1(j)),j) = 1;
        end
        sub_sub_x_f = [ones(size(x_v)),sub_sub_x1_yb,sub_mv];
        [~,~,x_v] = regress(x_v,sub_sub_x_f);
    else
        [~,ia,ib] = intersect(x(:,1),y(:,1));
        x_v = cell2mat(x(ia,2));
        y_v = cell2mat(y(ib,2));
    end
    
    [ic(i),p(i)] = corr(x_v,y_v,'Type','Spearman');
    
    [~,ia] = sort(x_v);
    y_v = y_v(ia);
    ind_cut = floor(linspace(0,length(y_v),11));
    temp = zeros(10,1);
    for j = 1:length(ind_cut)-1
        temp(j) = mean(y_v((ind_cut(j)+1):ind_cut(j+1)));
    end
    Y{i} = temp;
        
    sprintf('%d-%d',i,T_month_cut)
    
end
Y=[Y{:}]';

N = 10;
y_curve = cumprod(1+Y);
nh_all = zeros(N,1);
for i = 1:N
    [~,~,sta_val] = curve_static_month(y_curve(:,i));
    nh_all(i) = sta_val.nh*100;
end
r_month = sum(Y(:,1)-Y(:,end)-fee*2,2)/2;
y_curve_end = cumprod(1+r_month);

leg_str = cell(N,1);
for i = 1:N
    leg_str{i} = sprintf('S%d',i);
end
figure
subplot(2,1,1)
bar(nh_all)
ylabel('�껯����%')
subplot(2,1,2)
plot(y_curve,'LineWidth',2)
legend(leg_str,'NumColumns',2)
ylabel('��������')

figure
subplot(2,1,1)
bar(r_month*100)
ylabel('�������¶�����%')
set(gca,'XTickLabelRotation',90);
set(gca,'XTick',floor(linspace(1,length(month_cut_date),30)),'xlim',[1,length(month_cut_date)]);
set(gca,'XTickLabel',month_cut_date(floor(linspace(1,length(month_cut_date),30))));
subplot(2,1,2)
bpcure_plot_updateV2(month_cut_date,y_curve_end)
[v,v_str,sta_val] = curve_static_month(y_curve_end,2);
v([1:5,12:13]) = v([1:5,12:13])*100;
ic = ic * 100;
re = [v_str',num2cell(v');{'ic',mean(ic)};{'ic_ir',mean(ic)/std(ic)};{'p_value',mean(p)}];