%M_com_Hindenburg
%整合中证2011年9月30日前数据再次计算
%版本2
%时间范围扩大
%可以断点计算 后期再加
%如果用于实盘，1需要更新数据，2需要和数据库配合 工作量比较大
%由于中证全指数据缺少2011年9月30日以前指数数据，需要手动，无法使用本程序。
%

clear

%参数
window_cal = 30;
window_week = 5;
%数据库
gta_astock_db = 'futuredata.STK_MKT_BWARDQUOTATION';
index_com_symbol_db = 'futuredata.a_index_composition_data';

%指数数据
[index_data,index_code] = get_index_data('上证综指');
index_data(:,1) = cellfun(@(x) x(1:10),index_data(:,1),'UniformOutput',false);
% [~,~,sub_index_data] = xlsread('000985_index_before.csv');
% index_data = [index_data(:,[1,end]);sub_index_data(2:end,:)];
% index_data(:,1) = cellstr(datestr(index_data(:,1),'yyyy-mm-dd'));

index_code =['SHSE.',index_code];

%时间整理
tref_str1 = index_data(:,1);
tref_str2 = fetchmysql(sprintf(['select distinct tradingdate from futuredata.a_index_composition_data ',10,...
    'where index_code = ''%s'''],index_code),2);
[tref_str,ia,ib] = intersect(tref_str1,tref_str2);

index_data = index_data(ia,:);
close_price = cell2mat(index_data(:,end));
%y0 = cell2mat(index_data(:,end));
y0 = [0;close_price(2:end)./close_price(1:end-1)-1];
tref = datenum(index_data(:,1));


t0 = tref_str{1};
tt = tref_str{end};
symbol_all = fetchmysql(sprintf(['select distinct symbol from futuredata.STK_MKT_BWARDQUOTATION ',10,...
    'where tradingdate >=''%s'' and tradingdate<=''%s'' and symbol<900000'],t0,tt),2);

T = length(tref_str);
m = length(symbol_all);


sql_str1 = 'select symbol,closeprice/precloseprice-1 from %s where tradingdate= ''%s'' and filling = 0';
sql_str2 = 'select symbol from %s where index_code = ''%s'' and tradingdate = ''%s'' ';
fn=  sprintf('result%s.mat',index_code(6:end));
if exist(fn,'file')
   load(fn)
   X = re.X;
   Y_pre = re.Y_pre;
   factor_v = re.factor_v;
   %re.tref = tref;
   %re.symbol_all = symbol_all;
   %re.close_price = close_price;
else
    %load re000985_X.mat X
    %%{
    X = nan(m,T);
    Y = cell(T,1);
    %计算X
    %load Y0
    %必须并行
    parfor i = 1:T
        sub_sql = sprintf(sql_str1,gta_astock_db,tref_str{i});
        x = fetchmysql(sub_sql,2);
        sub_sql = sprintf(sql_str2,index_com_symbol_db,index_code,tref_str{i});
        y = fetchmysql(sub_sql,2);
        if strcmp(y{1}(end-4),'.')
            y = cellfun(@(x) x(1:6),y,'UniformOutput',false);
        else
            y = cellfun(@(x) x(6:end),y,'UniformOutput',false);
        end

        [~,ia] = intersect(x(:,1),y);
        x = x(ia,:);
        [~,ia,ib] = intersect(symbol_all,x(:,1),'stable');
        %X(ia,i) = cell2mat(x(ib,2));
        Y{i} = [ia,cell2mat(x(ib,2))];
        sprintf('%s-%d-%d',tref_str{i},i,T)
    end

    for i = 1:T
        temp  = Y{i};
        X(temp(:,1),i) = temp(:,2);
    end
    %}
    %load re000985_Ypre Y_pre
    %%{
    %合成趋同度因子
    Y_pre = nan(size(X));
    for i = 1:m
        sub_x = X(i,:);
        parfor j = window_cal:T
            sub_sub_x = sub_x(j-window_cal+1:j);
            sub_sub_y = y0(j-window_cal+1:j);
            window_ind_sub = ~isnan(sub_sub_x);
            sub_sub_x = sub_sub_x(window_ind_sub);
            sub_sub_y = sub_sub_y(window_ind_sub);  
            if length(sub_sub_x)>5
                Y_pre(i,j) = get_rsqure(sub_sub_x',sub_sub_y);
            end
        end
        sprintf('%d-%d',i,m)
    end
    %}

    %cal indicator
    factor_v = nan(T,1);
    for i = 1:T
        sub_y = Y_pre(:,i);
        sub_y(isnan(sub_y)) = [];
        if ~isempty(sub_y)
            factor_v(i) = mean(sub_y);
        end
    end

    re = [];
    re.X = X;
    re.Y_pre = Y_pre;
    re.tref = tref;
    re.symbol_all = symbol_all;
    re.close_price = close_price;
    re.factor_v = factor_v;
   
    if ~exist(fn,'file')
        save(fn,'re');
    end
end

% %cal indicator
% factor_v = nan(T,1);
% for i = 1:T
%     sub_y = Y_pre(:,i);
%     sub_y(isnan(sub_y)) = [];
%     if ~isempty(sub_y)
%         factor_v(i) = mean(sub_y);
%     end
% end


t0= datenum(2010,1,4);
tt = datenum(2016,12,12);
t_ind = find(tref>=t0&tref<=tt);

figure;

y_lim = nan(size(y0));
x_lim = y_lim;
y_lim(window_week+1:end) = close_price(window_week+1:end)./close_price(1:end-window_week)-1;
x_lim(window_week+1:end) = factor_v(window_week+1:end)./factor_v(1:end-window_week)-1;
%计算坐标轴象限位置
v_location = zeros(size(y_lim));
v_location(x_lim>0&y_lim>0) = 1;
v_location(x_lim<0&y_lim>0) = 2;
v_location(x_lim<0&y_lim<0) = 3;
v_location(x_lim>0&y_lim<0) = 4;
%计算转移矩阵
v_location = v_location(t_ind);
v_p = zeros(4,4);
for i = 1:4
    sub_ind = find(eq(v_location,i));
    if eq(sub_ind(end),length(v_location))
        sub_ind(end) = [];
    end
    for j = 1:4
        v_p(i,j) = sum(eq(v_location(sub_ind+1),j))/length(sub_ind)*100;        
    end
end
%相邻干扰，和文献结果不同，对回测无影响，有时间可以找文献做下

plot(x_lim(t_ind)*100,y_lim(t_ind)*100,'r.')
hold on
lims = axis(gca);
lims1 = [min(lims),max(lims)];
plot(lims1,[0,0],'-','LineWidth',2,'color',[0.47,0.67,0.19])
plot([0,0],lims1,'-','LineWidth',2,'color',[0.47,0.67,0.19]);
axis(lims);
xlabel('趋同度涨跌幅')
ylabel('指数涨跌幅')

figure
t_ind1 = find(tref>=datenum(2015,1,1));
yyaxis left
plot(tref(t_ind1),close_price(t_ind1),'-','linewidth',2,'color',[0.64 0.08 0.18]);
yyaxis right
plot(tref(t_ind1),factor_v(t_ind1),'-','LineWidth',2,'color',[0,0.451,0.7412]);

set(gca,'XTickLabelRotation',90);
set(gca,'XTick',tref(t_ind1(1:20:end)),'xlim',tref(t_ind1([1,end])));
datetick('x','yyyymmdd','keepticks');
set(gca,'fontsize',12);
box off
set(gca,'linewidth',1.5);

legend({'指数','趋同度指数'})



