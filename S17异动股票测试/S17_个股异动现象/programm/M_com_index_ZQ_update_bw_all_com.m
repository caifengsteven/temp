%{
为了公平对比指数增强前后的效果，我们使用指数成分股、权重数据合成指数，并在合成指数的
基础上做增强，这样结果更公平
全部使用预测者分钟数据计算
指数增强
使用权市场数据构建指数，权重平均


昨日触发的  昨天信号发出收盘卖出，早盘买入， 当日收益 收盘/开盘-1
今日触发的  今天信号发出收盘卖出，信号收盘/昨收盘-1
两日都未倍触发的 今收盘/昨天收盘-1

后复权升级
%}

clear

fee1 = 2/10000;
fee2 = 11/10000;
index_sel = 1;
%交易日历获取
t1 = '2013-01-01';
t2 = '2017-01-01';

sql_str = ['select distinct tradingdate from futuredata.STK_MKT_QUOTATION ',...
    'where tradingdate >=''%s'' and tradingdate<=''%s'' and filling =0 order by tradingdate'];
tref = fetchmysql(sprintf(sql_str,t1,t2),2);
tref_num = datenum(tref);
% %读取指数成分股及权重数据
% index_code = {'000905','000300','000016'};
% [~,~,w_index] = xlsread(sprintf('index_%s.xlsx',index_code{index_sel}));
% w_index = w_index(2:end,:);
% w_index(:,2) = cellfun(@(x) sprintf('%0.6d',x),w_index(:,2),'UniformOutput',false);


% tref_num_index = datenum(w_index(:,1));
% tref_num_u = unique(tref_num_index);
%读取后复权系数
sql_str_bw = 'select symbol,tradingdate,CumulateBwardFactor1 from gtadata.STK_MKT_RepriceFactor order by tradingdate desc';
coef_v = fetchmysql(sql_str_bw,2);
tref_coeff_num = datenum(coef_v(:,2));
ind1 = cellfun(@(x) strcmp(x(1),'0'),coef_v(:,1));
coef_v(ind1,1) = cellfun(@(x) ['sz',x],coef_v(ind1,1),'UniformOutput',false);
coef_v(~ind1,1) = cellfun(@(x) ['sh',x],coef_v(~ind1,1),'UniformOutput',false);

%合成
sql_strb = ['select symbol,open,close from ycz_min_history.`%s` where ',...
    ' time(tradingdate)=''15:00:00'' or time(tradingdate)=''09:31:00'' ',...
    ' order by tradingdate'];
% sql_stra = ['select symbol,open,close from ycz_min_history.`%s` where '...
%     ' time(tradingdate)=''09:31:00'''];

sql_str_signal = ['select symbol,closeprice from ycz_result.sta_re20190702_last_30min ',...
    'where date(tradingdate)=''%s'' and d >=1 '];
sql_str_signal_2 = ['select symbol,closeprice from ycz_result.sta_re20190702_last_30min ',...
    'where date(tradingdate)=''%s'' and d <=-3 '];

T = length(tref);
y = zeros(T,1);
y1 = y;
parfor i = 2:T
%     ind = find(tref_num_u<tref_num(i),1,'last');
%     sub_code = w_index(eq(tref_num_index,tref_num_u(ind)),:);
%     ind1 = cellfun(@(x) strcmp(x(1),'0'),sub_code(:,2));
%     sub_code(ind1,2) = cellfun(@(x) ['sz',x],sub_code(ind1,2),'UniformOutput',false);
%     sub_code(~ind1,2) = cellfun(@(x) ['sh',x],sub_code(~ind1,2),'UniformOutput',false);
    sub_x_t0 = fetchmysql(sprintf(sql_strb,datestr(tref_num(i-1),'yyyymmdd')),2);
    sub_x_t0 = arange_yczmin_data(sub_x_t0);%获取开盘、收盘价
    sub_code = sub_x_t0(:,[1,1,2]);
    sub_code(:,end) = num2cell(ones(size(sub_code(:,1)))./size(sub_code,1));
    sub_x_t1 = fetchmysql(sprintf(sql_strb,datestr(tref_num(i),'yyyymmdd')),2);
    sub_x_t1 = arange_yczmin_data(sub_x_t1);
    
    [inds,commValue] = suscc_intersect({sub_code(:,2),sub_x_t0(:,1),sub_x_t1(:,1)});
    
    sub_x = [sub_code(inds(:,1),3),sub_x_t0(inds(:,2),2:3),sub_x_t1(inds(:,3),2:3)];
    sub_x = cell2mat(sub_x);
    sub_w_sum = sum(sub_x(:,1));
    sub_x(:,1) = sub_x(:,1)./sub_w_sum;
    
    sub_coef0 = coef_v(tref_coeff_num<=tref_num(i-1),:);
    sub_coef1 = coef_v(tref_coeff_num<=tref_num(i),:);
    
    %复权
    [~,ia,ib] = intersect(commValue,sub_coef0(:,1),'stable');
    sub_coeff0_c = ones(size(commValue));
    sub_coeff0_c(ia) = cell2mat(sub_coef0(ib,3));
    sub_x(:,2:3) = sub_x(:,2:3).*repmat(sub_coeff0_c,1,2);
    
    [~,ia,ib] = intersect(commValue,sub_coef1(:,1),'stable');
    sub_coeff1_c = ones(size(commValue));
    sub_coeff1_c(ia) = cell2mat(sub_coef1(ib,3));
    sub_x(:,4:5) = sub_x(:,4:5).*repmat(sub_coeff1_c,1,2);
    
    sub_signal_yestoday = fetchmysql(sprintf(sql_str_signal,datestr(tref_num(i-1),'yyyy-mm-dd')),2);
    sub_signal_today = fetchmysql(sprintf(sql_str_signal,datestr(tref_num(i),'yyyy-mm-dd')),2);
    %T+1限制，今日的信号和昨日相同，无法触发（早晨刚买入）
    if  ~isempty(sub_signal_today)&&~isempty(sub_signal_yestoday)
        [~,ia] = setdiff(sub_signal_today(:,1),sub_signal_yestoday(:,1));
        sub_signal_today = sub_signal_today(ia,:);
    end
    
    temp = (sub_x(:,5)./sub_x(:,3)-1).*sub_x(:,1);
    temp1 = temp;
    
    if ~isempty(sub_signal_yestoday)
        [~,ia0,ib0] = intersect(sub_signal_yestoday(:,1),commValue);%昨日触发的        
        temp1(ib0) = (sub_x(ib0,5)./(sub_x(ib0,4).*(1+fee1))-1).*sub_x(ib0,1);%昨日触发的
    end
    if ~isempty(sub_signal_today)
        %复权        
        [~,ia1,ib1] = intersect(sub_signal_today(:,1),commValue);%今日触发的
        temp1(ib1) = ((cell2mat(sub_signal_today(ia1,2)).*(1-fee2)).*sub_coeff1_c(ib1)./(sub_x(ib1,3))-1).*sub_x(ib1,1);
    end
    %
    sub_signal_today_2 = fetchmysql(sprintf(sql_str_signal_2,datestr(tref_num(i),'yyyy-mm-dd')),2);
    if ~isempty(sub_signal_today_2)
        %复权        
        [~,ia1,ib1] = intersect(sub_signal_today_2(:,1),commValue);%今日触发的
        temp1(ib1) = temp1(ib1)+(sub_x(ib1,5)*(1-fee2)./(cell2mat(sub_signal_today_2(ia1,2)).*(1+fee1).*sub_coeff1_c(ib1))-1).*sub_x(ib1,1);
    end
    
    %temp = (sub_x(:,3)./sub_x(:,2)-1).*sub_x(:,1)./sum(sub_x(:,1));
    y(i) = sum(temp);
    y1(i) = sum(temp1);
    sprintf('%d-%d',i,T)
    
end


figure;
plot(cumprod(1+[y,y1]),'LineWidth',2);
hold on
plot(cumprod(1+y1)-cumprod(1+y)+1,'LineWidth',2)
legend({'合成指数','合成指数增强','增强超额收益'})
sub_y = y;
sub_x = cellstr(datestr(tref_num,'yyyymmdd'));
sub_ind = floor(linspace(1,length(sub_y),20));
set(gca,'xtick',sub_ind);
set(gca,'xlim',[1,length(sub_y)]);
set(gca,'XTickLabel',sub_x(sub_ind));
set(gca,'XTickLabelRotation',45);
figure;
y_add = cumprod(1+y1)-cumprod(1+y)+1;
bpcure_plot_updateV2(tref_num,y_add)
[v,v_str,sta_val] = curve_static0(y_add)




