%{
为了公平对比指数增强前后的效果，我们使用指数成分股、权重数据合成指数，并在合成指数的
基础上做增强，这样结果更公平
全部使用预测者分钟数据计算
指数增强
下跌异动信号

昨日触发的  昨天信号发出收盘买入相同仓位，收盘时卖出原来相同仓位 今天收盘/昨日买入价格-1 %
今日触发的  今天信号发出收盘卖出，今天收/昨收盘-1 不变
两日都未倍触发的 今收盘/昨天收盘-1

后复权升级
%}

clear

index_sel = 2;
fee = 0;
%交易日历获取
t1 = '2013-01-01';
t2 = '2017-01-01';

sql_str = ['select distinct tradingdate from futuredata.STK_MKT_QUOTATION ',...
    'where tradingdate >=''%s'' and tradingdate<=''%s'' and filling =0 order by tradingdate'];
tref = fetchmysql(sprintf(sql_str,t1,t2),2);
tref_num = datenum(tref);
%读取指数成分股及权重数据
index_code = {'000905','000300','000016'};
[~,~,w_index] = xlsread(sprintf('index_%s.xlsx',index_code{index_sel}));
w_index = w_index(2:end,:);
w_index(:,2) = cellfun(@(x) sprintf('%0.6d',x),w_index(:,2),'UniformOutput',false);
tref_num_index = datenum(w_index(:,1));
tref_num_u = unique(tref_num_index);
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
    'where date(tradingdate)=''%s'' and d <=-3 '];

%创建输出文件
%关闭未关闭的fid
fclose all;
fn_out=  sprintf('logre_Minus%s.log',index_code{index_sel});
fid = fopen(fn_out,'w+'); %会清除文件原来内容

T = length(tref);
y = zeros(T,1);
y1 = y;
y2 = y;

for i = 2:T
    ind = find(tref_num_u<tref_num(i),1,'last');
    sub_code = w_index(eq(tref_num_index,tref_num_u(ind)),:);
    ind1 = cellfun(@(x) strcmp(x(1),'0'),sub_code(:,2));
    sub_code(ind1,2) = cellfun(@(x) ['sz',x],sub_code(ind1,2),'UniformOutput',false);
    sub_code(~ind1,2) = cellfun(@(x) ['sh',x],sub_code(~ind1,2),'UniformOutput',false);
    sub_x_t0 = fetchmysql(sprintf(sql_strb,datestr(tref_num(i-1),'yyyymmdd')),2);
    sub_x_t0 = arange_yczmin_data(sub_x_t0);%获取开盘、收盘价
    
    sub_x_t1 = fetchmysql(sprintf(sql_strb,datestr(tref_num(i),'yyyymmdd')),2);
    sub_x_t1 = arange_yczmin_data(sub_x_t1);
    
    [inds,commValue] = suscc_intersect({sub_code(:,2),sub_x_t0(:,1),sub_x_t1(:,1)});
    
    sub_x = [sub_code(inds(:,1),3),sub_x_t0(inds(:,2),2:3),sub_x_t1(inds(:,3),2:3)];
    sub_x = cell2mat(sub_x);
        
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
%     %T+1限制，今日的信号和昨日相同，无法触发（早晨刚买入）
%     if  ~isempty(sub_signal_today)&&~isempty(sub_signal_yestoday)
%         [~,ia] = setdiff(sub_signal_today(:,1),sub_signal_yestoday(:,1));
%         sub_signal_today = sub_signal_today(ia,:);
%     end
    
    temp = (sub_x(:,5)./sub_x(:,3)-1).*sub_x(:,1)./100;
    temp1 = temp;
    
%     if ~isempty(sub_signal_yestoday)
%         [~,ia0,ib0] = intersect(sub_signal_yestoday(:,1),commValue);%昨日触发的   
%         temp1(ib0) = (sub_x(ib0,5)./(cell2mat(sub_signal_yestoday(ia0,2)).*sub_coeff0_c(ib0).*(1+fee))-1).*sub_x(ib0,1)./100;
%         %temp1(ib0) = (sub_x(ib0,5)./(sub_x(ib0,4).*(1+fee))-1).*sub_x(ib0,1)./100;%昨日触发的
%     end
    if ~isempty(sub_signal_today)
        %复权        
        [~,ia1,ib1] = intersect(sub_signal_today(:,1),commValue);%今日触发的
        temp1(ib1) = temp1(ib1)+(sub_x(ib1,5)./(cell2mat(sub_signal_today(ia1,2)).*(1+fee).*sub_coeff1_c(ib1))-1).*sub_x(ib1,1)./100;
        if ~isempty(ia1)
            y2(i) = mean(sub_x(ib1,5)./(cell2mat(sub_signal_today(ia1,2)).*sub_coeff1_c(ib1))-1);
        end
        
        for j = 1:length(ib1)
            fprintf(fid,'%s: %s %0.2f %0.2f\n',tref{i}(1:10),commValue{ib1(j)},cell2mat(sub_signal_today(ia1(j),2)).*sub_coeff1_c(ib1(j)),sub_x(ib1(j),5));
        end
        
    end
    %temp = (sub_x(:,3)./sub_x(:,2)-1).*sub_x(:,1)./sum(sub_x(:,1));
    y(i) = sum(temp);
    y1(i) = sum(temp1);
    sprintf('%d-%d',i,T)
    
end
fclose(fid);

sql_str2 = ['select closeprice from futuredata.index_ycz_data where tradingdate>=''%s''',...
    ' and tradingdate<=''%s'' and symbol = ''sh%s'' order by tradingdate'];
x_ref = fetchmysql(sprintf(sql_str2,t1,t2,index_code{index_sel}));

figure;
plot(cumprod(1+[y,y1]),'LineWidth',2);
hold on
plot(cumprod(1+y1)-cumprod(1+y)+1,'LineWidth',2)
plot(cumprod(1+y2),'LineWidth',2);
legend({'合成指数','合成指数增强','增强超额收益','异动信号贡献'})
sub_y = y;
sub_x = cellstr(datestr(tref_num,'yyyymmdd'));
sub_ind = floor(linspace(1,length(sub_y),20));
set(gca,'xtick',sub_ind);
set(gca,'xlim',[1,length(sub_y)]);
set(gca,'XTickLabel',sub_x(sub_ind));
set(gca,'XTickLabelRotation',45);







