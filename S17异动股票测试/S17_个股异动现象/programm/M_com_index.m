%{
为了公平对比指数增强前后的效果，我们使用指数成分股、权重数据合成指数，并在合成指数的
基础上做增强，这样结果更公平
全部使用预测者分钟数据计算
%}


clear

index_sel = 2;
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
%合成
sql_strb = ['select symbol,open,close from ycz_min_history.`%s` where '...
    ' time(tradingdate)=''15:00:00'''];
% sql_stra = ['select symbol,open,close from ycz_min_history.`%s` where '...
%     ' time(tradingdate)=''09:31:00'''];
T = length(tref);
y = zeros(T,1);
parfor i = 2:T
    ind = find(tref_num_u<tref_num(i),1,'last');
    sub_code = w_index(eq(tref_num_index,tref_num_u(ind)),:);
    ind1 = cellfun(@(x) strcmp(x(1),'0'),sub_code(:,2));
    sub_code(ind1,2) = cellfun(@(x) ['sz',x],sub_code(ind1,2),'UniformOutput',false);
    sub_code(~ind1,2) = cellfun(@(x) ['sh',x],sub_code(~ind1,2),'UniformOutput',false);
    sub_x_t0 = fetchmysql(sprintf(sql_strb,datestr(tref_num(i-1),'yyyymmdd')),2);
    sub_x_t1 = fetchmysql(sprintf(sql_strb,datestr(tref_num(i),'yyyymmdd')),2);
    
    [inds,commValue] = suscc_intersect({sub_code(:,2),sub_x_t0(:,1),sub_x_t1(:,1)});
    
    sub_x = [sub_code(inds(:,1),3),sub_x_t0(inds(:,2),3),sub_x_t1(inds(:,3),3)];
    sub_x = cell2mat(sub_x);
    
    temp = (sub_x(:,3)./sub_x(:,2)-1).*sub_x(:,1)./100;
    %temp = (sub_x(:,3)./sub_x(:,2)-1).*sub_x(:,1)./sum(sub_x(:,1));
    y(i) = sum(temp);
    sprintf('%d-%d',i,T)
    
end

sql_str2 = ['select closeprice from futuredata.index_ycz_data where tradingdate>=''%s''',...
    ' and tradingdate<=''%s'' and symbol = ''sh%s'' order by tradingdate'];
x_ref = fetchmysql(sprintf(sql_str2,t1,t2,index_code{index_sel}));

figure;
plot(x_ref/x_ref(1));
hold on
plot(cumprod(1+y));
legend({'指数','合成指数'})
sub_y = y;
sub_x = tref;
sub_ind = floor(linspace(1,length(sub_y),20));
set(gca,'xtick',sub_ind);
set(gca,'xlim',[1,length(sub_y)]);
set(gca,'XTickLabel',sub_x(sub_ind));
set(gca,'XTickLabelRotation',45);







