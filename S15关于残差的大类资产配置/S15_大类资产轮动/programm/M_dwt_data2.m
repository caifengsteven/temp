%获取数据
clear
t1 = datenum(2015,6,30);
t2 = datenum(2016,8,31);

symbol_pool = {'IF','IC'};
x1 = get_yq_data(symbol_pool{1});
x2 = get_yq_data(symbol_pool{2});
%%交易日期，合约代码，开，手，结

[~,ia,ib] = intersect(x1(:,1),x2(:,1));
x1 = x1(ia,:);
x2 = x2(ib,:);

T = size(x1,1);
index_break = false(T,2);
for i = 2:T
    if ~strcmp(x1(i,2),x1(i-1,2))
        index_break(i,1) = true;
    end
    if ~strcmp(x2(i,2),x2(i-1,2))
        index_break(i,2) = true;
    end
end

x_o = [cell2mat(x1(:,4)),cell2mat(x2(:,4))];
yield_v = zeros(size(x_o));
yield_v(2:end,:) = x_o(2:end,:)./x_o(1:end-1,:)-1;
yield_v(index_break(:,1),1) = 0;
yield_v(index_break(:,2),2) = 0;
x = x_o(1,:).*cumprod(1+yield_v);
tref = x1(:,1);
tref_num = datenum(tref);

T0 = 7;
ind = find(tref_num>=t1&tref_num<=t2);
%ind = ind:(ind+1024-1);
tref = tref(ind);
%sub_y = yield_v(ind,:);
sub_y = log(x(ind,:));
%sub_y = x(ind,:);
[h,pValue,stat,cValue,reg1,reg2] = egcitest(sub_y);
sub_y1 = reg1.res;
[A_a,D_a] = wt_msr(sub_y1',T0,'db8',0);


for i = 1:T0
    subplot(T0+1,1,T0-i+1);plot(D_a{i});
    set(gca,'xlim',[0,length(sub_y1)+1]);
    ylabel(sprintf('db%d',i))
end
y =  movmax(abs(D_a{1}),[20,0]);
subplot(T0+1,1,T0+1)
bar(ind,y);

































function  [x2,multiplier_v] = get_yq_data(symbol)
obj1= ad_future_method();
sql_str1 = obj1.get_future_basic_info_yq(symbol);
sql_str2 = obj1.get_future_data_yq(symbol);
%参数 

%上市日期，保证金比例，合约乘数，最小变动单位，最后交易日
x1 = fetchmysql(sql_str1,2);
%asure_v = x1{end,2};
%asure_v = 20;%保证金比例
multiplier_v = x1{end,3};
%ini_cash = 10000000; %ini_cash
%use_ratio = 0.2;
%use_ratio = asure_v/100; %建仓资金比例
%fee = 3/10000; %手续费


%%交易日期，合约代码，开，手，结
x2 = fetchmysql(sql_str2,2);
%price_close = cell2mat(x2(:,4));
end