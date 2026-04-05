%{
FICC系列研究之二方法
横截面动量策略

剔除上市日期小于半年365/2
剔除前20个交易日成交量存在小于1万手的
强制设定
保证金设定为20%
仓位设定为50%

%}

%所有成分的上市日期
%主力合约名称
%所有成分的收盘数据
%所有成分的成交量数据
%所有成分的合约乘数

clear

%获取数据
%上市时间
obj = ad_future_method();
[symbols1,list_date]=obj.get_future_listdate_yq();
list_date_num = datenum(list_date);

sql_str = ['select ticker,contractObject,tradedate,closeprice,turnoverVol ',...
    'from futuredata.yuqer_fusharedata where maincon=1'];
x = fetchmysql(sql_str,2);
%symbols1 = unique(x(:,2));
tref = unique(x(:,3));
tref_num = datenum(tref);
T1 = length(symbols1);
T2 = length(tref);

sql_str = 'select symbol,tradingdate,close_price  from futuredata.yq_future_rehabilitation_data';
x_add_reprice = fetchmysql(sql_str,2);
temp = cellfun(@(x) strsplit(x,'.'),x_add_reprice(:,1),'UniformOutput',false);
x_add_reprice_symbol = cellfun(@(x) x{2},temp,'UniformOutput',false);


a_main_code_name = cell(T2,T1);%主力合约代码名称
a_close_price = zeros(T2,T1);%收盘价格
a_volume = a_close_price;%成交量
a_close_price_recal = a_close_price;

for i = 1:T1
    sub_code = symbols1(i);

    sub_x = x(strcmp(x(:,2),sub_code),:);        
    [~,ia,ib] = intersect(tref,sub_x(:,3));
    a_main_code_name(ia,i) = sub_x(ib,1);
    a_close_price(ia,i) = cell2mat(sub_x(ib,4));    
    a_volume(ia,i) = cell2mat(sub_x(ib,5));    
    
    sub_add_data = x_add_reprice(strcmp(x_add_reprice_symbol,sub_code),:);
    [~,ia,ib] = intersect(tref,sub_add_data(:,2));
    a_close_price_recal(ia,i) = cell2mat(sub_add_data(ib,3));
    
    
    
end
a_close_price(isnan(a_close_price)) = 0;

paras_V = cell(T1,5);
for i = 1:T1
    sql_str1 = obj.get_future_basic_info_yq(symbols1{i});
    temp = fetchmysql(sql_str1,2);
    paras_V(i,:) = temp(end,:);
end
multipler_V = cell2mat(paras_V(:,3));

%准备开始回测
t0 = datenum(2005,01,04);
asure_v = 20;%保证金比例
ini_cash = 10000*T1*100; %ini_cash
%use_ratio = 0.2;
use_ratio = 0.5; %建仓资金比例
fee = 3/10000; %手续费

H = 10;%调仓频率
R = 15;%计算收益窗口

T = length(tref);
ini_ind0 = find(tref_num>=t0,1);
%消除起点影响
%ini_ind_all = ini_ind0:(ini_ind0+H-1);
ini_ind_all = ini_ind0;

Y = zeros(T,length(ini_ind_all));
for ii = 1:length(ini_ind_all)
    Y_bac = zeros(T,T1);    
    ini_ind = ini_ind_all(ii);
    Y_bac(ini_ind-1,:) = ini_cash/T1/length(ini_ind_all);
    for i = ini_ind:H:T
        sub_wid_volume_test = i-20:i;
        sub_wid = i:(i+H-1); %窗口
        sub_wid(sub_wid>T) = [];
        %准备数据和信号
        ind_test = false(T1,1);
        return_val = zeros(T1,1);
        for j = 1:T1
            sub_volume = a_volume(sub_wid,j);
            if all(sub_volume>10000)&&tref_num(i)-list_date_num(j)>365/2            
               ind_test(j) = true; 
               return_val(j) = a_close_price_recal(i,j)/a_close_price_recal(i-R,j)-1;
               if isnan(return_val(j)) || isinf(return_val(j))
                   return_val(j) = 0;
               end
            end
        end
        %排序并设定多空
        if sum(ind_test)<5
            ind_direction = ones(size(ind_test));
        else
            ind_direction = zeros(size(ind_test));
            temp_ind = find(ind_test);
            [~,ia] = sort(return_val(temp_ind));
            
            num1 = floor(length(ia)*0.2);
            ind_direction(temp_ind(ia(1:num1))) = -1;
            ind_direction(temp_ind(ia(end-num1+1:end))) = 1;
            ind_test(eq(ind_direction,0)) = false;
            
        end
        sub_ini=sum(Y_bac(i-1,:))/sum(ind_test);
        for j = 1:T1
            if ind_test(j)         
                sub_price_close = a_close_price(sub_wid,j);
                %signal_direction = a_close_price(i,j)>a_close_price(i-R,j);
                sub_main_code = a_main_code_name(sub_wid,j);
                sub_price = a_close_price(sub_wid,j);
                sub_ind_sel = ~eq(sub_price,0);
                sub_signal = ones(size(sub_price));
                if eq(ind_direction(j),-1)
                    sub_signal = -sub_signal;
                end
                for k = 2:length(sub_signal)
                    if ~strcmp(sub_main_code(k),sub_main_code(k-1))
                        sub_signal(k) = 0;
                    end
                end
                sub_signal(end) = 0;
                sub_signal=[0;sub_signal(sub_ind_sel)];
                sub_y_bac = future_bac_method(sub_ini,asure_v,multipler_V(j),...
                    use_ratio,fee,sub_price(sub_ind_sel),sub_signal);
                Y_bac(sub_wid(sub_ind_sel),j) = sub_y_bac(2:end);
            end
        end
        sprintf('%d-%d-%d',i,T,ii)
    end
    Y(:,ii) = sum(Y_bac,2);
end
figure;
Y2 = sum(Y(ini_ind0:end,:),2);
Y2(1:H) = ini_cash;
Y2 = Y2./Y2(1);
sub_t = tref_num(ini_ind:end);
bpcure_plot_updateV2(sub_t,Y2);





