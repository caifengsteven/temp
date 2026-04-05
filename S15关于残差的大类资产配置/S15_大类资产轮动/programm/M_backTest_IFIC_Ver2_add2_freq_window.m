%IF IC回测主程序
clear
symbol_pool = {'IF','IC'};
[x1,multiplier_v1] = get_yq_data(symbol_pool{1});
[x2,multiplier_v2] = get_yq_data(symbol_pool{2});
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

%breakdown_ind = [-1,0];
breakdown_ind =[76;115;157;189;216;236];
N_pool = (0:8)*5+20;
%参数
%协整窗口2month
window_CG = 40;
%谱分析窗口
window_fre_pool = (1:12)*20;
test_re = [];
for window_sel = 1:length(window_fre_pool)
    window_fre = window_fre_pool(window_sel);
%window_fre = 40;
%协整 信号/std 比率
signal_cri =2;
%非协整评价值计算窗口
window_UN = 40;
%非协整 信号/std 比率
signal_UN_cri =2;
t1 = datenum(2015,6,30);
t2 = datenum(2016,8,31);

ini_ind = find(tref_num>=t1,1);
T = find(tref_num>=t2,1)-1;
%结果初始化
signal_v = zeros(T,1); %信号 协整
signal_UN = zeros(T,1);%信号 非协整
state_v = zeros(T,1);  %状态
process_para = zeros(T,2);%窗口，std
indicator_UN = get_UNCN_value(yield_v,window_UN);
sub_H_un = window_UN; %初始化
%执行策略
for i = ini_ind:T-1
    if eq(state_v(i),0)
        %判断协整
        sub_x_cg = x(max(i-window_CG+1,1):i,:);
        [~,pValue] = egcitest(sub_x_cg);
        if pValue<0.05
            %复合协整
            state_v(i+1) =1;
            %谱分析
            sub_x_fre = x(max(i-window_fre+1,1):i,:);
            [~,sub_H] = cal_freq(sub_x_fre,N_pool);
            
        else
            %不符合协整
            state_v(i+1) =0;
        end
    else        
        if any(eq(breakdown_ind,i))
            %dwt识别协整中断
        	state_v(i+1) = 0;
            %谱分析计算周期
            sub_x_fre_un = x(max(i-window_fre+1,1):i,:);
            [~,sub_H_un] = cal_freq(sub_x_fre_un,N_pool);            
        else
            %继续协整
            state_v(i+1) = state_v(i);
        end
    end
    %计算信号
    if eq(state_v(i),1)
        %非协整信号改变
        signal_UN(i+1) = 0;
        %计算协整信号
        [~,~,~,~,reg1]  = egcitest(x(max(i-sub_H+1,1):i,:));
        sub_std = std(reg1.res);
        if reg1.res(end)/sub_std>=signal_cri
            signal_v(i+1) = -1;
        elseif reg1.res(end)/sub_std<=-signal_cri
            signal_v(i+1) = 1;
        else
            signal_v(i+1) = signal_v(i);
        end
    else
        %协整信号改变
        signal_v(i+1) = 0;
        %计算非协整信号
        sub_std = std(indicator_UN(max(i-sub_H_un+1,1):i));
        if indicator_UN(end)/sub_std>=signal_UN_cri
            signal_UN(i+1) = 1;
        elseif indicator_UN(end)/sub_std<=-signal_UN_cri
            signal_UN(i+1)=-1;
        else
            signal_UN(i+1) = signal_UN(i);
        end
                
    end
    sprintf('%d-%d',i,T)
end



signal_com = signal_v+signal_UN;
signal_v = signal_com;
% sub_y = ybac2;
% sub_x = tref(ini_ind:T);
% sub_ind = floor(linspace(1,length(sub_y),20));
% set(gca,'xtick',sub_ind);
% set(gca,'xlim',[1,length(sub_y)]);
% set(gca,'XTickLabel',sub_x(sub_ind));
% set(gca,'XTickLabelRotation',45);
% legend({'M1','M2'},'location','northwest')
ini_cash = 1000*10000/2;
asure_v = 45;
use_ratio = 0.6;
fee = 2/10000;

signal_v1 = signal_v;
ind_c1 = find(abs(diff(signal_v1))>1);
signal_v1(ind_c1+1) = 0;

signal_v2 = signal_v;
ind_c2 = find(abs(diff(signal_v2))>1);
signal_v2(ind_c2+1) = 0;
signal_v1(index_break(1:T,1)) = 0;
signal_v2(index_break(1:T,2)) = 0;

[y_bac1,re1]=future_bac_method(ini_cash,asure_v,multiplier_v1,...
    use_ratio,fee,x_o(1:T,1),[0;signal_v1]);

[y_bac2,re2]=future_bac_method(ini_cash,asure_v,multiplier_v1,...
    use_ratio,fee,x_o(1:T,2),[0;-signal_v2]);
y_bac3 = y_bac1(ini_ind:T)+y_bac2(ini_ind:T);
test_re = cat(2,test_re,y_bac3/y_bac3(1));
end




plot(test_re);
sub_y = y_bac3;
sub_x = tref(ini_ind:T);
sub_ind = floor(linspace(1,length(sub_y),20));
set(gca,'xtick',sub_ind);
set(gca,'xlim',[1,length(sub_y)]);
set(gca,'XTickLabel',sub_x(sub_ind));
set(gca,'XTickLabelRotation',45);
legend(cellstr(num2str(window_fre_pool')),'location','northwest')

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