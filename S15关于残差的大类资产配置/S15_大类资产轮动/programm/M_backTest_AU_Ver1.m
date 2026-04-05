%{
技术报告中回测框架以下 必要 条件没有写：
1计算协整所用的局部窗口是多少
2协整后，计算信号的阈值是多少
3小波分析发现异常，背离协整后，停止多长时间？下个信号如何发出。

回测框架 ver1
au-oil回测框架

1判断是否协整，否，跳过
2是，谱分析确定窗口，计算阈值
3当前数据超过阈值后触发信号 开仓信号
4回复后 平仓
5小波分析发现异常后，平仓，返回1

breakdown ind =[2585;2708;2817;2899;2955;3329;3383;3425;2911;3488;3510;3563;3613;3632;3662;...
3682;3704;3748;3819;3908]
programm
M_dwt_data1.m
%}

clear

%载入数据
load dataset1.mat
breakdown_ind =[2585;2708;2817;2899;2955;3329;3383;3425;2911;3488;3510;3563;3613;3632;3662;...
3682;3704;3748;3819;3908];
N_pool = (1:12)*20;
%参数
%协整窗口2month
window_CG = 40;
%谱分析窗口2年
window_fre = 20*24;
%信号/std 比率
signal_cri = 3;

t1 = datenum(2009,12,31);
t2 = datenum(2016,6,30);
%日收益率
yield_v = zeros(size(x));
yield_v(2:end,:) = x(2:end,:)./x(1:end-1,:)-1;

ind = find(tref_num>=t1,1);
T = find(tref_num>=t2,1)-1;
%结果初始化
signal_v = zeros(T,1); %信号
state_v = zeros(T,1);  %状态
process_para = zeros(T,2);%窗口，std

%执行策略
for i = ind:T-1
    if eq(state_v(i),0)
        %判断协整
        sub_x_cg = x(i-window_CG+1:i,:);
        [~,pValue] = egcitest(sub_x_cg);
        if pValue<0.05
            %复合协整
            state_v(i+1) =1;
            %谱分析
            sub_x_fre = x(i-window_fre+1:i,:);
            [~,sub_H] = cal_freq(sub_x_fre,N_pool);
            
        else
            %不符合协整
            state_v(i+1) =0;
        end
    else        
        if any(eq(breakdown_ind,i))
            %dwt识别协整中断
        	state_v(i+1) = 0;
        else
            %继续协整
            state_v(i+1) = state_v(i);
        end
    end
    %计算信号
    if eq(state_v(i),1)
        [~,~,~,~,reg1]  = egcitest(x(i-sub_H+1:i,:));
        sub_std = std(reg1.res);
        if reg1.res(end)/sub_std>=signal_cri
            signal_v(i+1) = -1;
        elseif reg1.res(end)/sub_std<=-signal_cri
            signal_v(i+1) = 1;
        else
            signal_v(i+1) = signal_v(i);
        end
    else
        signal_v(i+1) = 0;
    end
    sprintf('%d-%d',i,T)
end
y1 = yield_v(1:T,1).*signal_v;
y2 = yield_v(1:T,2).*-signal_v;

plot(0.5*cumprod(1+y1(ind:end))+0.5*cumprod(1+y2(ind:end)))
hold on
plot([0.5*cumprod(1+y1(ind:end)),0.5*cumprod(1+y2(ind:end))])


