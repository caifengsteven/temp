%{
技术报告中回测框架以下 必要 条件没有写：
1计算协整所用的局部窗口是多少
2协整后，计算信号的阈值是多少
3小波分析发现异常，背离协整后，停止多长时间？下个信号如何发出。

回测框架 ver2
au-oil回测框架

1判断是否协整，否，
根据非协整方法计算信号
2是，谱分析确定窗口，计算阈值
3当前数据超过阈值后触发信号 开仓信号
4回复后 平仓
5小波分析发现异常后，平仓，返回1


breakdown ind =[2585;2708;2817;2899;2955;3329;3383;3425;2911;3488;3510;3563;3613;3632;3662;...
3682;3704;3748;3819;3908]
programm
M_dwt_data1.m

测试谱分析窗口大小对结果的影响

%}

clear

%载入数据
load dataset1.mat
breakdown_ind =[2585;2708;2817;2899;2955;3329;3383;3425;2911;3488;3510;3563;3613;3632;3662;...
3682;3704;3748;3819;3908];
N_pool = (1:12)*20;
%参数
%协整窗口2month
window_CG = 20;
%谱分析窗口2年
window_fre_pool = (2:0.5:5)*240;
window_fre_re = [];
for window_UN_sel = 1:length(window_fre_pool)
window_fre = window_fre_pool(window_UN_sel);
%window_fre = 20*24;
%协整 信号/std 比率
signal_cri = 2;
%非协整评价值计算窗口
window_UN = 66;
%window_UN = window_UN_pool(window_UN_sel);
%非协整 信号/std 比率
signal_UN_cri = 2;
t1 = datenum(2009,12,31);
t2 = datenum(2016,6,30);
%日收益率
yield_v = zeros(size(x));
yield_v(2:end,:) = x(2:end,:)./x(1:end-1,:)-1;

ind_ind = find(tref_num>=t1,1);
T = find(tref_num>=t2,1)-1;
%结果初始化
signal_v = zeros(T,1); %信号 协整
signal_UN = zeros(T,1);%信号 非协整
state_v = zeros(T,1);  %状态
process_para = zeros(T,2);%窗口，std
indicator_UN = get_UNCN_value(yield_v,window_UN);
sub_H_un = window_UN; %初始化
%执行策略
for i = ind_ind:T-1
    if eq(state_v(i),0)
        %判断协整
        sub_x_cg = x(i-window_CG+1:i,:);
        [~,pValue] = egcitest(sub_x_cg);
        if pValue<0.05
            %符合协整
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
            %谱分析计算周期
            sub_x_fre_un = x(i-window_fre+1:i,:);
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
        %协整信号改变
        signal_v(i+1) = 0;
        %计算非协整信号
        sub_std = std(indicator_UN(i-sub_H_un+1:i));
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

y1 = yield_v(1:T,1).*signal_v;
y2 = yield_v(1:T,2).*-signal_v;

ybac1 = 0.5*cumprod(1+y1(ind_ind:end))+0.5*cumprod(1+y2(ind_ind:end));
ybac1_d = [0.5*cumprod(1+y1(ind_ind:end)),0.5*cumprod(1+y2(ind_ind:end))];

signal_com = signal_v+signal_UN;
y3 = yield_v(1:T,1).*signal_com;
y4 = yield_v(1:T,2).*-signal_com;

ybac2 = 0.5*cumprod(1+y3(ind_ind:end))+0.5*cumprod(1+y4(ind_ind:end));
ybac2_d = [0.5*cumprod(1+y3(ind_ind:end)),0.5*cumprod(1+y4(ind_ind:end))];
window_fre_re = cat(2,window_fre_re,[ybac1,ybac2]);
end




plot(window_fre_re(:,2:2:end),'LineWidth',2);
sub_y = ybac2;
sub_x = tref(ind_ind:T);
sub_ind = floor(linspace(1,length(sub_y),20));
set(gca,'xtick',sub_ind);
set(gca,'xlim',[1,length(sub_y)]);
set(gca,'XTickLabel',sub_x(sub_ind));
set(gca,'XTickLabelRotation',45);
legend(cellstr(num2str(window_fre_pool')),'location','northwest')