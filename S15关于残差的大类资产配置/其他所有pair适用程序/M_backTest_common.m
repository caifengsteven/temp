%{
通用回测程序
%}

clear

%载入数据
load dataset3.mat
breakdown_ind =[394,487,583,1016,1355,1703,1999,2161];
N_pool = (1:12)*20;
%参数
%协整窗口2month
window_CG = 20;
%谱分析窗口2年
window_fre = 20*24;
%协整 信号/std 比率
signal_cri = 2;
%非协整评价值计算窗口
window_UN = 260;
%非协整 信号/std 比率
signal_UN_cri = 2;
t1 = datenum(2012,1,1);
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
for i = ind_ind-100:T-1
    if eq(state_v(i),0)
        %判断协整
        sub_x_cg = x(i-window_CG+1:i,:);
        [~,pValue] = egcitest(sub_x_cg);
        if pValue<0.07
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

plot([ybac1,ybac2],'LineWidth',2);
sub_y = ybac2;
sub_x = tref(ind_ind:T);
sub_ind = floor(linspace(1,length(sub_y),20));
set(gca,'xtick',sub_ind);
set(gca,'xlim',[1,length(sub_y)]);
set(gca,'XTickLabel',sub_x(sub_ind));
set(gca,'XTickLabelRotation',45);
legend({'M1','M2'},'location','northwest')