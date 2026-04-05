clear
%load rbdata_update.mat t tref closeprice f codenum;ind = t>=datenum(2009,7,1);v_dir = -1;
%load L_data_update.mat t tref closeprice f codenum;ind =t>=datenum(2007,9,1);v_dir = -1;
%load AL_data_update.mat t tref closeprice f codenum;ind = t>=datenum(2005,3,1);v_dir = 1;
%load RU_data_update.mat t tref closeprice f codenum;ind = t>=datenum(2009,7,1);v_dir = -1;
%load RM_data_update.mat t tref closeprice f codenum;ind = t>=datenum(2013,2,1);v_dir = 1;
%load J_data_update.mat t tref closeprice f codenum;ind = t>=datenum(2011,8,1);v_dir = 1; %YES
%load I_data_update.mat t tref closeprice f codenum;ind = t>=datenum(2014,2,1);v_dir = -1; 
load HC_data_update.mat t tref closeprice f codenum;ind = t>=datenum(2014,9,1);v_dir = 1; 
t = t(ind);
closeprice= closeprice(ind);
r = [0;closeprice(2:end)./closeprice(1:end-1)-1];
f = v_dir*f(ind);
tref = tref(ind);
codenum = codenum(ind);

window = 10000;
T = length(closeprice);
f_p80 = zeros(T,1);
f_p20 = f_p80;
%计算20、80分位数
parfor i = window+1:T
    sub_x = f(i-window:i);
    f_p80(i) = prctile(sub_x,80);
    f_p20(i) = prctile(sub_x,20);
    sprintf('%d-%d',i,T)
end

%cal_signal
signal = zeros(T,1);
signal(1:window+2) = 0;
y1 = zeros(T,1);
y1(1:window+1) = 1; %sub signal
y2 = y1;
fee = 3/10000;
%fee = 0;
mark = 0; %是否止损、止盈或者切换合约轻仓
for i = window+2:T
    %没有持仓，判断下一步
    if eq(signal(i),0)
        start_i = i;
        if f(i)>f_p80(i) && ~eq(mark,1) %上穿，且不是止损
            signal(i+1) = 1;
            mark = 0;
        elseif f(i)<f_p20(i) && ~eq(mark,-1) %下穿，且不是止损
            signal(i+1) = -1;
            mark = 0;
        else
            signal(i+1) = signal(i);
        end
    end
    if eq(signal(i),1) %多
        if   f(i)<f_p20(i) && ~eq(mark,-1) %下穿，且不是止损
            signal(i+1) = -1;
            mark = 0;
        else
            signal(i+1) = signal(i);
        end
    end
    if eq(signal(i),-1) %空
        if f(i)>f_p80(i)  && ~eq(mark,1) %上穿，且不是止损
            signal(i+1) = 1;
            mark = 0;
        else
            signal(i+1) = signal(i);
        end
    end
    
    %是否更换合约
    if eq(codenum(i),codenum(i-1))
        if eq(signal(i),signal(i-1))
            sub_r = r(i)*signal(i);
        else
            %是否转换
            sub_r = r(i)*signal(i)-fee;
        end        
    else
        %更换合约处理
        r(i) = 0;
        sub_r = 0-fee;
        signal(i+1) = 0;
        mark = signal(i);
    end    
    %记录收益    
    if eq(signal(i),signal(i-1))
        y2(i) = y2(i-1)*(1+sub_r);        
        y1(i) = y1(i-1)*(1+sub_r);
        sub_y1 = y1(start_i:i);
        %判断是否止损、止盈
        if abs(sub_y1(end)/max(sub_y1)-1)>0.05 %%sub_y1(end)/sub_y1(1)-1<-0.05%sub_y1(end)/max(sub_y1)-1<-0.05%
            signal(i+1)=0;
            mark = signal(i);
        end
    else
        y2(i) = y2(i-1)*(1+sub_r);
        y1(i) = 1;
    end

end

%可视化
obj=plot(t,[cumprod(1+r),y2],'LineWidth',2);
obj(1).Color=[0.9294,0.6902,0.1294];
obj(2).Color=[0.6392,0.0784,0.1804];
datetick('x','yyyy');

