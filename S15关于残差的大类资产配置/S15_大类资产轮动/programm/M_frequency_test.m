clear
x = 1:150;
y = sin(x*pi/10)';

window = 30;
N_pool = 10:5:30;
y_sig = zeros(length(y),length(N_pool));
y_mean = y_sig;
y_sig2 = y_sig;
for i = window:length(y)
    for j = 1:length(N_pool)
        sub_N = N_pool(j);
        y_mean(i,j) = mean(y(i-sub_N+1:i));
        y_sig(i,j) = std(y(i-sub_N+1:i)-mean(y(i-sub_N+1:i)));
    end
    
end
for i = 1:length(N_pool)
    temp_mean = movmean(y,[N_pool(i)-1,0]);
    temp_mean(1:window-1)=0;
    for j = window:length(y)        
        y_sig2(j,i) = std(y(j-N_pool(i)+1:j)-temp_mean(j));
    end
end
y_sig3 = cal_freq(y,N_pool,window);

subplot(1,2,1)
plot(x,y)
subplot(1,2,2)
plot(x(window:end),y_sig(window:end,:));
legend(cellstr(num2str(N_pool)))