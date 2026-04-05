function re = win_rate(signal_val3,y_r3)
temp = find(~eq(signal_val3,0));
temp(temp>=length(signal_val3)) = [];
re = sum(y_r3(temp+1)>0)./(length(temp));