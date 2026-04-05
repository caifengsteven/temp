%获取非协整条件下判断值
%原文对窗口、阈值没有说明
function y = get_UNCN_value(x1,window)
    T = size(x1,1);
    y = zeros(T,1);
    for i = window:T
        sub_wid = i-window+1:i;
        sub_x = x1(sub_wid,:);
        %rab = sub_x;
        rab= cumprod(1+sub_x)-1;
        sub_p = polyfit(sub_x(:,2),sub_x(:,1),1);
        y(i) = (rab(end,1)-rab(end,2)*sub_p(1))/(std(sub_x(:,1)-sub_x(:,2)));
    end
end