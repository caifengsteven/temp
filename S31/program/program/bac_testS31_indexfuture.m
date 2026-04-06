%股指期货回测框架
%tref signal_val 时间和信号
%r每日的开盘收益率
function y_r = bac_testS31_indexfuture(tref,signal_val,r,fee)
    if nargin < 4
        fee = 0;
    end
    T_tref = length(tref);
    y_r = zeros(T_tref,1);
    for i = 2:T_tref
        sub_r = cell2mat(r(strcmp(r(:,1),tref(i)),2));
        if eq(signal_val(i),-1)
            if ~eq(signal_val(i-1),-1) %做空,开始
                y_r(i) = sub_r*signal_val(i-1)-fee;
            else
                y_r(i) = sub_r*signal_val(i); %继续做空
            end
        elseif eq(signal_val(i),1)
            if ~eq(signal_val(i-1),1) %做多，开始建仓
                y_r(i) = sub_r*signal_val(i-1)-fee;
            else
                y_r(i) = sub_r*signal_val(i); %继续做多
            end
        else
            if ~eq(signal_val(i-1),0)
                y_r(i) = sub_r*signal_val(i-1)-fee;
            else
                y_r(i) = 0;
            end
        end

    end

end