%股指期货回测框架
%tref signal_val 时间和信号
%r每日的开盘收益率
function y_r = bac_testS31_etf(tref,signal_val,r,fee)
    if nargin < 4
        fee = 0;
    end
    T_tref = length(tref);
    y_r = zeros(size(T_tref));
    for i = 2:T_tref
        sub_r = cell2mat(r(strcmp(r(:,1),tref(i)),2));
        if eq(signal_val(i),0)
            if eq(signal_val(i-1),1) %清仓
                y_r(i) = sub_r-fee;
            else
                y_r(i) = 0;
            end
        else
            if eq(signal_val(i-1),0) %建仓
                y_r(i) = 0-fee;
            else
                y_r(i) = sub_r;
            end
        end

    end

end