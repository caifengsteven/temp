%开盘买，收盘卖
%加入止损
function V = bac_test_CTA_update4(ind,o_c_price,code_para,share_num,cash0,fee)

    T = length(ind);
    V = zeros(T,1);
    V(1:2) = cash0;
    stop_not = false;%是否止损
    temp_cash0=[0,cash0];
    for i = 3:T
        if ~stop_not
            if eq(ind(i-1),0)
                if eq(ind(i-2),0)
                    V(i) = V(i-1);
                else
                    %开盘平仓
                    V(i) = V(i-1)+(o_c_price(i,1)-o_c_price(i-1,2))*ind(i-2)*code_para*share_num(i-2)-...
                        o_c_price(i,1)*code_para*fee*share_num(i-2); 
                end
            elseif eq(ind(i-1),1)
                if eq(ind(i-2),0)
                    %早盘建仓
                    temp_cash0 = [ind(i-1),V(i-1)];
                    V(i)= V(i-1)-o_c_price(i,1)*code_para*fee*share_num(i-1);
                    %尾盘统计收益
                    V(i)=V(i)+(o_c_price(i,2)-o_c_price(i,1))*ind(i-1)*code_para*share_num(i-1);
                elseif eq(ind(i-2),1)
                    %持有  只统计收益
                    V(i) = V(i-1)+(o_c_price(i,2)-o_c_price(i-1,2))*ind(i-1)*code_para*share_num(i-1); 
                else
                    %信号相反，早盘平仓并建仓，尾盘统计
                    %早盘平仓
                    V(i) = V(i-1)+(o_c_price(i,1)-o_c_price(i-1,2))*ind(i-2)*code_para*share_num(i-2)-...
                        o_c_price(i,1)*code_para*fee*share_num(i-2);                    
                    %早盘建仓
                    temp_cash0 = [ind(i),V(i)];
                    V(i) = V(i) - o_c_price(i,1)*code_para*fee*share_num(i-1);
                    %尾盘统计
                    V(i) = V(i)+(o_c_price(i,2)-o_c_price(i,1))*ind(i-1)*code_para*share_num(i-1);
                end

            else
                %-1
                if eq(ind(i-2),0)
                    %开盘建仓
                    temp_cash0 = [ind(i-1),V(i-1)];
                    V(i)= V(i-1)-o_c_price(i,1)*code_para*fee*share_num(i-1);
                    %尾盘统计收益
                    V(i)=V(i)+(o_c_price(i,2)-o_c_price(i,1))*ind(i-1)*code_para*share_num(i-1);
                elseif eq(ind(i-2),-1)
                    %持有  只统计收益
                    V(i) = V(i-1)+(o_c_price(i,2)-o_c_price(i-1,2))*ind(i-1)*code_para*share_num(i-1); 
                else
                    %信号相反，早盘平仓并建仓，尾盘统计
                    %早盘平仓
                    V(i) = V(i-1)+(o_c_price(i,1)-o_c_price(i-1,2))*ind(i-2)*code_para*share_num(i-2)-...
                        o_c_price(i,1)*code_para*fee*share_num(i-2);                    
                    %早盘建仓
                    temp_cash0 = [ind(i),V(i)];
                    V(i) = V(i) - o_c_price(i,1)*code_para*fee*share_num(i-1);
                    %尾盘统计
                    V(i) = V(i)+(o_c_price(i,2)-o_c_price(i,1))*ind(i-1)*code_para*share_num(i-1);
                end
            end
            %判断是否止损
            if ~eq(temp_cash0(1),0)
                if eq(temp_cash0(1),ind(i-1))
                    if V(i)/temp_cash0(2)-1<-0.03
                        stop_not = true;
                    end
                end
            end

        else
            V(i) = V(i-1);
        end
        
        %判断是否止损
        if ~eq(temp_cash0(1),0) || ~eq(ind(i),temp_cash0(1))
            stop_not = false;
        end
        
    end

end