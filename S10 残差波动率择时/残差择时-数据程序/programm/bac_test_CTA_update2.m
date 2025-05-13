%开盘买，收盘卖
function V = bac_test_CTA_update2(ind,o_c_price,code_para,share_num,cash0,fee)

    T = length(ind);
    V = zeros(T,1);
    V(1:2) = cash0;
    for i = 3:T
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
                V(i) = V(i) - o_c_price(i,1)*code_para*fee*share_num(i-1);
                %尾盘统计
                V(i) = V(i)+(o_c_price(i,2)-o_c_price(i,1))*ind(i-1)*code_para*share_num(i-1);
            end
            
        else
            %-1
            if eq(ind(i-2),0)
                %开盘建仓
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
                V(i) = V(i) - o_c_price(i,1)*code_para*fee*share_num(i-1);
                %尾盘统计
                V(i) = V(i)+(o_c_price(i,2)-o_c_price(i,1))*ind(i-1)*code_para*share_num(i-1);
            end
            
            
        end
        
    end

end