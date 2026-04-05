function v = future_bac_test_openT2(index,price_oc,ini_val,multiplier,stop_losses,takeprofit)
    if nargin<5
        stop_losses = [];
    end
    if nargin<6
        takeprofit = [];
    end

    if isempty(stop_losses);stop_losses = -inf;end
    if isempty(takeprofit);takeprofit = inf;end
    T = size(index,1);
    v = zeros(T,1);
    v(1:2) = ini_val;
    for i = 3:T
        if eq(index(i-2,2),-1)
            v(i) = v(i-1)+share_index_return(-1*(price_oc(i,1)-price_oc(i-1,1)),multiplier);
        elseif eq(index(i-2,2),1)
            v(i) = v(i-1)+share_index_return(1*(price_oc(i,1)-price_oc(i-1,1)),multiplier);
        else
            v(i) = v(i-1);
        end
        %stop_loss and takeprofit
        if v(i)/v(i-1)-1<stop_losses
            v(i) = v(i-1)*(1+stop_losses);
        end
        if v(i)/v(i-1)-1>takeprofit
            v(i) = v(i-1)*(1+takeprofit);
        end
    end

    
end

function share_e=share_index_return(d_index,multiplier)
    share_e = d_index*multiplier;
end