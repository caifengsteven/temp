%消除起点影响
ini_ind_all = ini_ind0:(ini_ind0+H-1);
Y = zeros(T,H);
for ii = 1:length(ini_ind_all)
    Y_bac = zeros(T,T1);    
    ini_ind = ini_ind_all(ii);
    Y_bac(ini_ind-1,:) = ini_cash/T1/H;
    for i = ini_ind:H:T
        sub_wid_volume_test = i-20:i;
        sub_wid = i:(i+H-1); %窗口
        sub_wid(sub_wid>T) = [];
        %准备数据和信号
        ind_test = false(T1,1);
        for j = 1:T1
            sub_volume = a_volume(sub_wid,j);
            if all(sub_volume>10000)&&tref_num(i)-list_date_num(j)>365/2            
               ind_test(j) = true; 
            end
        end
        sub_ini=sum(Y_bac(i-1,:))/sum(ind_test);
        for j = 1:T1
            if ind_test(j)         
                sub_price_close = a_close_price(sub_wid,j);
                signal_direction = a_close_price(i,j)>a_close_price(i-R,j);
                sub_main_code = a_main_code_name(sub_wid,j);
                sub_price = a_close_price(sub_wid,j);
                sub_ind_sel = ~eq(sub_price,0);
                sub_signal = ones(size(sub_price));
                if ~signal_direction
                    sub_signal = -sub_signal;
                end
                for k = 2:length(sub_signal)
                    if ~strcmp(sub_main_code(k),sub_main_code(k-1))
                        sub_signal(k) = 0;
                    end
                end
                sub_signal(end) = 0;
                sub_signal=[0;sub_signal(sub_ind_sel)];
                sub_y_bac = future_bac_method(sub_ini,asure_v,multipler_V(j),...
                    use_ratio,fee,sub_price(sub_ind_sel),sub_signal);
                Y_bac(sub_wid(sub_ind_sel),j) = sub_y_bac(2:end);
            end
        end
        sprintf('%d-%d',i,T)
    end
    Y(:,ii) = sum(Y_bac,2);
end
figure;
Y2 = sum(Y(ini_ind0:end,:),2);
Y2(1:H) = ini_cash;
Y2 = Y2./Y2(1);
sub_t = tref_num(ini_ind:end);
bpcure_plot_updateV2(sub_t,Y2);