%com
T_tref = length(tref);
y_bac = zeros(T_tref,1);
N = 20;

ind_ini = find(sum(y_re,2),1);

for i = ind_ini:N:T_tref
    %选定数据
    ind_sel = ~eq(y_re(i,:),0);
    %获取收益率数据,并平均
    sub_ind = i:(i+N-1);
    sub_ind(sub_ind>T_tref) = [];
    sub_y_r = y_re(sub_ind,ind_sel);
    y_bac(sub_ind) = mean(sub_y_r,2);    
end

y_bac_t = cumprod(y_bac+1);
%plot(tref_num,y_bac_t,'LineWidth',3);
bpcure_plot_updateV2(tref_num,y_bac_t);
