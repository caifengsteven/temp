mod = 5;
T_tref = length(tref);
m_num = 5;
m_num_2 = floor(H/m_num);
y_bac = zeros(T_tref,m_num);
ind_ini = find(sum(y_re,2),1);
if ind_ini<R
    ind_ini = (R+1);
end
for i0 = 1:m_num
    for i = ind_ini+(i0-1)*m_num_2:H:T_tref
        %1/K
        %选定数据
        ind_sel0 = find(~eq(y_re(i,:),0)&vol_re(i,:)>10000);
        
        sub_r2 = r_re2(i-1,ind_sel0);
        sub_r3 = r_re3(i-1,ind_sel0);
        sub_r4 = r_re4(i-1,ind_sel0);
        [~,ia2] = sort(sub_r2);
        [~,ia3] = sort(sub_r3);
        [~,ia4] = sort(sub_r4);
        sub_r5 = r_re5(i-1,ind_sel0);
        [~,ia5] = sort(sub_r5);
        sub_r1 = r_re1(i-1,ind_sel0);
        [~,ia1] = sort(sub_r1);

        k_score = zeros(size(y_re(1,:)));
        for j = 1:mod
            ia = eval(sprintf('ia%d',j));
            if eq(j,1)
                k_score(ia>0) = 1;
                k_score(ia<0) = -1;
            else
                if length(ia)>=5
                    num1 = floor(length(ia)*0.2);
                    ia1 = ia(1:num1);
                    ind_sel1 = ind_sel0(ia1);
                    ia2 = ia(end-num1+1:end);
                    ind_sel2 = ind_sel0(ia2);
                    k_score(ind_sel1) = k_score(ind_sel1)-1;
                    k_score(ind_sel2) = k_score(ind_sel2)+1;
                end
            end            
        end
        
        ind_sel1 = find(k_score<0);
        ind_sel2 = find(k_score>0);
        
        %归一化多、空权重
        sub_w = zeros(size(k_score));
        sub_w(k_score>0) = k_score(k_score>0)./sum(k_score(k_score>0));
        sub_w(k_score<0) = k_score(k_score<0)./sum(k_score(k_score<0));
        if i > 300
            sub_w0 = sub_w;
            %调整权重
            sub_x = y_re(i-240:i,:);
            sigma_s = std(sub_x)./close_price(i-240:i,:);
            %less
            v1 = 0;
            for j = 1:length(ind_sel1)
                sub_sub_x = sub_x(:,ind_sel1);
                sub_v1 = sub_w(ind_sel1(j)).*sub_w(ind_sel1).*corr(sub_sub_x(:,j),sub_sub_x);
                sub_v1 = -sub_v1(j) + sum(sub_v1);
                v1 = v1 + sub_v1;
            end
            sub_w(ind_sel1) = sigma_targ*sub_w(ind_sel1)./(sigma_s(ind_sel1)*sqrt(sum(sub_w(ind_sel1).^2)+2*v1));
            %more
            v2 = 0;
            for j = 1:length(ind_sel2)
                sub_sub_x = sub_x(:,ind_sel2);
                sub_v1 = sub_w(ind_sel2(j)).*sub_w(ind_sel2).*corr(sub_sub_x(:,j),sub_sub_x);
                sub_v1 = -sub_v1(j) + sum(sub_v1);
                v2 = v2 + sub_v1;
            end
            sub_w(ind_sel2) = sigma_targ*sub_w(ind_sel2)./(sigma_s(ind_sel2)*sqrt(sum(sub_w(ind_sel2).^2)+2*v2));
            sub_w(isnan(sub_w)) = sub_w0(isnan(sub_w));
            sub_w(ind_sel1) = sub_w(ind_sel1)/sum(sub_w(ind_sel1));
            sub_w(ind_sel2) = sub_w(ind_sel2)/sum(sub_w(ind_sel2));
            if any(sub_w<0);keyboard;end
        end
        %获取收益率数据,并平均
        sub_ind = i:(i+H-1);
        sub_ind(sub_ind>T_tref) = [];
        %多
        sub_y_r_m = y_re(sub_ind,ind_sel2);    
        %手续费
        sub_y_r_m(1,:) = sub_y_r_m(1,:)-3/10000;
        sub_y_r_m(end,:) = sub_y_r_m(end,:)-3/10000;
        temp = sub_w(ind_sel2).*cumprod((1+sub_y_r_m));
        temp = [1;sum(temp,2)];
        temp_m = temp(2:end)./temp(1:end-1)-1;
        if ~isempty(ind_sel1)
            %空
            sub_y_r = y_re(sub_ind,ind_sel1);    
            %手续费
            sub_y_r([1,end],:) = sub_y_r([1,end],:);
            temp = sub_w(ind_sel1).*cumprod((1+sub_y_r));
            temp = [1;sum(temp,2)];
            temp = temp(2:end)./temp(1:end-1)-1;
        else
            temp=0;
        end
        y_bac(sub_ind,i0) = temp_m-temp;
    end
end
ind = tref_num>datenum(2010,1,1);
y_bac1 = y_bac(ind,:);
tref_num1 = tref_num(ind);

y_bac_t = 1/m_num*cumprod(y_bac1+1);
y_bac_t = sum(y_bac_t,2);
bpcure_plot_updateV2(tref_num1,y_bac_t);
[v,v_str,sta_val] = curve_static(y_bac_t)