%{
符合策略
%}

function [sub_w,k_score] = get_idiosyncractic_signal(tref,y_re,r_re1,r_re2,r_re3,r_re4,r_re5,vol_re)
%key_str = '打分策略';

mod = 3;%3因子还是5因子
R = 30;
H = 10;
%com
T_tref = length(tref);
m_num = 5;
m_num_2 = floor(H/m_num);
y_bac = zeros(T_tref,m_num);
ind_ini = find(sum(y_re,2),1);
if ind_ini<R
    ind_ini = (R+1);
end
i = T_tref;

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

%ind_sel1 = find(k_score<0);
%ind_sel2 = find(k_score>0);

%归一化多、空权重
sub_w = zeros(size(k_score));
sub_w(k_score>0) = k_score(k_score>0)./sum(k_score(k_score>0));
sub_w(k_score<0) = k_score(k_score<0)./sum(k_score(k_score<0));


