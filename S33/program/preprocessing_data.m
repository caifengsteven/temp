%预处理程序
function sub_y = preprocessing_data(sub_code,sub_x_v)
    sub_code_u = unique(sub_code);
    sub_T = length(sub_code_u);
    sub_y = zeros(size(sub_x_v));
    %分组排序
    for k = 1:sub_T
        sub_ind = eq(sub_code,sub_code_u(k));
        sub_sub_x_v = sub_x_v(sub_ind);
        [~,~,ia] = unique(sub_sub_x_v);
        %标准化
        sub_y(sub_ind) = zscore(ia);
    end
    
    %sub_y = zscore(sub_y);
end