function sub_r = get_sub_momentium(x)
    sub_code1 = x(strcmpi(x(:,3),'L0'),:);%当月;
    sub_code2 = x(strcmpi(x(:,3),'L1'),:);%次月;

    sub_code3 = x(eq(cell2mat(x(:,4)),1),:);%主力;
    sub_code4 = x(eq(cell2mat(x(:,5)),1),:);%次主力;

    sub_code5 = x(end,:);%最远月

    sub_code_pair = {sub_code1,sub_code2,sub_code3,sub_code4,sub_code5};

    sub_r = zeros(size(sub_code_pair));
    for j = 1:length(sub_code_pair)

        sub_x = sub_code_pair{j};%近月
        if ~isempty(sub_x)
            if ~isnan(sub_x{2}) || isinf(sub_x{2})
                sub_r(j) = sub_x{2};
            end
        end
    end
end