function sub_r = get_sub_roll_return(x,f_detail_date,f_detail_date_num,tref_num)
    sub_code1 = x(strcmpi(x(:,3),'L0'),:);%当月;
    sub_code2 = x(strcmpi(x(:,3),'L1'),:);%次月;

    sub_code3 = x(eq(cell2mat(x(:,4)),1),:);%主力;
    sub_code4 = x(eq(cell2mat(x(:,5)),1),:);%次主力;

    sub_code5 = x(end,:);%最远月


    sub_code_pair = {sub_code1,sub_code2;sub_code1,sub_code3;sub_code1,sub_code5;sub_code3,sub_code4};

    sub_r = zeros(4,1);
    for j = 1:4

        sub_code_n = sub_code_pair{j,1};%近月
        sub_code_d = sub_code_pair{j,2};%远月
        if ~isempty(sub_code_n)&&~isempty(sub_code_d)
            sub_r(j) = call_roll_return(sub_code_n,sub_code_d,f_detail_date,f_detail_date_num,tref_num);
        else
            sub_r(j) = nan;
        end
    end
end