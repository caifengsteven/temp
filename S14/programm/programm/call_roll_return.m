function r = call_roll_return(sub_code_n,sub_code_d,f_detail_date,f_detail_date_num,t0)
    ia = strcmpi(f_detail_date(:,1),sub_code_n(1));
    sub_t_n = f_detail_date_num(ia)-t0;
    if length(sub_t_n)>1
        sub_t_n=sub_t_n(1);
    end
    ib = strcmpi(f_detail_date(:,1),sub_code_d(1));
    sub_t_d = f_detail_date_num(ib)-t0;
    if length(sub_t_d)>1
        sub_t_d = sub_t_d(1);
    end
    r = (log(sub_code_n{2})-log(sub_code_d{2}))*365/(sub_t_d-sub_t_n);
end