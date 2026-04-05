ind = find(tref_num_u<tref_num(i),1,'last');
    sub_code = w_index(eq(tref_num_index,tref_num_u(ind)),:);
    ind1 = cellfun(@(x) strcmp(x(1),'0'),sub_code(:,2));
    sub_code(ind1,2) = cellfun(@(x) ['sz',x],sub_code(ind1,2),'UniformOutput',false);
    sub_code(~ind1,2) = cellfun(@(x) ['sh',x],sub_code(~ind1,2),'UniformOutput',false);
    sub_x_t0 = fetchmysql(sprintf(sql_strb,datestr(tref_num(i-1),'yyyymmdd')),2);
    sub_x_t0 = arange_yczmin_data(sub_x_t0);%获取开盘、收盘价
    
    sub_x_t1 = fetchmysql(sprintf(sql_strb,datestr(tref_num(i),'yyyymmdd')),2);
    sub_x_t1 = arange_yczmin_data(sub_x_t1);
    
    [inds,commValue] = suscc_intersect({sub_code(:,2),sub_x_t0(:,1),sub_x_t1(:,1)});
    
    sub_x = [sub_code(inds(:,1),3),sub_x_t0(inds(:,2),2:3),sub_x_t1(inds(:,3),2:3)];
    sub_x = cell2mat(sub_x);
        
    sub_coef0 = coef_v(tref_coeff_num<=tref_num(i-1),:);
    sub_coef1 = coef_v(tref_coeff_num<=tref_num(i),:);
    
    %复权
    [~,ia,ib] = intersect(commValue,sub_coef0(:,1),'stable');
    sub_coeff0_c = ones(size(commValue));
    sub_coeff0_c(ia) = cell2mat(sub_coef0(ib,3));
    sub_x(:,2:3) = sub_x(:,2:3).*repmat(sub_coeff0_c,1,2);
    
    [~,ia,ib] = intersect(commValue,sub_coef1(:,1),'stable');
    sub_coeff1_c = ones(size(commValue));
    sub_coeff1_c(ia) = cell2mat(sub_coef1(ib,3));
    sub_x(:,4:5) = sub_x(:,4:5).*repmat(sub_coeff1_c,1,2);
    
    sub_signal_yestoday = fetchmysql(sprintf(sql_str_signal,datestr(tref_num(i-1),'yyyy-mm-dd')),2);
    sub_signal_today = fetchmysql(sprintf(sql_str_signal,datestr(tref_num(i),'yyyy-mm-dd')),2);
    %T+1限制，今日的信号和昨日相同，无法触发（早晨刚买入）
    if  ~isempty(sub_signal_today)&&~isempty(sub_signal_yestoday)
        [~,ia] = setdiff(sub_signal_today(:,1),sub_signal_yestoday(:,1));
        sub_signal_today = sub_signal_today(ia,:);
    end
    
    temp = (sub_x(:,5)./sub_x(:,3)-1).*sub_x(:,1)./100;
    temp1 = temp;
    
    if ~isempty(sub_signal_yestoday)
    [~,ia0,ib0] = intersect(sub_signal_yestoday(:,1),commValue);%昨日触发的        
    temp1(ib0) = (sub_x(ib0,5)./sub_x(ib0,4)-1).*sub_x(ib0,1)./100;%昨日触发的
    end
    if ~isempty(sub_signal_today)
        %复权        
        [~,ia1,ib1] = intersect(sub_signal_today(:,1),commValue);%今日触发的
        temp1(ib1) = (cell2mat(sub_signal_today(ia1,2)).*sub_coeff1_c(ib1)./sub_x(ib1,3)-1).*sub_x(ib1,1)./100;
    end
    %temp = (sub_x(:,3)./sub_x(:,2)-1).*sub_x(:,1)./sum(sub_x(:,1));
    y(i) = sum(temp);
    y1(i) = sum(temp1);
    sprintf('%d-%d',i,T)