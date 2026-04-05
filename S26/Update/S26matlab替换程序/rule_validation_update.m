function [y,y1] = rule_validation_update(re_rule1)
    T = size(re_rule1,1);

    sql_str = 'select tradingdate,f_val from S26.F1 where symbol=''%s'' and tradingdate>=''%s'' order by tradingdate';
    sql_str2 = 'select * from yuqerdata.yq_tradingdate where tradingdate>=''%s'' order by tradingdate';
    re = cell(T,1);
    parfor i = 1:T

        sub_rule = re_rule1(i,:);
        sub_symbol = sub_rule{2};
        sub_t = sub_rule{1};

        sub_t_ref = fetchmysql(sprintf(sql_str2,sub_t),2);
        sub_t_ref = sub_t_ref(1:252);

        sub_sql_str = sprintf(sql_str,sub_symbol,sub_t);    
        sub_x = fetchmysql(sub_sql_str,2);

        sub_re = nan(252,1);
        if ~isempty(sub_x)
            [~,ia,ib] = intersect(sub_t_ref,sub_x(:,1),'stable');        
            if ~isempty(ib)
                sub_re(ia) = cell2mat(sub_x(ib,2));
            end
        end

        re{i} = sub_re;
        sprintf('%d-%d',i,T)

    end

    y = nan(252,T);
    for i = 1:T
        sub_y = re{i};
        y(1:length(sub_y),i) = sub_y;
    end

    y1 = zeros(252,1);
    for i = 1:252
        sub_y = y(i,:);
        sub_y = sub_y(~isnan(sub_y));
        sub_y = sub_y(abs(sub_y)<=0.1);
        if ~isempty(sub_y)
            y1(i) = mean(sub_y);
        end
    end

    yyaxis left
    plot(cumprod(1+y1),'LineWidth',2)
    yyaxis right
    bar(y1)
end
