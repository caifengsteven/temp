classdef factor_preprocessing< handle
    methods(Static)
        %remove outliers according median crition
        function y1 = median_outlier_remove(y,delta)
            if nargin < 2
                delta=5;
            end
            DM = median(y);
            DM1 = median(abs(y-DM));
            y1 = y;
            y1(y1>DM+delta*DM1) = DM+delta*DM1;
            y1(y1<DM-delta*DM1) = DM-delta*DM1;
        end
        %get factor data according datecut
        function x=  get_factor_data(tn,t1,t2)
            sql_str = ['select symbol,tradingdate,f_val from factors_com.%s ',...
                'where tradingdate>=''%s'' and tradingdate<=''%s'' and f_val is not null'];
            x = fetchmysql(sprintf(sql_str,tn,t1,t2),2);
        end
        %时间前退N个月
        function x=  get_factor_data_1(tn,t1)
            sql_str = ['select symbol,tradingdate,f_val from factors_com.%s ',...
                'where tradingdate=''%s'' and f_val is not null'];
            x = fetchmysql(sprintf(sql_str,tn,t1),2);
        end
        %symbol比较，去掉或者选相同的
        function ia = symbol_com(s1,s2,mod)
            if eq(mod,1)
                [~,ia] = intersect(s1,s2,'stable');
            else
                [~,ia] = setdiff(s1,s2,'stable');
            end
        end
    end
end