%合成APM因子
%升级程序，终版本

clear

print_sel = true;
tN= 'S32.factor_apm';
var_info = {'symbol','tradingdate','f_val'};

tref = yq_methods.get_tradingdate('2013-05-03','2020-01-13');
tref_num = datenum(tref);

sql_str1 = 'select symbol,f_val from S32.factor_delta where tradingdate = ''%s''';

sql_str2 = 'select symbol,f_val from S32.ret20d_update where tradingdate = ''%s''';


T = length(tref);
r = zeros(T,1);
r2 = r;
parfor i = 1:T
    warning('off')
    
    x = fetchmysql(sprintf(sql_str2,tref{i}),2); %ret20
    y = fetchmysql(sprintf(sql_str1,tref{i}),2); %delta

    inds = suscc_intersect({y(:,1),x(:,1)});
    sub_symbol = x(inds(:,2),1);
    
    y = cell2mat(y(inds(:,1),2));
    x = cell2mat(x(inds(:,2),2));    
    
    [~,~,resi] = regress(y,x);
    
    sub_f = [sub_symbol,sub_symbol,num2cell(resi)];
    sub_f(:,2) = tref(i);
    
    %into mysql
    if ~isempty(sub_f)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,sub_f);
        close(conna);
    end
    
    if print_sel
        sprintf('%d-%d',i,T)
    end
end

warning('on')
