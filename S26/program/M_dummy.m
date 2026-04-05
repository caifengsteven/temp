clear
%para
tn_fullname = 'S26.F1';
var_info = {'symbol','tradingdate','f_val'};

tref = yq_methods.get_tradingdate('2007-01-01','2019-12-01');

T = length(tref);

sql_str = 'select symbol,chgPct from yuqerdata.yq_dayprice where tradedate=''%s'' and chgPct is not null';
sql_str_f = ['select ticker,HBETA,RSTR24,MLEV,FEARNG,EGRO,VOL20+VOL60+VOL240,Volatility,exp(LCAP),LCAP ',...
    'from S26.yq_mktstockfactorsonedayget_s26 where tradedate = ''%s'' '];
parfor i = 1:T
    obj_yq=yq_methods();
    x = fetchmysql(sprintf(sql_str,tref{i}),2);
    indus_code = obj_yq.get_industry_class(tref{i});
    f = fetchmysql(sprintf(sql_str_f,tref{i}),2);
    
    [inds,commValue] = suscc_intersect({x(:,1),indus_code(:,1),f(:,1)});
    
    sub_symbols = x(inds(:,1),1);
    y = cell2mat(x(inds(:,1),2));
    F = cell2mat(f(inds(:,3),2:end));
    del_ind = isnan(sum(F));
    F(:,del_ind) = [];
    sub_sub_x1 = cell2mat(indus_code(inds(:,2),2));
    %—∆±‰¡øæÿ’Û
    u_sub_sub_x1 = unique(sub_sub_x1);
    sub_sub_x1_yb = zeros(length(sub_sub_x1),length(u_sub_sub_x1));
    for j = 1:length(u_sub_sub_x1)
        sub_sub_x1_yb(eq(sub_sub_x1,u_sub_sub_x1(j)),j) = 1;
    end
    
    sub_sub_x_f = [ones(size(y)),sub_sub_x1_yb,F];
    warning('off','stats:regress:RankDefDesignMat')
    [~,~,resi] = regress(y,sub_sub_x_f);
    sub_re = sub_symbols(:,[1,1,1]);
    sub_re(:,2) = tref(i);
    sub_re(:,3) = num2cell(resi);
    
    if ~isempty(sub_re)
        conna = mysql_conn();
        %write data to mysql
        datainsert(conna,tn_fullname,var_info,sub_re)
        close(conna)
    end
    sprintf('%d-%d',i,T)
end
