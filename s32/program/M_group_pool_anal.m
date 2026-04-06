%IC ICIR t_value 
%got date and cal IC
%中性化
%限定股票池
clear
%close all
%parameters
print_sel = true;
tN_pool = {'S32.factor_apm','S32.factor_q','S32.s32_factor_inverse','S32.com_factor'};
tN_key_pool = {'apm','q','inver','all'};
tN_dir = [1,-1,-1,1];
tN_dir_str = {' ','-','-',' '};
group_num = 5;
window = 60;

sql_str4 = ['select ticker,listDate from yuqerdata.equget where listStatusCd !=''UN''',...
    'and listDate is not null']; 
symbol_info = fetchmysql(sql_str4,2);
symbol_listdate = datenum(symbol_info(:,2));
T_id_sel = length(tN_pool);

for id_sel = 1:length(tN_pool)
    tn_fullname = tN_pool{id_sel};
    sub_tN_dir = tN_dir(id_sel);

    tN2 = 'yuqerdata.MktEqumAdjAfGet';

    %是否中性化 0否，1是
    neutralization_sel = 1;
    %手续费
    fee = 3/1000;
    %股票池选择
    symbol_pool_all = {[],'000300','000905','000906','000852'};
    symbol_pool_info = {'全市场','沪深300','中证500','中证800','中证1000'};
    T_pool = length(symbol_pool_all);
    re = cell(T_pool,1);
    sql_str3 = 'select ticker from   yuqerdata.st_info where tradedate =''%s''';

    for pool_id = 1:T_pool

        symbol_pool=symbol_pool_all{pool_id};
        sub_pool_info = symbol_pool_info{pool_id};

        %month_cut
        month_cut_date = fetchmysql(sprintf('select distinct(tradingdate) from %s where tradingdate >=''2014-08-01''',tn_fullname),2);
        tref = fetchmysql(sprintf('select distinct(endDate) from %s order by endDate',tN2),2);
        month_cut_date = intersect(month_cut_date,tref);

        T_month_cut = length(month_cut_date);
        sql_str_f = 'select symbol,f_val from %s where tradingdate = ''%s''';
        sql_str2 = 'select ticker,chgPct from %s where enddate = ''%s''';
        ic = zeros(T_month_cut,1);
        p = ic;
        Y = cell(T_month_cut,1);
        parfor i = 1:T_month_cut-1
            warning off
            %factor data
            x = fetchmysql(sprintf(sql_str_f,tn_fullname,month_cut_date{i}),2);        
            %股票池限定
            if ~isempty(symbol_pool)

                sub_t = fetchmysql(sprintf('select tradingdate from yuqerdata.IdxCloseWeightGet where tradingdate < ''%s'' and ticker = ''%s'' order by tradingdate desc limit 1',...
                    month_cut_date{i},symbol_pool),2);
                if isempty(sub_t)
                    sub_t = fetchmysql(sprintf(...
                    'select tradingdate from yuqerdata.IdxCloseWeightGet where tradingdate >= ''%s'' and ticker = ''%s''  order by tradingdate limit 1',...
                    month_cut_date{i},symbol_pool),2);
                end
                sub_symbol_pool = fetchmysql(sprintf('select symbol from yuqerdata.IdxCloseWeightGet where tradingdate = ''%s'' and ticker = ''%s''',sub_t{1},symbol_pool),2);
                [~,ia] = intersect(x(:,1),sub_symbol_pool);
                x = x(ia,:);
            end
           
            %st
            st = fetchmysql(sprintf(sql_str3,month_cut_date{i}),2);
            st = cellfun(@str2double,st,'UniformOutput',false);
            st = cellfun(@(x) sprintf('%0.6d',x),st,'UniformOutput',false);
            [~,ia] = intersect(x(:,1),st);
            x(ia,:) = [];
            %上市未满 60 日的新股
            ind = datenum(month_cut_date{i})-symbol_listdate>window;
            [~,ia] = intersect(x(:,1),symbol_info(ind,1));
            x = x(ia,:);
            
            %return 
            %y = fetchmysql(sprintf(sql_str_f,tn_1month_return,month_cut_date{i}),2);
            y = fetchmysql(sprintf(sql_str2,tN2,month_cut_date{i+1}),2);
            if neutralization_sel>0
                %industry
                sub_indus_code = yq_methods.get_industry_class(month_cut_date{i});
                %市值
                sub_mv = yq_methods.get_market_value_lt(month_cut_date{i});
                %对齐
                inds = suscc_intersect({x(:,1),y(:,1),sub_indus_code(:,1),sub_mv(:,1)});
                x = x(inds(:,1),:);
                y = y(inds(:,2),:);
                sub_indus_code = sub_indus_code(inds(:,3),:);
                sub_mv = sub_mv(inds(:,4),:);
                x_v = cell2mat(x(:,2));
                y_v = cell2mat(y(:,2));
                sub_mv = cell2mat(sub_mv(:,2));
                sub_indus_code = cell2mat(sub_indus_code(:,2));
                %中性化
                %哑变量矩阵
                u_sub_sub_x1 = unique(sub_indus_code);
                sub_sub_x1_yb = zeros(length(sub_indus_code),length(u_sub_sub_x1));
                for j = 1:length(u_sub_sub_x1)
                    sub_sub_x1_yb(eq(sub_indus_code,u_sub_sub_x1(j)),j) = 1;
                end
                sub_sub_x_f = [ones(size(x_v)),sub_sub_x1_yb,sub_mv];
                
                [~,~,x_v] = regress(x_v,sub_sub_x_f);
            else
                [~,ia,ib] = intersect(x(:,1),y(:,1));
                x_v = cell2mat(x(ia,2));
                y_v = cell2mat(y(ib,2));
            end

            [ic(i+1),p(i+1)] = corr(x_v,y_v,'Type','Spearman');

            [~,ia] = sort(x_v);
            y_v = y_v(ia);
            ind_cut = floor(linspace(0,length(y_v),group_num+1));
            temp = zeros(group_num,1);
            for j = 1:length(ind_cut)-1
                temp(j) = mean(y_v((ind_cut(j)+1):ind_cut(j+1)));
            end
            Y{i+1} = temp;
            if print_sel
                sprintf('%d-%d %d-%d %d-%d',i,T_month_cut,pool_id,T_pool,id_sel,T_id_sel)
            end
        end
        warning on
        Y=[Y{:}]';

        y_curve = cumprod(1+Y);
        nh_all = zeros(group_num,1);
        for i = 1:group_num
            [~,~,sta_val] = curve_static_month(y_curve(:,i));
            nh_all(i) = sta_val.nh*100;
        end
        r_month = (Y(:,end)-Y(:,1))*sub_tN_dir-fee;
        y_curve_end = cumprod(1+r_month);

        re{pool_id} = {y_curve,nh_all,r_month,y_curve_end};
    end
    for i = 1:T_pool
        if eq(i,1)
            y = zeros(length(re{i}{end}),T_pool);
        end
        y(:,i) = re{i}{end};
        [v,v_str,sta_val] = curve_static(y(:,i),12);
    end


    t_str = month_cut_date;
    T=length(t_str);
    figure
    plot(y,'LineWidth',2);
    set(gca,'xlim',[0,T+0.5]);
    set(gca,'XTick',floor(linspace(1,T,15)));

    set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
    set(gca,'XTickLabelRotation',90)    
    setpixelposition(gcf,[223,365,1345,420]);
    box off
    legend(symbol_pool_info,'NumColumns',T_pool,'Location','northwest')
    title(tN_key_pool{id_sel});
end