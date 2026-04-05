%{
1.细分因子标准化。对细分价值因子进行分位数变换标准化。
2.合成价值风格因子。将标准化之后的细分因子等权合成价值风格因子。
3.价值风格因子 中性化。对价值风格因子进行行业和市值的中性化处理。

限制股票池
pool_500 = cell(T,1);
sql_str = 'select Stkcd,Weight from gta_web.gta_idx_smprat where Indexcd = ''000905'' and Enddt = ''%s''';
parfor i = 1:T
    pool_500{i} = fetchmysql(sprintf(sql_str,month_cut_date2{i}),2);
end

普通均线代替HP滤波
%}

clear
print_sel = 0; %是否屏幕输出进度  0 不输出
tns_sel = 2; %只可以设置为2

indicator_pools = {'000300','000905',[]};
indicato_str = {'300股票池','500股票池','全股票池'};
tns = {'F_month_final','F_month_final_adj_avg'};
tns_str = {'滤波前','移动平均滤波窗口-%d'};
for pool_sel = 1:3  %股票池选择  1-300   2-500 3-全股
    sub_indicato_str = indicato_str{pool_sel};
    if pool_sel < 3
        indicator_pool = indicator_pools{pool_sel};
    end
    for w_sel = 1:5 %滤波参数
        sub_window_str = sprintf(tns_str{tns_sel},w_sel*12);
        title_str = sprintf('计算%s-%s结果',sub_indicato_str,sub_window_str);
        
        tn_name = tns{tns_sel};
        %tref = fetchmysql('select distinct(tradeDate) from yuqerdata.yq_dayprice order by tradeDate',2);
        month_cut_date = yq_methods.get_month_data();
        sel_ind = datenum(month_cut_date)>=datenum(2010,1,1);
        month_cut_date = month_cut_date(sel_ind);

        %载入ST信息数据
        sql_str = 'SELECT * FROM yuqerdata.st_info order by tradedate desc';
        x_st = fetchmysql(sql_str,2);
        x_st(:,1) = cellfun(@str2double,x_st(:,1),'UniformOutput',false);
        x_st_codenum = cell2mat(x_st(:,1));
        x_st_u_codenum = unique(x_st_codenum);
        x_st_data = cell(length(x_st_u_codenum),3);
        for i = 1:length(x_st_u_codenum)
            sub_x_st_data=x_st(eq(x_st_codenum,x_st_u_codenum(i)),:);
            x_st_data(i,:) = {sprintf('%0.6d',x_st_u_codenum(i)),sub_x_st_data{1,2},sub_x_st_data{end,2}};
        end
        x_st_symbol = x_st_data(:,1);
        x_st_date0 = datenum(x_st_data(:,3));
        x_st_date1 = datenum(x_st_data(:,2));


        T = length(month_cut_date);
        sql_str = ['select symbol,f_val from S30.%s where tradingdate = ''%s'' and f_val is not null and w=',num2str(w_sel)];
        sql_str_f2 = 'select symbol,log(negMarketValue) from yuqerdata.yq_dayprice where tradeDate=''%s'' and negMarketValue is not null';
        symbolpool = cell(T,1);
        symbolpool_f = cell(T,1);
        %sql_str_f3 = ['select Stkcd from gta_web.gta_idx_smprat where Indexcd = ''',indicator_pool,''' and Enddt = ''%s'''];

        parfor i = 1:T-1
            warning('off')
            tref_sec = month_cut_date{i};
            tref_sec_num = datenum(tref_sec);
            %横截面数据
            x = fetchmysql(sprintf(sql_str,tn_name,tref_sec),2);
            %st等数据
            sub_st_symbol = x_st_symbol(tref_sec_num>=x_st_date0&tref_sec_num<=x_st_date1);
            sub_st_symbol = cellfun(@(x) sprintf('%0.6d',x),sub_st_symbol,'UniformOutput',false);
            [~,ia] = intersect(x(:,1),sub_st_symbol);
            x(ia,:) = [];
            %股票池限制
            if ~isempty(indicator_pool)
                x_ind = yq_methods.get_index_pool(indicator_pool,tref_sec);
                %x_ind = fetchmysql(sprintf(sql_str_f3,tref_sec),2);
                [~,ia] = intersect(x(:,1),x_ind);
                x = x(ia,:);
            end
            %因子标准化 分位数标准化
            f = cell2mat(x(:,2));
            [~,~,sub_x] = unique(f);
            f = zscore(sub_x);
            %价值风格因子 中性化。对价值风格因子进行行业和市值的中性化处理。
            indus_code = yq_methods.get_industry_class(tref_sec);
            market_value = fetchmysql(sprintf(sql_str_f2,tref_sec),2);
            inds = suscc_intersect({x(:,1),indus_code(:,1),market_value(:,1)});
            sub_symbol = x(inds(:,1),1);
            f = f(inds(:,1),1);
            sub_indus_code = cell2mat(indus_code(inds(:,2),2));
            sub_market_value = cell2mat(market_value(inds(:,3),2));

            %dummy
            sub_code_v_u = unique(sub_indus_code);
            dummy_v = zeros(length(sub_indus_code),length(sub_code_v_u));
            for j = 1:length(sub_code_v_u)
                dummy_v(eq(sub_indus_code,sub_code_v_u(j)),j) = 1;
            end
            %中性化   
            [~,~,r] = regress(f,[ones(size(f)),dummy_v,sub_market_value]);
            %save
            [r,ia] = sort(r);
            symbolpool{i+1} = sub_symbol(ia);
            symbolpool_f{i+1} = [f(ia),r];
            if print_sel>0
                sprintf('symbol: %d-%d',i,T)
            end
        end

        %分组计算
        cut_num = 10;
        sql_str = 'select ticker,chgPct from yuqerdata.MktEqumAdjAfGet where endDate = ''%s''';
        sql_str_day = 'select symbol,chgPct from yuqerdata.yq_dayprice where tradeDate = ''%s''';
        Y = cell(T,1);
        IC = cell(T,1);
        parfor i = 2:T
            tref_sec = month_cut_date{i};
            x = fetchmysql(sprintf(sql_str,tref_sec),2);
            sub_symbol = symbolpool{i};
            sub_f = symbolpool_f{i};
            [~,ib,ia] = intersect(sub_symbol,x(:,1),'stable');
            r = cell2mat(x(ia,2));
            f = sub_f(ib,:);
            IC{i} = corr(r,f,'type','Spearman')';
            sub_y = zeros(cut_num,1);
            sub_l = length(r)/cut_num;
            for j = 1:cut_num
                if j<cut_num
                    sub_ind = floor((j-1)*sub_l)+1:floor(j*sub_l);
                else
                    sub_ind = floor((j-1)*sub_l)+1:length(r);
                end
                sub_y(j) = mean(r(sub_ind));
            end
            Y{i} = sub_y;
            if print_sel>0
                sprintf('return: %d-%d',i,T)   
            end
        end
        IC = [IC{:}]';
        y_r = [Y{:}]';
        r_f = y_r(:,end)-y_r(:,1);
        
        figure 
        subplot(2,1,1)
        yyaxis left
        plot(cumprod(1+r_f),'LineWidth',3)
        yyaxis right
        bar(r_f);

        [v,v_str,sta_val] = curve_static(cumprod(1+r_f),12);
        y_static = [v_str',num2cell(v')];

        set(gca,'xlim',[0,T]);
        set(gca,'XTick',floor(linspace(1,T,15)));
        t_str = month_cut_date(floor(linspace(1,T,15)));
        set(gca,'XTickLabel',t_str);
        set(gca,'XTickLabelRotation',90)
        box off
        title(sprintf('分组多空曲线-%s',title_str))

        subplot(2,1,2)
        bar(IC)
        set(gca,'xlim',[0,T]);
        set(gca,'XTick',floor(linspace(1,T,15)));
        t_str = month_cut_date(floor(linspace(1,T,15)));
        set(gca,'XTickLabel',t_str);
        set(gca,'XTickLabelRotation',90)
        legend({'IC','IC-adj'},'Location','best','NumColumns',2)
        title(sprintf('IC-%s',title_str))
        
        setpixelposition(gcf,[223,365,1345,420*1.8]);
        movegui(gcf,'center')

        [~,~,~,temp] = ttest(IC);
        L = mean(cellfun(@length,symbolpool));
        IC_re = [mean(IC);std(IC);min(IC);max(IC);temp.tstat;[L,L];sum(IC>0)];
        IC_re = [{'','IC','adj_IC'};{'平均','标准差','最小值','最大值','t值','平均股票数','有效期数'}',num2cell(IC_re)];
        disp(IC_re)
    end
end