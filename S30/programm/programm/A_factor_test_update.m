%{
因子测试框架

要求
1因子数据格式为symbol，tradingdate，f_val
2数据输入为日度数据（月度也可以）

流程
1读入数据
2转换月度数据
3HP滤波转换因子
4均值滤波转换因子
5分别从全市场、300、500股票池验证因子
结果包括多空曲线参数，IC值
%}

clear
%因子测试参数
print_sel = 1; %是否屏幕输出进度  0-不输出进度
draw_sel = 0; %是否作图 0不做图 1 作图
indicator_pools = {'000300','000905'};
indicato_str = {'300股票池','500股票池','全股票池'};
tns_str = {'滤波前','窗口12','窗口24','窗口36','窗口48','窗口60','HP滤波'};

dN = 'yuqer_cubdata';
tns_pool = fetchmysql(sprintf('show tables from %s',dN),2);
re = [];
%数据源设置
for tns_sel = 1:length(tns_pool)
    tn = sprintf('%s.%s',dN,tns_pool{tns_sel});
    [OK,~] = check_table_format(dN,tns_pool{tns_sel});
    if ~OK
        continue
    end
    %检查数据格式是否正确
    
    tn_single = strsplit(tn,'.');
    tN = sprintf('S30.%s',tn_single{2});

    exemysql(sprintf('drop table %s',tN))

    %如果有数据，直接清空重算

    var_info = {'symbol','tradingdate','w','f_val'};
    var_type = cell(size(var_info));
    var_type(:) = {'float'};
    var_type(1:3) = {'varchar(6)','date','int'};
    %key_var = {'symbol','tradingdate'};
    key_var = strjoin(var_info([1,2,3]),',');
    %key_var = var_info{1};
    create_table_adair('S30',tn_single{2},var_info,var_type,key_var)

    sql_symbol = 'select distinct(symbol) from %s';
    sql_tref = 'select distinct(tradingdate) from %s  order by tradingdate';

    symbol = fetchmysql(sprintf(sql_symbol,tn),2);
    tref0 = fetchmysql(sprintf(sql_tref,tn),2);
    tref_num0 = datenum(tref0);
    load tref
    tref_num = datenum(tref);
    
    tref_ind = tref_num>=min(tref_num0)&tref_num<=max(tref_num);
    tref = tref(tref_ind);
    tref_num = tref_num(tref_ind);
    freq_sel = 1; %1-day, 2-month
    %转换月度数据
    if eq(freq_sel,1)
        %last day for the month
        month_index = month(tref_num);
        month_cut = [0;find(diff(month_index))];
        month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
        %month_cut_date1 = tref(month_cut(:,1));
        month_cut_date2 = tref(month_cut(:,2));
    else
        month_cut_date2 = tref;
    end

    %滤波1-5为均值滤波，6为HP滤波
    T = length(symbol);
    sql_str = 'select tradingdate,f_val from %s where symbol=''%s'' and f_val is not null order by tradingdate';

    parfor i = 1:T
        sub_symbol = symbol{i};
        sub_x = fetchmysql(sprintf(sql_str,tn,sub_symbol),2);
        %转换为月度
        [sub_x_v,sub_tref] = yq_methods.find_near_data(month_cut_date2,sub_x(:,1),cell2mat(sub_x(:,2)));
        sub_tref = cellstr(datestr(sub_tref,'yyyy-mm-dd'));
        sub_T = size(sub_x_v,1);
        for j = 1:6
            if ~eq(j,6)
                sub_w = 12*j;
            else
                sub_w = 12*5;
            end
            sub_y = nan(sub_T,1);
            if eq(j,1)
                sub_y0=sub_y;
            end
            for k = sub_w:sub_T
                sub_wid = k-sub_w+1:k;
                temp_y = sub_x_v(sub_wid);
                if ~eq(j,6)
                    temp = mean(temp_y);%趋势项
                else
                    temp = whitsm(temp_y,129600);%趋势项
                end
                sub_y(k) = temp_y(end)-temp(end);
                if eq(j,1)
                    sub_y0(k) = temp_y(end);
                end
            end
            f = [sub_tref(:,[1,1,1]),num2cell(sub_y)];
            f(:,1) = symbol(i);
            f(:,3) = {j};
            f = f(sub_w:end,:);
            %将原始数据的月度数据写入数据库
            if eq(j,1)
                f0 = [sub_tref(:,[1,1,1]),num2cell(sub_y0)];
                f0(:,1) = symbol(i);
                f0(:,3) = {0};
                f0 = f0(sub_w:end,:);
                f = cat(1,f0,f);
            end
            %to mysql
            if ~isempty(f)
                conna = mysql_conn();
                datainsert(conna,tN,var_info,f);
                close(conna);            
            end
            sprintf('因子滤波 Complete: %d-%d,%d',j,i,T)
        end
    end

    IC_EM = [];
    IC_EM_var = [];
    for pool_sel = 1:3  %股票池选择  1-300   2-500 3-全股
        sub_indicato_str = indicato_str{pool_sel};
        if pool_sel < 3
            indicator_pool = indicator_pools{pool_sel};
        end
        for w_sel = 1:7 %滤波参数
            sub_window_str = tns_str{w_sel};
            title_str = sprintf('计算%s-%s结果',sub_indicato_str,sub_window_str)
            IC_EM_var = cat(2,IC_EM_var,{title_str});
            tn_name = tN;
            month_cut_date2 = fetchmysql(sprintf('select distinct(tradingdate) from %s where w = 5',tN),2);
    %         %tref = fetchmysql('select distinct(tradeDate) from yuqerdata.yq_dayprice order by tradeDate',2);
    %         load tref
    %         tref_num = datenum(tref);
    % 
    %         sel_ind = tref_num>=datenum(2010,1,1);
    %         tref = tref(sel_ind);
    %         tref_num = tref_num(sel_ind);
    %         %last day for the month
    %         month_index = month(tref_num);
    %         month_cut = [0;find(diff(month_index))];
    %         month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
    %         month_cut_date1 = tref(month_cut(:,1));
    %         month_cut_date2 = tref(month_cut(:,2));

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

            T = length(month_cut_date2);
            sql_str = ['select symbol,f_val from %s where tradingdate = ''%s'' and f_val is not null and w=',num2str(w_sel-1)];
            sql_str_f2 = 'select symbol,f_val from S30.mv_month where tradingdate=''%s'' and f_val is not null';
            symbolpool = cell(T,1);
            symbolpool_f = cell(T,1);
            sql_str_f3 = ['select Stkcd from gta_web.gta_idx_smprat where Indexcd = ''',indicator_pool,''' and Enddt = ''%s'''];

            parfor i = 1:T-1
                warning('off')
                tref_sec = month_cut_date2{i};
                tref_sec_num = datenum(tref_sec);
                %横截面数据
                x = fetchmysql(sprintf(sql_str,tn_name,tref_sec),2);
                %st等数据
                sub_st_symbol = x_st_symbol(tref_sec_num>=x_st_date0&tref_sec_num<=x_st_date1);
                sub_st_symbol = cellfun(@(x) sprintf('%0.6d',x),sub_st_symbol,'UniformOutput',false);
                [~,ia] = intersect(x(:,1),sub_st_symbol);
                x(ia,:) = [];
                %股票池限制
                if pool_sel<3
                    x_ind = fetchmysql(sprintf(sql_str_f3,tref_sec),2);
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
                %update 如果市值因子，中性化需要去掉市值
                if abs(corr(f,sub_market_value,'type','Spearman'))>0.95
                    [~,~,r] = regress(f,[ones(size(f)),dummy_v]);
                else
                    [~,~,r] = regress(f,[ones(size(f)),dummy_v,sub_market_value]);
                end
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
                tref_sec = month_cut_date2{i};
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
            [v,v_str,sta_val] = curve_static(cumprod(1+r_f),12);
            v([1:5,8,11:13]) = v([1:5,8,11:13]) * 100;
            y_static = [v_str',num2cell(v')];

            if draw_sel>0
                figure 
                subplot(2,1,1)
                yyaxis left
                plot(cumprod(1+r_f),'LineWidth',3)
                yyaxis right
                bar(r_f);
                set(gca,'xlim',[0,T]);
                set(gca,'XTick',floor(linspace(1,T,15)));
                t_str = month_cut_date2(floor(linspace(1,T,15)));
                set(gca,'XTickLabel',t_str);
                set(gca,'XTickLabelRotation',90)
                box off
                title(sprintf('分组多空曲线-%s',title_str))

                subplot(2,1,2)
                bar(IC)
                set(gca,'xlim',[0,T]);
                set(gca,'XTick',floor(linspace(1,T,15)));
                t_str = month_cut_date2(floor(linspace(1,T,15)));
                set(gca,'XTickLabel',t_str);
                set(gca,'XTickLabelRotation',90)
                legend({'IC','IC-adj'},'Location','best','NumColumns',2)
                title(sprintf('IC-%s',title_str))

                setpixelposition(gcf,[223,365,1345,420*1.8]);
                movegui(gcf,'center')
            end

            [~,~,~,temp] = ttest(IC);
            L = mean(cellfun(@length,symbolpool));
            IC_re_v = [mean(IC);std(IC);min(IC);max(IC);temp.tstat;[L,L];sum(IC>0)];
            IC_re_var={'平均','标准差','最小值','最大值','t值','平均股票数','有效期数'}';
            IC_re = [{'','IC','adj_IC'};IC_re_var,num2cell(IC_re_v)];
            IC_EM = cat(2,IC_EM,[v';IC_re_v(:,end)]);
            disp(IC_re)
        end
    end
    IC_EM_column = [v_str';IC_re_var];
    sub_re = [[{'股票池'},IC_EM_var];IC_EM_column,num2cell(IC_EM)]';
    sub_re = sub_re(:,[1,1:end]);
    sub_re(:,1) = {tn_single{2}};
    sub_re(1,1) = {'因子名称'};
    if eq(tns_sel,1)
        re = cat(1,re,sub_re);
    else
        re = cat(1,re,sub_re(2:end,:));
    end
end
fn = sprintf('Factor_test%s.xlsx',datestr(now,'yymmddHHMM'));
xlswrite(fn,re)
sprintf('结果已经保存至%s',fn)