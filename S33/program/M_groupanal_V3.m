%改进的月度选股，单日计算收益程序
%扩展股票池限制
%增加中性化步骤
%综合测试程序
close all
clear

print_sel = true;

re_para = [];
tt_para = [];
for neutralization_sel = 0:1 %是否中性化

    neutralization_info = containers.Map([1,0],{'中性化',''});

    for tN_sel = 1:2
        tN_info = {'方法1','方法2'};
        tN_all = {'S33.factor_cvar_month','S33.factor_cvar_month_v2'};
        tN = tN_all{tN_sel};

        for factor_sel = [1,3]
            factor_key = [1,2,3,4];
            factor_info = {'CVaR','rev-CVaR','VCVaR','rev-VCVaR'};

            for pool_id =1:3
                symbol_pool_all = {[],'000905','000300','000906','000852'};
                symbol_pool_info = {'全市场','中证500','沪深300','中证800','中证1000'};
                index_pool=symbol_pool_all{pool_id};

                title_str = {neutralization_info(neutralization_sel),tN_info{tN_sel},...
                    factor_info{factor_sel},symbol_pool_info{pool_id}};
                title_str = strjoin(title_str,'-');

                tN2 = 'yuqerdata.MktEqumAdjAfGet';
                fee = 3/1000;
                fee1 = 8/10000+5/10000;
                fee2 = fee1 + 1/1000;

                window = 60;
                g_num = 5;

                %set color
                color_para =[0 0.4470 0.7410;0.8500 0.3250 0.0980;0.9290 0.6940 0.1250;...
                            0.4940 0.1840 0.5560;0.4660 0.6740 0.1880;...
                            0.3010 0.7450 0.9330;0.6350 0.0780 0.1840];

                g_str = cell(g_num+1,1);
                for i = 1:g_num
                    g_str{i} = sprintf('第%d组',i);
                end
                g_str{end} = '对冲组';
                tref = yq_methods.get_tradingdate('2015-12-01','2020-01-13');
                tref_num = datenum(tref);
                %获取月底日期
                %last day for the month
                month_index = month(tref_num);
                month_cut = [0;find(diff(month_index))];
                month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
                month_cut_date1 = tref(month_cut(:,1));
                month_cut_date2 = tref(month_cut(:,2));

                %tref = tref(datenum(tref)<=datenum(2016,5,31));
                T = length(month_cut_date2);
                sql_str1 = ['select symbol,f_val',sprintf('%d',factor_key(factor_sel)),' from %s where tradingdate = ''%s'''];
                sql_str2 = 'select ticker,closeprice/openprice-1 from %s where enddate = ''%s''';
                sql_str3 = 'select ticker from   yuqerdata.st_info where tradedate =''%s''';
                sql_str4 = ['select ticker,listDate from yuqerdata.equget where listStatusCd !=''UN''',...
                    'and listDate is not null']; 

                sql_str5 = ['select symbol,tradeDate,closePrice/openPrice-1,chgPct from ',...
                    'yuqerdata.yq_dayprice where tradeDate >= ''%s'' and tradeDate <= ''%s''  and chgPct is not null'];

                sql_str6 = ['select symbol,log(f_mv),f_reverse,f_std,f_change from S33.factor_zxh ',...
                    'where tradingdate = ''%s'''];
                symbol_info = fetchmysql(sql_str4,2);
                symbol_listdate = datenum(symbol_info(:,2));
                r = zeros(T,g_num);
                r0 = zeros(T,1);
                r1 = r0;
                symbol_pool = cell(T,1);

                parfor i = 1:T-1

                    %因子值
                    x1 = fetchmysql(sprintf(sql_str1,tN,month_cut_date2{i}),2);

                    if pool_id>1
                        sub_t = fetchmysql(sprintf(['select tradingdate from yuqerdata.IdxCloseWeightGet ',...
                            'where tradingdate < ''%s'' and ticker = ''%s'' order by tradingdate desc limit 1'],...
                                        month_cut_date2{i},index_pool),2);
                        if isempty(sub_t)
                            sub_t = fetchmysql(sprintf(['select tradingdate from yuqerdata.IdxCloseWeightGet ',...
                                'where tradingdate >= ''%s'' and ticker = ''%s''  order by tradingdate limit 1'],...
                            month_cut_date2{i},index_pool),2);
                        end
                        sub_symbol_pool = fetchmysql(sprintf(['select symbol from yuqerdata.IdxCloseWeightGet ',...
                            'where tradingdate = ''%s'' and ticker = ''%s'''],sub_t{1},index_pool),2);
                        [~,ia] = intersect(x1(:,1),sub_symbol_pool);
                        x1 = x1(ia,:);
                    end
                    %未来一个月收益率
                    x2 = fetchmysql(sprintf(sql_str2,tN2,month_cut_date2{i+1}),2);

                    %st
                    st = fetchmysql(sprintf(sql_str3,tref{i}),2);
                    st = cellfun(@str2double,st,'UniformOutput',false);
                    st = cellfun(@(x) sprintf('%0.6d',x),st,'UniformOutput',false);
                    [~,ia] = intersect(x1(:,1),st);
                    x1(ia,:) = [];

                    %上市未满 60 日的新股
                    ind = datenum(tref{i})-symbol_listdate>window;
                    [~,ia] = intersect(x1(:,1),symbol_info(ind,1));
                    x1 = x1(ia,:);

                    [sub_symbol,ia,ib] = intersect(x1(:,1),x2(:,1));
                    x1 = x1(ia,:);
                    x2 = x2(ib,:);
                    x1_v = cell2mat(x1(:,2));
                    x2_v = cell2mat(x2(:,2));
                    ia = isnan(x1_v+x2_v);    
                    x1_v(ia,:) = [];
                    x2_v(ia,:) = [];
                    sub_symbol(ia,:) = [];
                    x1(ia,:) = [];
                    x2(ia,:) = [];
                    %中性化步骤
                    if eq(neutralization_sel,1)
                        warning('off');
                        %industry code
                        sub_f_ner = fetchmysql(sprintf(sql_str6,month_cut_date2{i}),2);
                        x_indus = yq_methods.get_industry_class_2(month_cut_date2{i});
                        inds = suscc_intersect({x1(:,1),x2(:,1),sub_f_ner(:,1),x_indus(:,1)});
                        sub_symbol = sub_symbol(inds(:,1));
                        x1 = x1(inds(:,1),:);
                        x2 = x2(inds(:,2),:);
                        sub_f_ner=sub_f_ner(inds(:,3),:);
                        x_indus = x_indus(inds(:,4),:);

                        x1_v = cell2mat(x1(:,2));
                        x2_v = cell2mat(x2(:,2));
                        sub_f_ner_v = cell2mat(sub_f_ner(:,2:end));
                        x_indus_v = cell2mat(x_indus(:,2));        
                        dummy_f = yq_methods.trans_dummy(x_indus_v(:,end));        
                        %regress
                        %[~,~,x1_v] = regress(x1_v,[ones(size(x1_v)),sub_f_ner_v,dummy_f]);
                        [~,~,x1_v] = regress(x1_v,[ones(size(x1_v)),sub_f_ner_v(:,[1,2]),dummy_f]); 
                    end

                    r1(i) = corr(x1_v,x2_v,'type','Spearman');
                    [~,ia] = sort(x1_v);
                    sub_t = floor(length(ia)/g_num);
                    temp_pool = cell(1,g_num);
                    for j = 1:g_num
                        if ~eq(j,g_num)
                            sub_w = (j-1)*sub_t+1:j*sub_t;
                        else
                            sub_w = (j-1)*sub_t+1:length(ia);
                        end
                        r(i+1,j) = mean(x2_v(ia(sub_w)));
                        temp_pool{j} = sub_symbol(ia(sub_w));
                    end
                    r0(i+1) = mean(x2_v)-fee;    
                    symbol_pool{i+1} = temp_pool;
                    if print_sel
                        sprintf('%d-%d',i,T)
                    end

                end
                warning('on');
                %转换为每日行情
                r_day_temp = cell(T,1);
                parfor i = 2:T
                    temp_pool = symbol_pool{i};
                    sub_t1 = month_cut_date1{i};
                    sub_t2 = month_cut_date2{i};
                    sub_t_ind = find(tref_num>=datenum(sub_t1) &tref_num<=datenum(sub_t2));
                    sub_tref = tref(sub_t_ind);
                    sub_x = fetchmysql(sprintf(sql_str5,sub_t1,sub_t2),2);
                    sub_symbol = unique(sub_x(:,1));
                    sub_r = zeros(length(sub_symbol),length(sub_tref));
                    for j = 1:length(sub_symbol)
                        sub_sub_x = sub_x(strcmp(sub_x(:,1),sub_symbol(j)),2:end);
                        [~,ia] = intersect(sub_tref,sub_sub_x(:,1));
                        sub_sub_x_v = cell2mat(sub_sub_x(:,2:end));
                        sub_sub_x_v = [sub_sub_x_v(1,1);sub_sub_x_v(2:end,2)]; %损失放在第一天
                        sub_sub_x_v(1) = sub_sub_x_v(1) -fee1; %买入时手续费
                        sub_sub_x_v(end)= sub_sub_x_v(end)-fee2; %卖出时手续费
                        sub_r(j,ia) = sub_sub_x_v;
                    end
                    sub_r_g_num = zeros(g_num,length(sub_tref));
                    for j = 1:g_num
                        [~,ia,ib] = intersect(temp_pool{j},sub_symbol);
                        temp_r = zeros(length(temp_pool{j}),length(sub_tref));
                        temp_r(ia,:) = sub_r(ib,:);
                        sub_r_g_num(j,:) = mean(temp_r);
                    end
                    r_day_temp{i} = {sub_t_ind,sub_r_g_num'};
                    if print_sel
                        sprintf('day_return %d-%d',i,T)
                    end
                end

                T = length(tref);
                r_day = zeros(T,g_num);
                for i = 2:length(r_day_temp)
                    sub_re = r_day_temp{i};
                    r_day(sub_re{1},:) = sub_re{2};
                end
                ia = tref_num>=datenum(month_cut_date2{1});
                tref = tref(ia);
                tref_num = tref_num(ia);
                r_day = r_day(ia,:);


                t_str = tref;
                T=length(t_str);
                r_c = cumprod(1+r_day);
                r_2 = cumprod(1+r_day(:,end)-r_day(:,1));
                figure
                yyaxis  left
                obj1 = plot(r_c,'-','LineWidth',2);
                set(gca,'xlim',[0,T]);
                set(gca,'XTick',floor(linspace(1,T,15)));
                yyaxis right
                obj2=plot(r_2,'-','LineWidth',2);
                obj = [obj1;obj2];
                for i = 1:size(obj,1)
                    obj(i).Color = color_para(i,:);
                end

                set(gca,'xlim',[0,T]);
                set(gca,'XTick',floor(linspace(1,T,15)));
                set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
                set(gca,'XTickLabelRotation',90)    
                setpixelposition(gcf,[223,365,1345,420]);
                box off
                legend(g_str,'NumColumns',g_num+1,'Location','best');
                title(title_str);

                figure
                bpcure_plot_updateV2(tref,r_2)
                setpixelposition(gcf,[223,365,1345,420]);
                title(title_str);

                [v,v_str,sta_val] = curve_static(r_2);
                re = [mean(r1);mean(r1/std(r1));sta_val.sharp;sta_val.drawdown;sta_val.nh];

                %%%%%%%%%%%%%%%
                t_str = month_cut_date2;
                T=length(t_str);
                r_c = cumprod(1+r);
                r_2 = cumprod(1+r(:,end)-r(:,1));

                %验证需要，勿删
                % figure
                % yyaxis  left
                % obj1 = plot(r_c,'-','LineWidth',2);
                % set(gca,'xlim',[0,T]);
                % set(gca,'XTick',floor(linspace(1,T,15)));
                % yyaxis right
                % obj2=plot(r_2,'-','LineWidth',2);
                % obj = [obj1;obj2];
                % for i = 1:size(obj,1)
                %     obj(i).Color = color_para(i,:);
                % end
                % set(gca,'xlim',[0,T]);
                % set(gca,'XTick',floor(linspace(1,T,15)));
                % set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
                % set(gca,'XTickLabelRotation',90)    
                % setpixelposition(gcf,[223,365,1345,420]);
                % box off
                % legend(g_str,'NumColumns',g_num+1,'Location','best');
                % title(title_str);

                figure
                bar(mean(r)-mean(r0))
                setpixelposition(gcf,[223,365,1345,420]);
                title(title_str);

                figure
                bar(r1)
                set(gca,'xlim',[0,T]);
                set(gca,'XTick',floor(linspace(1,T,15)));

                set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
                set(gca,'XTickLabelRotation',90)    
                setpixelposition(gcf,[223,365,1345,420]);
                title(title_str);
                box off
                re = re';
                re_para = cat(1,re_para,re);
                tt_para = cat(1,tt_para,{title_str});
            end
        end
    end
end
%显示所有统计参数
re = [{' ','RankIC','ICIR','夏普','最大回撤','年化收益'};tt_para,num2cell(re_para)]