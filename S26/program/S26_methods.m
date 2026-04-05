classdef S26_methods<handle
    methods(Static)
        function re_rule1 = rule1()
            %
            key_str = 'S26财务规则1';
            window = 3;
            obj_yq = yq_methods();
            x = obj_yq.get_HeBingLiRun('NIncomeAttrp');
            x = x(strcmp(x(:,5),'A'),:);

            sql_str = 'SELECT ticker,publishDate FROM s26.equrestructuringget where isSucceed=1 order by publishDate';
            y=fetchmysql(sql_str,2);

            symbol_N = unique(y(:,1));
            T = length(symbol_N);
            re_rule1 = cell(T,1);
            parfor i = 1:T    
                sub_x = x(strcmp(x(:,1),symbol_N(i)),:);
                [~,ia] = unique(sub_x(:,3),'stable');
                sub_x = sub_x(ia,:);
                sub_symbol = sprintf('%0.6d',str2double(symbol_N(i)));
                sub_y = y(strcmp(y(:,1),sub_symbol),:);
                if ~isempty(sub_y)
                    sub_y_t = datenum(sub_y(:,2));
                else
                    sub_y_t = 0;
                end

                sub_t = sub_x(:,2);
                sub_t_num = datenum(sub_t);
                sub_v = cell2mat(sub_x(:,6));
                [sub_t_num,ia] = sort(sub_t_num);
                sub_v = sub_v(ia,:);
                sub_t = sub_t(ia);

                sub_T = length(sub_t);
                sub_re = cell(sub_T,1);
                for j = window+1:sub_T
                    sub_window_v = sub_v(j-window:j-1);
                    if sub_v(j)/median(sub_window_v)<0.2 && sub_v(j)<10e6 && sub_v(j)>0 &&sum(sub_y_t>=sub_t_num(j)-365*2&sub_y_t<=sub_t_num(j))>0 && sub_t_num(j) >datenum(2010,1,1)
                        %
                        %re_rule1 = cat(1,re_rule1,[sub_t(j),sub_symbol]);
                        sub_re{j} = [sub_t(j),sub_symbol]';
                    end    
                end

                re_rule1(i) = {[sub_re{:}]};
                sprintf('%s:%d-%d',key_str,i,T)

            end
            re_rule1 = [re_rule1{:}]';
            insert_to_dataS26(re_rule1,1);
        end
        
        function re_rule1 = rule2()
            key_str = 'S26财务规则2';
            window = 2;
            obj_yq = yq_methods();
            %净利润，营业收入
            x = obj_yq.get_HeBingLiRun('NIncomeAttrp,revenue');
            x = x(strcmp(x(:,5),'A'),:);

            %sql_str = 'SELECT ticker,publishDate FROM s26.equrestructuringget where isSucceed=1 order by publishDate';
            %y=fetchmysql(sql_str,2);

            symbol_N = unique(x(:,1));
            T = length(symbol_N);

            re_rule1 = cell(T,1);
            parfor i = 1:T    
                sub_x = x(strcmp(x(:,1),symbol_N(i)),:);
                [~,ia] = unique(sub_x(:,3),'stable');
                sub_x = sub_x(ia,:);
                sub_symbol = sprintf('%0.6d',str2double(symbol_N(i)));
                sub_t = sub_x(:,2);
                sub_t_num = datenum(sub_t);
                sub_v = cell2mat(sub_x(:,6:7));
                [sub_t_num,ia] = sort(sub_t_num);
                sub_v = sub_v(ia,:);
                sub_t = sub_t(ia);

                sub_T = length(sub_t);
                sub_re = cell(sub_T,1);
                for j = window+1:sub_T
                    sub_window_v = sub_v(j-window:j-1,1);
                    if all(sub_window_v<0) && sub_v(j,2)/sub_v(j-1,2)>0.2 && sub_t_num(j) >datenum(2010,1,1)
                        sub_re{j} = [sub_t(j),sub_symbol]';
                    end    
                end
                re_rule1(i) = {[sub_re{:}]};


                sprintf('%s:%d-%d',key_str,i,T)

            end
            re_rule1 = [re_rule1{:}]';
            insert_to_dataS26(re_rule1,2);
        end
        
        function re_rule1 = rule3()
            key_str = 'S26财务规则3';
            XF1 = yq_methods.get_HeBingLiRun('finanExp');
            XF2 = yq_methods.get_HeBingZiChanFuZhai('cashCEquiv');

            del_ind1 = cellfun(@isnan,XF1(:,end));
            XF1(del_ind1,:) = [];
            XF1 = XF1(strcmp(XF1(:,5),'A'),:);
            del_ind2 = cellfun(@isnan,XF2(:,end));
            XF2(del_ind2,:) = [];

            symbols = unique(XF1(:,1));
            T = length(symbols);
            re = cell(T,1);
            parfor i = 1:T
                sub_xf1 = XF1(strcmp(XF1(:,1),symbols(i)),:);
                sub_xf1 = flipud(sub_xf1);
                sub_symbol = sprintf('%0.6d',str2double(symbols{i}));
                sub_xf2 = XF2(strcmp(XF2(:,1),sub_symbol),:);
                sub_xf2 = flipud(sub_xf2);
                [~,ia] = unique(sub_xf2(:,3),'stable');
                sub_xf2 = sub_xf2(ia,:);
                sub_xf2_v = cell2mat(sub_xf2(:,end));
                %sub_xf2_v = diff(sub_xf2_v);
                %sub_xf2 = sub_xf2(2:end,:);
                sub_xf2_v = movmean(sub_xf2_v,[4-1,0]);
                sub_xf2(:,end) = num2cell(sub_xf2_v);

                [~,ia,ib] = intersect(sub_xf1(:,3),sub_xf2(:,3));
                sub_xf12 = [sub_xf1(ia,:),sub_xf2(ib,end)];
                sub_v = cell2mat(sub_xf12(:,end-1))./cell2mat(sub_xf12(:,end));

                ind = 1.35 - abs(sub_v)*100>1;
                if any(ind)
                    sub_re = sub_xf2(ind,[2,2]);
                    sub_re(:,2) = {sub_symbol};
                    re{i} = sub_re';
                end
                sprintf('%s: %d-%d',key_str,i,T)
            end
            re_rule1 = [re{:}]';

            ia = datenum(re_rule1(:,1));
            ia = ia>=datenum(2010,1,1);
            re_rule1 = re_rule1(ia,:);
            
        end
        
        function re_rule1 = rule4()
            key_str = 'S26财务规则4';
            obj_yq = yq_methods();
            %净利润，营业收入
            x = obj_yq.get_HeBingZiChanFuZhai('AR,TCA');

            symbol_N = unique(x(:,1));
            T = length(symbol_N);

            re_rule1 = cell(T,1);
            parfor i = 1:T    
                sub_x = x(strcmp(x(:,1),symbol_N(i)),:);
                [~,ia] = unique(sub_x(:,3),'stable');
                sub_x = sub_x(ia,:);
                sub_symbol = symbol_N(i);
                ia = sum(cell2mat(sub_x(:,6:7)),2);
                sub_x(isnan(ia),:) = [];
                if isempty(sub_x)
                    continue
                end
                sub_t = sub_x(:,2);
                sub_t_num = datenum(sub_t);
                sub_v = cell2mat(sub_x(:,6:7));
                [sub_t_num,ia] = sort(sub_t_num);
                sub_v = sub_v(ia,:);
                sub_v = sub_v(:,1)./sub_v(:,2);
                sub_t = sub_t(ia);

                sub_T = length(sub_t);
                sub_re = cell(sub_T,1);
                for j = 1:sub_T
                    if sub_v(j)>0.8 && sub_t_num(j) >datenum(2010,1,1)% && sub_t_num(j)<datenum(2016,1,1)
                        %

                        %re_rule1 = cat(1,re_rule1,[sub_t(j),sub_symbol]);
                        sub_re{j} = [sub_t(j),sub_symbol]';
                    end    
                end
                re_rule1{i} = [sub_re{:}];

                sprintf('%s:%d-%d',key_str,i,T)

            end
            re_rule1 = [re_rule1{:}]';
            insert_to_dataS26(re_rule1,4);
        end
        
        function re_rule1 = rule5()
            key_str = 'S26财务规则5';
            %
            window = 2;
            obj_yq = yq_methods();
            %主营业务收入，应收账款
            x = fetchmysql('select Stkcd,Accper,B110101 from gtadata.FAR_Finidx where B110101 is not null and B110101!=0 order by Accper,Annodt',2);
            %x = obj_yq.get_HeBingLiRun('revenue-othGain'); %营业收入-其他收益
            %x = obj_yq.get_YeJiKuaiBao('primeOperRev');
            y = obj_yq.get_HeBingZiChanFuZhai('AR'); %应收账款
            %x = x(strcmp(x(:,5),'A'),:);
            y = y(strcmp(y(:,5),'A'),:);
            del_ind = cellfun(@isnan,y(:,end));
            y(del_ind,:) = [];
            %合并数据

            xid = cellfun(@(x,y) [x,',',y],x(:,1),x(:,2),'UniformOutput',false);
            yid = cellfun(@(x,y) [x,',',y],y(:,1),y(:,3),'UniformOutput',false);
            [~,ia,ib] = intersect(xid,yid);
            z = [y(ib,1:end-1),x(ia,end),y(ib,end)];
            symbol_N = unique(z(:,1));

            T  = length(symbol_N);
            re_rule1 = cell(T,1);
            parfor i = 1:T    
                sub_x = z(strcmp(z(:,1),symbol_N(i)),:);
                [~,ia] = unique(sub_x(:,3),'stable');
                sub_x = sub_x(ia,:);
                sub_symbol = sprintf('%0.6d',str2double(symbol_N(i)));
                ia = sum(cell2mat(sub_x(:,6:7)),2);
                sub_x(isnan(ia),:) = [];
                if isempty(sub_x)
                    continue
                end
                sub_t = sub_x(:,2);
                sub_t_num = datenum(sub_t);
                sub_v = cell2mat(sub_x(:,6:7));
                [sub_t_num,ia] = sort(sub_t_num);
                sub_v = sub_v(ia,:);
                sub_t = sub_t(ia);

                sub_T = length(sub_t);
                %YOY(主营业务收入)>0 AND YOY(应收账款)>0.5 AND YOY(应收账款)/ YOY(主营业务收入)>3
                sub_re = cell(sub_T,1);
                for j = window:sub_T
                    sub_YOY = sub_v(j,:)./sub_v(j-1,:)-1;
                    if sub_YOY(1)>0 && sub_YOY(2)>0.5 && sub_YOY(2)/sub_YOY(1)>3 && sub_t_num(j) >datenum(2010,1,1) %&& sub_t_num(j)<datenum(2016,1,1)
                        %
                        %re_rule1 = cat(1,re_rule1,[sub_t(j),sub_symbol]);
                        sub_re{j} = [sub_t(j),sub_symbol]';
                    end    
                end

                re_rule1{i} = [sub_re{:}];
                sprintf('%s:%d-%d',key_str,i,T)

            end
            re_rule1 = [re_rule1{:}]';
            
        end
        
        function re_rule1 = rule6()
            key_str = 'S26财务规则6';
            %
            window = 2;
            obj_yq = yq_methods();
            x = obj_yq.get_HeBingXianJinLiu('CFrOthOperateA'); %收到其他与经营活动有关的现金
            %x = obj_yq.get_YeJiKuaiBao('primeOperRev');
            y = obj_yq.get_HeBingZiChanFuZhai('NotesReceiv,TCA'); %应收票据，流动资产
            x = x(strcmp(x(:,5),'A'),:);
            y = y(strcmp(y(:,5),'A'),:);
            %合并数据

            xid = cellfun(@(x,y) [x,',',y],x(:,1),x(:,3),'UniformOutput',false);
            yid = cellfun(@(x,y) [x,',',y],y(:,1),y(:,3),'UniformOutput',false);
            [~,ia,ib] = intersect(xid,yid);
            z = [x(ia,:),y(ib,end-1:end)];

            symbol_N = unique(x(:,1));
            T = length(symbol_N);
            
            re_rule1 = cell(T,1);
            parfor i = 1:T    
                sub_x = z(strcmp(z(:,1),symbol_N(i)),:);
                [~,ia] = unique(sub_x(:,3),'stable');
                sub_x = sub_x(ia,:);
                sub_symbol = sprintf('%0.6d',str2double(symbol_N(i)));
                ia = sum(cell2mat(sub_x(:,6:end)),2);
                sub_x(isnan(ia),:) = [];
                if isempty(sub_x)
                    continue
                end
                sub_t = sub_x(:,2);
                sub_t_num = datenum(sub_t);
                sub_v = cell2mat(sub_x(:,6:end));
                [sub_t_num,ia] = sort(sub_t_num);
                sub_v = sub_v(ia,:);
                sub_t = sub_t(ia);

                sub_T = length(sub_t);
                sub_re = cell(sub_T,1);
                %应收票据/流动资产>50% AND 收到其他与经营活动有关的现金/应收票据<0.3 AND YOY(收到其他与经营活动有关的现金/应收票据)<-50%
                %收到其他与经营活动有关的现金,应收票据,流动资产
                for j = window:sub_T

                    sub_test1 = sub_v(j,2)/sub_v(j,3);
                    sub_test2 = sub_v(j,1)/sub_v(j,2);
                    sub_test3 = sub_v(j-window+1:end,1)./sub_v(j-window+1:end,2);
                    sub_test3 = sub_test3(end)/sub_test3(end-1)-1;
                    if sub_test1>0.5 && sub_test2<0.3 && sub_test3<-0.5 && sub_t_num(j) >datenum(2010,1,1) %&& sub_t_num(j)<datenum(2016,1,1)
                        %
                        %re_rule1 = cat(1,re_rule1,[sub_t(j),sub_symbol]);
                        sub_re{j} = [sub_t(j),sub_symbol]';
                    end    
                end
                re_rule1{i} = [sub_re{:}];
                sprintf('%s:%d-%d',key_str,i,T)

            end
            re_rule1 = [re_rule1{:}]';
        end
        
        function re_rule1 = rule7()
            key_str = 'S26财务规则7';
            window = 2;
            obj_yq = yq_methods();
            x = obj_yq.get_HeBingLiRun('NIncomeAttrP'); %归属于母公司的净利润
            %x = obj_yq.get_YeJiKuaiBao('primeOperRev');
            y = obj_yq.get_HeBingZiChanFuZhai('othReceiv,TCA'); %其他应收款，流动资产
            x = x(strcmp(x(:,5),'A'),:);
            y = y(strcmp(y(:,5),'A'),:);
            %合并数据
            xid = cellfun(@(x,y) [x,',',y],x(:,1),x(:,3),'UniformOutput',false);
            yid = cellfun(@(x,y) [x,',',y],y(:,1),y(:,3),'UniformOutput',false);
            [~,ia,ib] = intersect(xid,yid);
            z = [x(ia,:),y(ib,end-1:end)];
            symbol_N = unique(z(:,1));

            T = length(symbol_N);
            re_rule1 = cell(T,1);
            parfor i = 1:T    
                sub_x = z(strcmp(z(:,1),symbol_N(i)),:);
                [~,ia] = unique(sub_x(:,3),'stable');
                sub_x = sub_x(ia,:);
                sub_symbol = sprintf('%0.6d',str2double(symbol_N(i)));
                ia = sum(cell2mat(sub_x(:,6:end)),2);
                sub_x(isnan(ia),:) = [];
                if isempty(sub_x)
                    continue
                end
                sub_t = sub_x(:,2);
                sub_t_num = datenum(sub_t);
                sub_v = cell2mat(sub_x(:,6:end));
                [sub_t_num,ia] = sort(sub_t_num);
                sub_v = sub_v(ia,:);
                sub_t = sub_t(ia);

                sub_T = length(sub_t);
                %其他应收款/流动资产>10% AND YOY（其他应收款/流动资产）>10 AND 归母净利润>0
                %归属于母公司的净利润,其他应收款，流动资产 
                sub_re = cell(sub_T,1);
                for j = window:sub_T

                    sub_test1 = sub_v(j,2)/sub_v(j,3);
                    sub_test2 = sub_v(:,2)./sub_v(:,3);
                    sub_test2 = sub_test2(end)/sub_test2(end-1)-1;        
                    sub_test3 = sub_v(j,1);

                    if sub_test1>0.1 && sub_test2>10 && sub_test3>0 && sub_t_num(j) >datenum(2010,1,1) && sub_t_num(j)<datenum(2016,1,1)
                        %
                        sub_re{j} = [sub_t(j),sub_symbol]';
                        %sub_num=sub_num+1;
                        %re_rule1 = cat(1,re_rule1,[sub_t(j),sub_symbol]);
                    end    
                end
                re_rule1{i} = [sub_re{:}];
                sprintf('%s:%d-%d',key_str,i,T)

            end
            re_rule1 = [re_rule1{:}]';
            insert_to_dataS26(re_rule1,7);
        end
        
        function re_rule1 = rule8()
            key_str = 'S26财务规则8';
            window = 2;
            obj_yq = yq_methods();
            %净利润，营业收入
            x = obj_yq.get_HeBingZiChanFuZhai('othCA,TCA');

            symbol_N = unique(x(:,1));
            T = length(symbol_N);
            re_rule1 = cell(T,1);
            parfor i = 1:T    
                sub_x = x(strcmp(x(:,1),symbol_N(i)),:);
                [~,ia] = unique(sub_x(:,3),'stable');
                sub_x = sub_x(ia,:);
                sub_symbol = symbol_N(i);
                ia = sum(cell2mat(sub_x(:,6:7)),2);
                sub_x(isnan(ia),:) = [];
                if isempty(sub_x)
                    continue
                end
                sub_t = sub_x(:,2);
                sub_t_num = datenum(sub_t);
                sub_v = cell2mat(sub_x(:,6:7));
                [sub_t_num,ia] = sort(sub_t_num);
                sub_v = sub_v(ia,:);
                sub_v = sub_v(:,1)./sub_v(:,2);
                sub_t = sub_t(ia);

                sub_T = length(sub_t);
                sub_re = cell(sub_T,1);
                for j = window:sub_T
                    sub_test1 = sub_v(j);
                    sub_test2 = sub_v(j)/sub_v(j-1)-1;
                    if sub_test1>0.8 &&sub_test2>10 && sub_t_num(j) >datenum(2010,1,1) && sub_t_num(j)<datenum(2016,1,1)
                        sub_re{j} = [sub_t(j),sub_symbol]';
                        %sub_num=sub_num+1;
                        %re_rule1 = cat(1,re_rule1,[sub_t(j),sub_symbol]);
                    end    
                end
                re_rule1{i} = [sub_re{:}];
                sprintf('%s:%d-%d',key_str,i,T)

            end
            re_rule1 = [re_rule1{:}]';            
            insert_to_dataS26(re_rule1,8);
        end
        
        function re_rule1 = rule9()
            key_str = 'S26财务规则9';
            tref = yq_methods.get_tradingdate('2010-01-01',datestr(now,'yyyy-mm-dd'));
            sql_str = ['select symbol,f_val from S26.f_nrProfitLoss ',...
                'where tradingdate=''%s'' and f_val is not null'];

            T = length(tref);
            re_rule1 = cell(T,1);
            parfor i = 1:T
                x = fetchmysql(sprintf(sql_str,tref{i}),2);
                if isempty(x)
                    continue
                end
                obj_yq = yq_methods();
                indus_code = obj_yq.get_industry_class(tref{i});

                [~,ia,ib] = intersect(x(:,1),indus_code(:,1),'stable');
                X = [x(ia,:),indus_code(ib,end)];
                X_v = cell2mat(X(:,2:end));

                indus_code_u = unique(X_v(:,2));
                sub_T= length(indus_code_u);
                indus_mx = zeros(sub_T,1);
                indus_mx_all = zeros(size(X_v(:,2)));
                indus_rank = indus_mx_all;
                for j = 1:sub_T
                    temp_ind = eq(X_v(:,2),indus_code_u(j));
                    indus_mx(j) = mean(X_v(temp_ind,1))+std(X_v(temp_ind,1))*3; 
                    indus_mx_all(temp_ind) = indus_mx(j);
                    indus_rank(temp_ind) = prctile(X_v(temp_ind,1),0.03*100);
                end

                %ind = X_v(:,1)>indus_mx_all & X_v(:,1)<prctile(X_v(:,1),0.03*100);
                ind = X_v(:,1)>indus_mx_all & X_v(:,1)>indus_rank;
                if any(ind)
                    temp = X(ind,[1,1]);
                    temp(:,1) = tref(i);
                    re_rule1{i} = temp';
                end

                sprintf('%s: %d-%d',key_str,i,T)
            end
            re_rule1 = [re_rule1{:}]';
            %[~,ia ] = unique(re_rule1(:,2));
            %re_rule1 = re_rule1(ia,:);
        end
        
        function re_rule1 = rule10()
            key_str = 'S26财务规则10';
            tref = yq_methods.get_tradingdate('2010-01-01',datestr(now,'yyyy-mm-dd'));
            sql_str = ['select ticker,GrossIncomeRatio from S26.yq_mktstockfactorsonedayget_add_s26 ',...
                'where tradedate=''%s'' and GrossIncomeRatio is not null'];

            T = length(tref);
            re_rule1 = cell(T,1);
            t0 = get_t0_S26(10);
            num0 = find(strcmp(tref,t0));
            parfor i = num0+1:T
                x = fetchmysql(sprintf(sql_str,tref{i}),2);
                if isempty(x)
                    continue
                end
                obj_yq = yq_methods();
                indus_code = obj_yq.get_industry_class(tref{i});

                [~,ia,ib] = intersect(x(:,1),indus_code(:,1),'stable');
                X = [x(ia,:),indus_code(ib,end)];
                X_v = cell2mat(X(:,2:end));

                indus_code_u = unique(X_v(:,2));
                sub_T= length(indus_code_u);
                indus_mx = zeros(sub_T,1);
                indus_mx_all = zeros(size(X_v(:,2)));
                indus_rank = indus_mx_all;
                for j = 1:sub_T
                    temp_ind = eq(X_v(:,2),indus_code_u(j));
                    indus_mx(j) = mean(X_v(temp_ind,1))+std(X_v(temp_ind,1))*3; 
                    indus_mx_all(temp_ind) = indus_mx(j);
                    indus_rank(temp_ind) = prctile(X_v(temp_ind,1),0.03*100);
                end

                %ind = X_v(:,1)>indus_mx_all & X_v(:,1)<prctile(X_v(:,1),0.03*100);
                ind = X_v(:,1)>indus_mx_all & X_v(:,1)>indus_rank;
                if any(ind)
                    temp = X(ind,[1,1]);
                    temp(:,1) = tref(i);
                    re_rule1{i} = temp';
                end

                sprintf('%s:%d-%d',key_str,i,T)
            end

            re_rule1 = [re_rule1{:}]';
            insert_to_dataS26(re_rule1,10);
            x = get_his_data(10);
            re_rule1 = [x(:,2:end);re_rule1];
            
            %[~,ia] = unique(re_rule1(:,2));
            %re_rule1 = re_rule1(ia,:);
        end
        
        function re_rule1 = rule11()
            key_str = 'S26财务规则11';
            tref = yq_methods.get_tradingdate('2010-01-01',datestr(now,'yyyy-mm-dd'));
            sql_str = ['select ticker,OperCashInToCurrentLiability from S26.yq_mktstockfactorsonedayget_add_s26 ',...
                'where tradedate=''%s'' and GrossIncomeRatio is not null'];

            T = length(tref);
            re_rule1 = cell(T,1);
            parfor i = 1:T
                x = fetchmysql(sprintf(sql_str,tref{i}),2);
                if isempty(x)
                    continue
                end
                obj_yq = yq_methods();
                indus_code = obj_yq.get_industry_class(tref{i});

                [~,ia,ib] = intersect(x(:,1),indus_code(:,1),'stable');
                X = [x(ia,:),indus_code(ib,end)];
                X_v = cell2mat(X(:,2:end));

                indus_code_u = unique(X_v(:,2));
                sub_T= length(indus_code_u);
                indus_mx = zeros(sub_T,1);
                indus_mx_all = zeros(size(X_v(:,2)));
                indus_rank = indus_mx_all;
                for j = 1:sub_T
                    temp_ind = eq(X_v(:,2),indus_code_u(j));
                    indus_mx(j) = mean(X_v(temp_ind,1))+std(X_v(temp_ind,1))*3; 
                    indus_mx_all(temp_ind) = indus_mx(j);
                    indus_rank(temp_ind) = prctile(X_v(temp_ind,1),0.03*100);
                end

                %ind = X_v(:,1)>indus_mx_all & X_v(:,1)<prctile(X_v(:,1),0.03*100);
                ind = X_v(:,1)>indus_mx_all & X_v(:,1)>indus_rank;
                if any(ind)
                    temp = X(ind,[1,1]);
                    temp(:,1) = tref(i);
                    re_rule1{i} = temp';
                end

                sprintf('%s:%d-%d',key_str,i,T)
            end

            re_rule1 = [re_rule1{:}]';
            %[~,ia] = unique(re_rule1(:,2));
            %re_rule1 = re_rule1(ia,:);
        end
        function re_rule1 = rule12()
            key_str = 'S26财务规则12';
            sql_str_f1 = ['select symbol,publishdate,enddate,invenTurnover ',...
                'from yuqerdata.yq_FdmtIndiTrnovrPitGet order by endDate,publishdate'];
            XF1 = fetchmysql(sql_str_f1,2);
            %XF2 = fetchmysql(sql_str_f2,2);
            XF2 = yq_methods.get_YingLiNengLi('grossMARgin');
            XF2 = flipud(XF2(:,[1,3,4]));
            %合并数据
            symbols = unique(XF1(:,1));
            T = length(symbols);
            re_rule1 = cell(T,1);
            parfor i = 1:T
                sub_xf1 = XF1(strcmp(XF1(:,1),symbols(i)),:);
                sub_xf2 = XF2(strcmp(XF2(:,1),symbols(i)),:);
                [~,ia,ib] = intersect(sub_xf1(:,3),sub_xf2(:,2),'stable');
                sub_xf12 = [sub_xf1(ia,:),sub_xf2(ib,end)];
                if isempty(sub_xf12)
                    continue
                end
                %mark    
                sub_t = datenum(sub_xf12(:,3));
                sub_v = cell2mat(sub_xf12(:,4:5));
                sub_T = length(sub_t);
                sub_re = cell(sub_T,1);
                for j = 1:sub_T
                    sub_ind = sub_t>=sub_t(j)-365*3-40&sub_t<=sub_t(j);
                    sub_test = sub_v(sub_ind,:);
                    if isempty(sub_test)
                        continue
                    end
                    sub_sub_t = sub_t(sub_ind);
                    if sub_sub_t(end)-sub_sub_t(1)<365*3-8
                        continue
                    end
                    sub_test2 = sub_v(sub_ind,:);
                    %计算近3年的数值
                    sub_test3 = zeros(3,2);
                    temp_ind = 1;
                    for k = 1:3
                        sub_ind2 = find(sub_sub_t>=sub_sub_t(1)+365*k-8,1);
                        sub_test3(k,:) = sub_test2(sub_ind2,:)./sub_test2(temp_ind,:)-1;
                        temp_ind = sub_ind2;
                    end
                    %YOY3(存货周转率)<0 AND YOY3(毛利率)>0
                    if all(sub_test3(:,1)<0) && all(sub_test3(:,2)>0)
                        sub_re{j} = [sub_xf12(j,2),symbols(i)]';
                        %re_rule1 = cat(1,re_rule1,sub_re);
                    end
                end
                re_rule1{i} = [sub_re{:}];
                sprintf('%s:%d-%d',key_str,i,T)
            end
            re_rule1 = [re_rule1{:}]';
            ia = datenum(re_rule1(:,1));
            ia = ia>=datenum(2010,1,1); %& ia<=datenum(2017,1,1);
            re_rule1 = re_rule1(ia,:);
            %[~,ia] = unique(re_rule1(:,2));
            %re_rule1 = re_rule1(ia,:);
        end
        function re_rule1 = rule13()
            key_str = 'S26财务规则13';
            window = 2;
            XF1 = yq_methods.get_HeBingLiRun('NIncome');
            XF1 = XF1(strcmp(XF1(:,5),'A'),:);
            del_ind = cellfun(@isnan,XF1(:,end));
            XF1(del_ind,:) = [];

            XF2 = fetchmysql('select Stkcd,Accper,B110101 from gtadata.FAR_Finidx where B110101 is not null and B110101!=0 order by Accper,Annodt',2);
            %合并数据
            symbols = unique(XF1(:,1));
            T = length(symbols);
            re_rule1 = cell(T,1);
            parfor i = 1:T
                sub_symbol = sprintf('%0.6d',str2double(symbols(i)));
                sub_xf1 = XF1(strcmp(XF1(:,1),symbols(i)),:);
                sub_xf1 = flipud(sub_xf1);
                sub_xf2 = XF2(strcmp(XF2(:,1),sub_symbol),:);

                [~,ia,ib] = intersect(sub_xf1(:,3),sub_xf2(:,2),'stable');

                sub_xf12 = [sub_xf1(ia,:),sub_xf2(ib,end)];

                if isempty(sub_xf12)
                    continue
                end
                [~,ia] = unique(sub_xf12(:,3),'stable');
                sub_xf12 = sub_xf12(ia,:);
                sub_v = cell2mat(sub_xf12(:,end-1:end));
                sub_T = size(sub_v,1);
                sub_re = cell(sub_T,1);
                for j = window:sub_T
                    sub_YOY = sub_v(j,:)./sub_v(j-1,:)-1;
                    %((YOY(主营业务收入)-YOY(净利润)) / abs(YOY(净利润)) > 10
                    if (sub_YOY(1)-sub_YOY(2))/abs(sub_YOY(2)) > 10
                        sub_re{j} = [sub_xf12(j,2),sub_symbol]';
                        %re_rule1 = cat(1,re_rule1,sub_re);
                    end
                end
                re_rule1{i} = [sub_re{:}];
                sprintf('%s:%d-%d',key_str,i,T)
            end
            re_rule1 = [re_rule1{:}]';
            ia = datenum(re_rule1(:,1));
            ia = ia>=datenum(2010,1,1);% & ia<=datenum(2017,1,1);
            re_rule1 = re_rule1(ia,:);
        end

    end
end
function t0 = get_t0_S26(id)
    sql_str = 'select tradingdate from S26.S26_result where rule_name = %d order by tradingdate desc limit 1';
    t0 = fetchmysql(sprintf(sql_str,id),2);
    if isempty(t0)
        t0 = '1990-01-01';
    else
        t0 = t0{1};
    end
end
function insert_to_dataS26(re_rule1,id)
    if isempty(re_rule1)
        return
    end
    tn = 'S26.S26_result';   
    var_info = {'rule_name','tradingdate','symbol'};
    t0 = get_t0_S26(id);
    if isempty(t0)
        t0 = '1990-01-01';
    end
    sub_re = re_rule1(datenum(re_rule1(:,1))>datenum(t0),:);
    if ~isempty(sub_re)
        a = cellfun(@(x,y) [x,y],sub_re(:,1),sub_re(:,2),'UniformOutput',false);
        [~,ia] = unique(a,'stable');
        sub_re = sub_re(ia,[1,1:end]);
        sub_re(:,1) = {id};
        datainsert_adair(tn,var_info,sub_re);
    end
end
function x = get_his_data(id)
    tn = 'S26.S26_result'; 
    sql_str = 'select * from %s where rule_name = %d order by tradingdate';
    x = fetchmysql(sprintf(sql_str,tn,id),2);
end