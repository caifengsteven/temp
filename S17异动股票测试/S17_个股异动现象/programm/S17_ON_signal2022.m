%{
尾盘信号
尾盘30分钟内触发信号后开仓、收盘平仓、开盘平仓
阈值包括1%、2%、3%、4%、5%

%update
由于后复权数据库只到2017年6月2日，为了能够扩展，我们使用了优矿的后复权数据。
%update 2022
拓展到最新日期

%}
function do_n = S17_ON_signal2022()
    do_n = true;
    t1 = '2017-06-02';
    t2 = '3030-07-12';

    cut_value = (1:5)/100;
    max_d = 10;
    var1 = {'symbol','tradingdate','precoloseprice','closeprice','r1','d','r2'};
    db_name = 'ycz_result';
    tb_name = 'ycz_result.s17_sta_relast_30min';
    %代码，时间，前收，现收，收盘价，时间间隔，第二日开盘价
    %获取交易日历
    sql_str = 'show tables from ycz_min_history';
    tref = fetchmysql(sql_str,2);
    del_ind = cellfun(@length,tref);
    tref = tref(eq(del_ind,8));
    tref_num = datenum(tref,'yyyymmdd');
    tref_num = sort(tref_num);
    ind = tref_num>=datenum(t1)&tref_num<=datenum(t2);
    tref_num = tref_num(ind);
    tref = cellstr(datestr(tref_num,'yyyy-mm-dd'));
    %tref0
    tref0 =yq_methods.get_tref2(tref{1},tref{end});
    %查看数据完整性
    if length(tref0) == length(tref) %数据未缺失
        %初始时间点
        fn_t0 = 'S17_t0.mat';
        if exist(fn_t0,'file')
            t0 = load(fn_t0);
            t0 = t0.t0;
        else
            t0 = t1;
        end

        tmp = fetchmysql('select tradingdate from %s order by tradingdate desc limit 1',2);
        if datenum(t0)<datenum(tmp)
            t0 = tmp;
        end

        T = length(tref);
        sql_str1 = ['select symbol,tradingdate,close from ycz_min_history.`%s` ',...
            'where  (hour(tradingdate)>=14 and minute(tradingdate)>=30) or hour(tradingdate)=15  order by symbol,tradingdate'];

        sql_str2 = 'select symbol,close from ycz_min_history.`%s` where tradingdate = ''%s 15:00:00''';
        sql_str3 = ['select symbol,open from ycz_min_history.`%s` where ',...
            'hour(tradingdate)=9 and minute(tradingdate)<=31'];
        re_all = cell(T,1);

        parfor i = 3:T-1
            if datenum(tref(i))>datenum(t0)
                sprintf('begin %d-%d',i,T)
                sub_x = fetchmysql(sprintf(sql_str1,replace(tref{i},'-','')),2);
                sub_y = fetchmysql(sprintf(sql_str2,replace(tref{i-1},'-',''),tref{i-1}),2);
                sub_x_next = fetchmysql(sprintf(sql_str3,replace(tref{i+1},'-','')),2);
                sub_symbols = unique(sub_x(:,1));


                Q = length(sub_symbols);
                sub_re = cell(Q*20,7);
                sub_re_ind = 0;
                for j = 1:Q
                    temp_v = cell2mat(sub_y(strcmp(sub_y(:,1),sub_symbols{j}(3:end)),2));
                    if isempty(temp_v)
                        temp_v = 0;
                    end
                    temp_v2 = cell2mat(sub_x_next(strcmp(sub_x_next(:,1),sub_symbols(j)),2));
                    if isempty(temp_v2)
                        temp_v2 = 0;
                    end
                    sub_sub_x_a = sub_x(strcmp(sub_x(:,1),sub_symbols(j)),:);
                    temp_v3 = sub_sub_x_a{end,end};
                    sub_sub_x = cell2mat(sub_sub_x_a(:,3));
                    sub_sub_r = [0;sub_sub_x(2:end)./sub_sub_x(1:end-1)-1];      
                    for k = 1:5
                        sub_ind = find(sub_sub_r>cut_value(k),1);
                        if ~isempty(sub_ind)
                            %代码，时间，前收，现收，收盘价，时间间隔，第二日开盘价
                            sub_sub_re = {sub_symbols{j},sub_sub_x_a{sub_ind,2},temp_v,...
                                sub_sub_x(sub_ind),temp_v3,k,temp_v2};
                            sub_re(sub_re_ind+1:sub_re_ind+size(sub_sub_re,1),:) = sub_sub_re;
                            sub_re_ind = sub_re_ind+size(sub_sub_re,1);
                        end
                    end

                    %信号2
                    for k = 1:5
                        sub_ind = find(sub_sub_r<-cut_value(k),1);
                        if ~isempty(sub_ind)
                            %代码，时间，前收，现收，收盘价，时间间隔，第二日开盘价
                            sub_sub_re = {sub_symbols{j},sub_sub_x_a{sub_ind,2},temp_v,...
                                sub_sub_x(sub_ind),temp_v3,-k,temp_v2};
                            sub_re(sub_re_ind+1:sub_re_ind+size(sub_sub_re,1),:) = sub_sub_re;
                            sub_re_ind = sub_re_ind+size(sub_sub_re,1);
                        end
                    end

                end
                sprintf('com %d-%d',i,T)
                sub_re = sub_re(1:sub_re_ind,:);    
                %conna = database('futuredata','root','liudehua','com.mysql.jdbc.Driver','jdbc:mysql://localhost:3306/futuredata?useSSL=false&');
                %datainsert(conna,tb_name,var1,sub_re);
                %close(conna);
                re_all{i} = sub_re';
            end
        end
        re_all1 = [re_all{:}]';
        if ~isempty(re_all1)
            OK = datainsert_adair(tb_name,var1,re_all1);
            t0 = tref{T-1};
            save(fn_t0,'t0');
        end
    else
        do_n = false;
        tmp = setdiff(tref0,tref);
        tmp = strjoin(tmp,',');
        sprintf('预测者数据有缺失，缺失日期为:%s',tmp)
    end
end
