%从yuqer获取数据方法合集
classdef yq_methods< handle
    methods(Static)
        %输入t1 起点  t2 终点
        function tref = get_tradingdate_future(t1,t2)
            if nargin < 1
                t1 = [];
            end
            if nargin < 2
                t2=[];
            end
            
            if isempty(t1)
                t1 = '1980-01-01';
            end
            if isempty(t2)
                t2 = datestr(now+200,'yyyy-mm-dd');
            end
            
            sql_str = ['select tradingdate from yuqerdata.yq_tradingdate_future ',...
                'where tradingdate >= ''%s'' and tradingdate<=''%s'' order by tradingdate ;'];
            sql_str = sprintf(sql_str,t1,t2);
            tref = fetchmysql(sql_str,2);
        end
        %
        function tref = get_tradingdate(t1,t2)
            if nargin < 1
                t1 = [];
            end
            if nargin < 2
                t2=[];
            end
            
            if isempty(t1)
                t1 = '1980-01-01';
            end
            if isempty(t2)
                t2 = datestr(now,'yyyy-mm-dd');
            end
            
            sql_str = ['select tradeDate from yuqerdata.yq_index ',...
                'where tradeDate >= ''%s'' and tradeDate<=''%s'' and ',...
                'symbol = ''000001'' order by tradeDate ;'];
            sql_str = sprintf(sql_str,t1,t2);
            tref = fetchmysql(sql_str,2);
        end
        %获取月底最后一天
        function [month_cut_date1,month_cut_date2] = get_month_day(tref)
            tref_num = datenum(tref);
            month_index = month(tref_num);
            month_cut = [0;find(diff(month_index))];
            month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
            month_cut_date1 = tref(month_cut(:,1));
            month_cut_date2 = tref(month_cut(:,2));
        end
        %获取业绩快报数据
        %symbol，发布日期，会计截至日期，时间长度，类型，数值
        function x = get_YeJiKuaiBao(var_name)
            sql_str1 = ['select ticker,publishdate,enddate,fiscalPeriod,',...
                'reportType,%s from yuqerdata.yq_FdmtEeGet '];
            sql_str2 = [' where enddate>''2005-01-01'' ',...
                'and mergedFlag = 1 and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'and reporttype in (''Q1'',''S1'',''CQ3'',''A'') order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取合并利润表数据
        %symbol，发布日期，会计截至日期，时间长度，类型，数值
        function x = get_HeBingLiRun(var_name)
            sql_str1 = ['select ticker,publishdate,enddate,fiscalPeriod,',...
                'reportType,%s from yuqerdata.nincome '];
            sql_str2 = [' where enddate>''2005-01-01'' and enddate=endDaterep ',...
                'and mergedFlag = 1 and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'and reporttype in (''Q1'',''S1'',''CQ3'',''A'') order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取合并资产负债表
        %symbol，发布日期，会计截至日期，时间长度，类型，数值
        function x = get_HeBingZiChanFuZhai(var_name)
            sql_str1 = ['select ticker,publishdate,enddate,fiscalPeriod,',...
                'reportType,%s from yuqerdata.yq_FdmtBSGet '];
            sql_str2 = [' where enddate>''2005-01-01'' and enddate=endDaterep ',...
                'and mergedFlag = 1 and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'and reporttype in (''Q1'',''S1'',''Q3'',''A'') order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取合并现金流量表
        %symbol，发布日期，会计截至日期，时间长度，类型，数值
        function x = get_HeBingXianJinLiu(var_name)
            sql_str1 = ['select symbol,publishdate,enddate,fiscalPeriod,',...
                'reportType,%s from yuqerdata.yq_FdmtCFGetAll '];
            sql_str2 = [' where enddate>''2007-01-01'' and enddate=endDaterep ',...
                'and mergedFlag = 1 and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'and reporttype in (''Q1'',''S1'',''CQ3'',''A'') order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取财务指标.盈利能力表数据
        %symbol，发布日期，会计截至日期，时间长度，类型，数值
        function x = get_YingLiNengLi(var_name)
            sql_str1 = ['select symbol,publishdate,enddate,',...
                '%s from yuqerdata.yq_FdmtIndiRtnPitGet '];
            sql_str2 = [' where enddate>''2007-01-01'' ',...
                'and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获得财务指标 衍生FdmtDerPitGet
        %symbol，发布日期，会计截至日期，时间长度，类型，数值
        function x = get_CaiWu_yansheng(var_name)
            sql_str1 = ['select symbol,publishdate,enddate,',...
                '%s from yuqerdata.yq_FdmtDerPitGet '];
            sql_str2 = [' where enddate>''2007-01-01'' ',...
                'and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取财务指标.单股
        %symbol，发布日期，会计截至日期，数值
        function x = get_CaiWu_DanGu(var_name)
            sql_str1 = ['select symbol,publishdate,enddate,',...
                '%s from yuqerdata.yq_FdmtIndiPSPitGet '];
            sql_str2 = [' where enddate>''2007-01-01'' ',...
                'and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取财务指标.运营能力
        %symbol，发布日期，会计截至日期，数值
        function x = get_CaiWu_YunYingNengLi(var_name)
            sql_str1 = ['select symbol,publishdate,enddate,',...
                '%s from yuqerdata.yq_FdmtIndiTrnovrPitGet '];
            sql_str2 = [' where enddate>''2007-01-01'' ',...
                'and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取单季度财务指标 这个表没有发布日期，需要和别的表联合查询
        %symbol，发布日期，会计截至日期，数值
        function x = get_CaiWu_DanJiDu(var_name)
            sql_str1 = ['select symbol,enddate,',...
                '%s from yuqerdata.yq_FdmtIndiQGet '];
            sql_str2 = [' where enddate>''2007-01-01'' ',...
                'and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取某个时间点申万一级行业分类
        function x = get_industry_class(t)
            str_str1 = ['select ticker,industryID1 from yuqerdata.yq_industry where ',...
                'industryVersionCD=''010303'' and intodate <= ''%s'' and ',...
                '(outDate>''%s'' or outDate is null)'];
            sql_str = sprintf(str_str1,t,t);
            x = fetchmysql(sql_str,2);
        end
        function x = get_industry_class_2(t)
            str_str1 = ['select ticker,industryID1 from yuqerdata.yq_industry_sw where ',...
                'industryVersionCD=''010303'' and intodate <= ''%s'' and ',...
                '(outDate>''%s'' or outDate is null)'];
            sql_str = sprintf(str_str1,t,t);
            x = fetchmysql(sql_str,2);
        end
        function [x_st_symbol,x_st_date0,x_st_date1] = get_st_date()
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
        end
        %获取某个时间点市值
        function x = get_market_value(t)
            %当前股票市值
            sql_str_f1 = 'select symbol,log(marketValue) from yuqerdata.yq_dayprice where tradedate = ''%s'' and marketValue is not null and marketValue !=0 order by symbol';
            x = fetchmysql(sprintf(sql_str_f1,t),2);
        end
        %获取某个时间点流通市值
        function x = get_market_value_lt(t)
            %当前股票市值
            sql_str_f1 = 'select symbol,log(negMarketValue) from yuqerdata.yq_dayprice where tradedate = ''%s'' and marketValue is not null and marketValue !=0 order by symbol';
            x = fetchmysql(sprintf(sql_str_f1,t),2);
        end
        %填充数据
        function [x1,tref1] = filling_data(tref1,tref2,x2)
            if iscell(tref1)
                tref1 = datenum(tref1);
                tref2 = datenum(tref2);
            end
            [tref2,ia] = sort(tref2);
            x2 = x2(ia);
            T = length(tref2);
            tref1 = tref1(tref1>tref2(1));
            x1 = zeros(size(tref1));
            for i = 1:T
                if ~eq(i,T)
                    sub_ind = tref1>tref2(i) & tref1<=tref2(i+1);
                else
                    sub_ind = tref1>tref2(i);
                end
                x1(sub_ind) = x2(i);
            end
        end
        %找时间点最近的前面一个时间点的数据
        function [x1,tref1] = find_near_data(tref1,tref2,x2)
            if iscell(tref1)
                tref1 = datenum(tref1);
                tref2 = datenum(tref2);
            end
            [tref2,ia] = sort(tref2);
            x2 = x2(ia);
            T = length(tref1);
            x1 = nan(size(tref1));
            for i = 1:T
                ia = find(tref2<tref1(i),1,'last');
                if ~isempty(ia)
                    x1(i) = x2(ia);
                end
            end
            nan_ind = isnan(x1);
            x1(nan_ind) = [];
            tref1(nan_ind) = [];
        end
        function y = trans_dummy(x)
            u_x = unique(x);
            T = length(u_x);
            y = zeros(length(x),T);
            for i = 1:T
                ind = eq(x,u_x(i));
                y(ind,i) = 1;
            end
        end
    end
    
    
end