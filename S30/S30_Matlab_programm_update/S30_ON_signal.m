function [y,t_c,t_str]=S30_ON_signal()
%全部为正向因子（因子越大越好）
remain_num = 100;
t_str = fetchmysql('select tradingdate from S30.F_month_final_adj order by tradingdate desc limit 1',2);
t_str = t_str{end};%交易日
sql_str = 'select symbol,%d*%s from %s where tradingdate=''%s''';%因子越小，越好

tN_all = 'S30.F_month_final_adj';
paretol_pool = {'f_val'};

tN_dir = [1];%因子方向
%tN_info = {'买卖单Spread因子'};
tN_info = {'HP滤波后价值因子'};
%两个方法，一共8个因子
T = length(tN_info);
y = cell(T,1);

symbol_pool_all = {   [],    '000905','000300'};
symbol_pool_info = {'全市场','中证500','沪深300'};
T_pool = length(symbol_pool_all);
t_c = cell(1,T);
parfor i = 1:T
    t_c{i} = tN_info{i};
    x0 = fetchmysql(sprintf(sql_str,tN_dir(i),paretol_pool{i},tN_all,t_str),2); 
    
    sub_y = cell(3,1);
    for j = 1:T_pool
        sub_index = symbol_pool_all{j};
        sub_index_name = symbol_pool_info{j};
        if sub_index>0
            sub_symbol_pool = get_index_pool(sub_index,t_str);
            [~,ia] = intersect(x0(:,1),sub_symbol_pool);
            x = x0(ia,:);
        else
            x = x0;
        end
        x = S32_nero_test(x,t_str);
        
        [~,ia] = sort(cell2mat(x(:,2)),'descend');
        x = x(ia,:);
        if eq(i,1)
            sub_re = x([1:100,end-100+1:end],[1,1]);
            sub_re(:,1) = {sub_index_name};
        else
            sub_re = x([1:remain_num,end-remain_num+1:end],1);
        end
        sub_y{j} = sub_re';
    end
    y{i} = [sub_y{:}]';
end
y = [y{:}];

end
function x1 =S32_nero_test(x1,t_str)

    window1 = 180;%33原本是60天，为了和S36统一，使用了180
    %上市时间
    sql_str4 = ['select ticker,listDate from yuqerdata.equget where listStatusCd !=''UN''',...
                    'and listDate is not null'];
    %sql_str6 = ['select symbol,log(f_mv),f_reverse,f_std,f_change from S33.factor_zxh ',...
    %    'where tradingdate = ''%s'''];
    sql_str6 = 'select symbol,log(negMarketValue) from yuqerdata.yq_dayprice where tradeDate = ''%s''';
    symbol_info = fetchmysql(sql_str4,2);
    symbol_listdate = datenum(symbol_info(:,2));
    sql_str3 = 'select ticker from   yuqerdata.st_info where tradedate =''%s''';
    %st
    st = fetchmysql(sprintf(sql_str3,t_str),2);
    st = cellfun(@str2double,st,'UniformOutput',false);
    st = cellfun(@(x) sprintf('%0.6d',x),st,'UniformOutput',false);
    [~,ia] = intersect(x1(:,1),st);
    x1(ia,:) = [];
    %新股
    ind = datenum(t_str)-symbol_listdate>window1;
    [~,ia] = intersect(x1(:,1),symbol_info(ind,1));
    x1 = x1(ia,:);
    
    %中性化
    %sub_f_ner = get_ner_dataS36(t_str);
    sub_f_ner = fetchmysql(sprintf(sql_str6,t_str),2);
    %industry code
    x_indus = yq_methods.get_industry_class_2(t_str);
    inds = suscc_intersect({x1(:,1),sub_f_ner(:,1),x_indus(:,1)});
    x1 = x1(inds(:,1),:);
    sub_f_ner=sub_f_ner(inds(:,2),:);
    x_indus = x_indus(inds(:,3),:);

    x1_v = cell2mat(x1(:,2));
    sub_f_ner_v = cell2mat(sub_f_ner(:,2:end));
    x_indus_v = cell2mat(x_indus(:,2));        
    dummy_f = yq_methods.trans_dummy(x_indus_v(:,end));        
    %regress
    %[~,~,x1_v] = regress(x1_v,[ones(size(x1_v)),sub_f_ner_v,dummy_f]);
    warning off
    [~,~,x1_v] = regress(x1_v,[ones(size(x1_v)),sub_f_ner_v(:,end),dummy_f]); 
    warning on
    x1 = [x1(:,1),num2cell(x1_v)];
end
% function sub_f_ner = get_ner_dataS36(t)
%     sql_str6 = ['select f_type,symbol,f_val from S36.factor_zxhS36 where f_type<=3 and ',...
%         'tradingdate =''%s'''];
%     sql_str6a = 'select symbol,log(negMarketValue) from yuqerdata.yq_dayprice where tradeDate = ''%s''';
%     sub_f1 = fetchmysql(sprintf(sql_str6a,t),2);
%     sub_zxh = fetchmysql(sprintf(sql_str6,t),2);
%     sub_f_ner = sub_f1(:,[1,end]);
%     for j = 1:3
%         temp_ind = cellfun(@(x) eq(x,j), sub_zxh(:,1));
%         sub_sub_zxh = sub_zxh(temp_ind,2:end);
%         [~,ia,ib] = intersect(sub_f_ner(:,1),sub_sub_zxh(:,1));
%         sub_f_ner = cat(2,sub_f_ner(ia,:),sub_sub_zxh(ib,end));
%     end
% end

function sub_symbol_pool = get_index_pool(index_pool,t_str)
    sub_t = fetchmysql(sprintf(['select tradingdate from yuqerdata.IdxCloseWeightGet ',...
        'where tradingdate < ''%s'' and ticker = ''%s'' order by tradingdate desc limit 1'],...
                    t_str,index_pool),2);
    if isempty(sub_t)
        sub_t = fetchmysql(sprintf(['select tradingdate from yuqerdata.IdxCloseWeightGet ',...
            'where tradingdate >= ''%s'' and ticker = ''%s''  order by tradingdate limit 1'],...
        t_str,index_pool),2);
    end
    sub_symbol_pool = fetchmysql(sprintf(['select symbol from yuqerdata.IdxCloseWeightGet ',...
        'where tradingdate = ''%s'' and ticker = ''%s'''],sub_t{1},index_pool),2);
end