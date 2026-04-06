%补充程序结果
%延迟执行信号
%forex_day csi 5day
clear
close all
key_str = sprintf('S43 双底策略指数成分股组合%s',datestr(now,'yyyymmdd'));
write_sel = false;

index0 = 'csi';
para = containers.Map({'US','HK','forex-day','csi'},[15,15,5,5]);

if write_sel
    pn_write = fullfile(pwd,'计算结果');
    if ~exist(pn_write,'dir')
        mkdir(pn_write)
    end
    obj_wd = wordcom(fullfile(pn_write,sprintf('%s.doc',key_str)));
    xls_fn = fullfile(pn_write,sprintf('%s.xlsx',key_str));
    index_com_fn = fullfile(pn_write,sprintf('%s成分股.mat',key_str));
end

sql_str = sprintf('select ticker,tradeDate,r_%d-1  from S37.s43_addre where index0="%s" order by tradeDate',para(index0),index0);
x = fetchmysql(sql_str,2);
x = cell2table(x,'VariableNames',{'s','t','v'});
X = unstack(x,'v','t');
% ind = unstack(x,'v','s');
% ind.Properties.RowNames = ind.t;
% ind.t = [];
% x = table2array(ind);
% x(isnan(x)) = 0;
% x(x~=0) = 1;
% signal = array2table(x,'VariableNames',ind.Properties.VariableNames,'RowNames',ind.Properties.RowNames);
% writetable(ind,sprintf('signal_%s.csv',index0),'WriteRowNames',true)


tref = X.Properties.VariableNames;
tref = cellfun(@(x) x(2:end),tref,'UniformOutput',false);
tref = tref(2:end);
X = table2cell(X); 
symbols = X(:,1);
X = cell2mat(X(:,2:end))';
X(isnan(X)) = 0;

ind0 = eq(sum(abs(X)),0);
X(:,ind0) = [];
symbols(ind0) = [];

t_str = cellfun(@(x) [x(1:4),x(6:7),x(9:10)],tref,'UniformOutput',false);
T = length(t_str);

symbol_pool_all = { '000905','000300'};
symbol_pool_info = {'中证500','沪深300'};
T_index_pool = length(symbol_pool_all);
sta_re2 = cell(T_index_pool,1);
symbols_comp = cell(T_index_pool,1);
for i = 1:T_index_pool
    sub_index = symbol_pool_all{i};
    title_str = symbol_pool_info{i};
    title_str(strfind(title_str,'_')) = '-';

    sub_symbols = yq_methods.get_index_pool(sub_index,datestr(now,'yyyy-mm-dd'));
    symbols_comp{i} = sub_symbols;
    [~,ia] = intersect(symbols,sub_symbols);
    sub_r = X(:,ia);
    sub_symbols = cellfun(@(x) ['x_',x], symbols(ia),'UniformOutput',false);
    
    y = array2table(sub_r,'RowNames',tref','VariableNames',sub_symbols');
    
    writetable(y,sprintf('signal_%s_%s.csv',index0,sub_index),'WriteRowNames',true)
end