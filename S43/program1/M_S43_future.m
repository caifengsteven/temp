clear

key_str = 'S43国内期货验证';
write_sel = true;
if write_sel
    pn_write = fullfile(pwd,'计算结果');
    if ~exist(pn_write,'dir')
        mkdir(pn_write)
    end
    obj_wd = wordcom(fullfile(pn_write,sprintf('%s.doc',key_str)));
    xls_fn = fullfile(pn_write,sprintf('%s.xlsx',key_str));
end

[~,~,X] = xlsread('S43_future_re.csv');
tref = X(2:end,end-1);
symbols = X(2:end,end);
var_name = X(1,2:end-2);
X = cell2mat(X(2:end,2:end-2));

u_symbols = unique(symbols);
T_u_symbols = length(u_symbols);
sta_re1 = cell(T_u_symbols,1);
for symbol_sel = 1:T_u_symbols
    sub_symbol = u_symbols{symbol_sel};
    sub_ind = strcmp(symbols,u_symbols(symbol_sel));
    x=X(sub_ind,:);
    t = tref(sub_ind);
    t = cellstr(datestr(datenum(t),'yyyy-mm-dd'));
    t_str = cellfun(@(x) [x(1:4),x(6:7),x(9:10)],t,'UniformOutput',false);
    T = length(t_str);
    T_type = size(x,2);
    sta_re2 = cell(T_type,1);
    for i = 1:T_type
        r_c = cumprod(x(:,i));
        title_str = sprintf('%s-%s',sub_symbol,var_name{i});
        title_str(strfind(title_str,'_')) = '-';
        
        if write_sel
            if eq(i,4)
                h=figure;        
                plot(r_c,'-','LineWidth',2);
                set(gca,'xlim',[0,T]);
                set(gca,'XTick',floor(linspace(1,T,15)));
                set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
                set(gca,'XTickLabelRotation',90)    
                setpixelposition(h,[223,365,1345,420]);
                box off
                title(title_str)
                obj_wd.pasteFigure(h,title_str);
            end
        end
        
        [v0,v_str0] = curve_static(r_c,[],false);
        [v,v_str] = ad_trans_sta_info(v0,v_str0); 
        result2 = [v_str;v]';
        result = [{sub_symbol,title_str};result2];
        if ~eq(i,1)
            result = result(:,2);
        end
        sta_re2{i} = result;
        sprintf('%s %d-%d',key_str,i,T_type)
    end
    y = [sta_re2{:}];
    sta_re1{symbol_sel} = y;
end
y = [sta_re1{:}];
y = y';

if write_sel
    obj_wd.CloseWord();
    xlswrite(xls_fn,y);
end



