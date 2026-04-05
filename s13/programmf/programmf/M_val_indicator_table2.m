%股指期货数据
%使用几何收益率计算
clear
close all
index_name_pool = {'沪深300股指期货','上证50股指期货','中证500股指期货'};
index_code = {'IF','IH','IC'};    
index_pool = {'CFFEX','CFFEX','CFFEX'};
tb_re = [];
for index_sel = 1:3
    sub_index_name = index_name_pool{index_sel};
    t0 = cell(size(index_name_pool));
    t0{1} = '2014-05-01';

    key_contracts = [index_pool{index_sel},'.',index_code{index_sel}];
    if isempty(t0{index_sel})
        sql_str = 'select tradingdate,open,close from futuredata.price_if_data where  variety0 = ''%s'' and variety = ''%s'' order by tradingdate';
        sub_sql_str = sprintf(sql_str,index_pool{index_sel},index_code{index_sel});
    else
        sql_str = 'select tradingdate,open,close from futuredata.price_if_data where  variety0 = ''%s'' and variety = ''%s'' and tradingdate >= ''%s'' order by tradingdate';
        sub_sql_str = sprintf(sql_str,index_pool{index_sel},index_code{index_sel},t0{index_sel});
    end
    index_data = fetchmysql(sub_sql_str,2);
    sql_str2 = 'select tradingdate,symbol from futuredata.future_contracts_data where variety=''%s'' and tradingdate<=''2019-03-27'' order by tradingdate';
    sub_sql_str2 = sprintf(sql_str2,key_contracts);
    index_contracts = fetchmysql(sub_sql_str2,2);
    [~,ia,ib] = intersect(index_data(:,1),index_contracts(:,1));
    index_data = index_data(ia,:);
    index_contracts = index_contracts(ib,:);
    index_contracts_num = cellfun(@(x) str2double(x(length(key_contracts)+1:end)),index_contracts(:,end));
    index_contracts_num = [0;diff(index_contracts_num)];
    index_contracts_num = ~eq(index_contracts_num,0);

    tref_str = index_data(:,1);
    tref = datenum(tref_str);
    o_c_price = cell2mat(index_data(:,2:3));
    open_price = o_c_price(:,1);
    close_price = o_c_price(:,2);
    %g_cum; g_jump g_inner %累计收益，跳价收益，日内收益
    %几何收益率
    g_cum = [0;log(close_price(2:end)./close_price(1:end-1))];
    g_jump = [0;log(open_price(2:end)./close_price(1:end-1))];
    g_inner = log(close_price./open_price);
    g_inner(1) = 0;

    g_cum(index_contracts_num) = 0;
    g_jump(index_contracts_num) = 0;
    g_inner(index_contracts_num) = 0;


    g_info = {'累计对数收益','累计日内收益','累计跳价收益'};
    g_data_geo = [g_cum,g_inner,g_jump];
    %算术收益率
    g_cum_m = [0;close_price(2:end)./close_price(1:end-1)-1];
    g_jump_m = [0;open_price(2:end)./close_price(1:end-1)-1];
    g_inner_m = close_price./open_price-1;
    g_inner_m(1) = 0;
    
    g_cum_m(index_contracts_num) = 0;
    g_jump_m(index_contracts_num) = 0;
    g_inner_m(index_contracts_num) = 0;
    
    g_data_math = [g_cum_m,g_inner_m,g_jump_m];

    obj = zeros(3,1);
    colors = [0.64,0.78,0.18;0.93,0.69,0.13;ones(1,3)*0.65];
    obj(1)= plot(tref,cumsum(g_cum),'-','linewidth',2,'color','r');
    hold on
    obj(2)= plot(tref,cumsum(g_inner),'-','linewidth',2,'color',colors(2,:));
    obj(3)= plot(tref,cumsum(g_jump),'-','linewidth',2,'color',colors(3,:));
    %obj(1)= plot(tref,cumsum([g_cum,g_inner,g_jump]),'linewidth',2);
    plot(tref([1,end]),[0,0],'k-','LineWidth',2);

    set(gca,'XTickLabelRotation',90);
    set(gca,'XTick',tref(floor(linspace(1,length(tref),40))),'xlim',tref([1,end]));
    datetick('x','yyyymmdd','keepticks');
    set(gca,'fontsize',12);

    box off
    set(gca,'linewidth',1.5);
    legend(obj,g_info,'Location','northwest',...
        'NumColumns',length(obj),'location','best')
    legend('boxoff')

    val = get(gca,'YTick');
    labels = num2str(val'*100,'%5.1f%%');
    % Adjust labels on plot
    set(gca,'yticklabel',labels)
    title(sub_index_name)
    setpixelposition(gcf,[416,397,961,420]);

    %[v,v_str] = curve_static(-cumprod(1+g_jump_m))
    V = zeros(3,3);
    for i = 1:3
        %sprintf('%s',g_info{i})
        %时间限制
        ind = tref<=datenum(2019,3,29);
        %ind = tref>=datenum(2015,04,16)&tref<=datenum(2019,3,27);
        sub_tref = tref(ind);
        sub_g_data_geo = g_data_geo(ind,:);
        [V(:,i),v_str] = cal_para_geo(cumsum(sub_g_data_geo(:,i)),sub_tref(end)-sub_tref(1)+1);
        %[V(:,i),v_str] = cal_para_geo(cumsum(sub_g_data_geo(:,i)),length(sub_tref));
        %sub_g_data_math = g_data_math(ind,:);
        %temp = cal_para_geo(cumsum(sub_g_data_math(:,i)),sub_tref(end)-sub_tref(1)+1);
        %V(1,i) = temp(1);
        
    end
    datestr(sub_tref([1,end]),'yyyy-mm-dd')
    f_str = '%s结果\n\t\t\t 年化收益率\t\t Sharp值 \t\t年化波动率\n';
    for i = 1:3
        f_str = [f_str,g_info{i},'\t%0.3f \t\t %0.3f \t\t %0.3f \n'];
    end

    sprintf(f_str,sub_index_name,V)
    temp_tb_re = [g_info',g_info',num2cell(V')];
    temp_tb_re(:,1) = {sub_index_name};
    temp_tb_re = temp_tb_re(3:-1:1,[1,2,2+[1,3,2]]);
    tb_re = [tb_re;temp_tb_re];
    index_sel
end
title('');

%统计几何收益率曲线的参数
function [v,v_str] =cal_para_geo(y,N)

if y(end)-y(1)<0
    y =-y;
    f = -1;
else
    f = 1;
end


v_str{1} = '年化收益率';
%v(1) = ((exp(y(end)-y(1))-1)^(365/N)-1)*100;
%年收益率=[（投资内收益 / 本金）/ 投资天数] * 365 ×100%
v(1) = (y(end)-y(1))/N*365*100;
v_str{2} = '夏普比率';
temp = diff(y);
temp(isinf(temp)|isnan(temp)) = [];
v(2) = ((mean(temp)-0))/(std(temp))*sqrt(252);

v_str{3} = '年化波动率';
a = std(diff(y));
v(3)=(a*245^(1/2))*100;
if a<0
    keyboard
end


v(1:2) = v(1:2) * f;
% str = [];
% for i = 1:length(v)
%     str = [str,sprintf('%s: %0.4f',v_str{i},v(i))];
% end
% sprintf('回测曲线参数：%s \n',str)
end
