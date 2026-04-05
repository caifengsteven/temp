%Rule12：YOY3(存货周转率)<0 AND YOY3(毛利率)>0
%存货周转率 invenTurnover FdmtIndiTrnovrPitGet  财务指标 运营能力 （或者单日因子）
%毛利率  grossMargin DataAPI.FdmtMoStdItemGet  主营业务构成
%grossMARgin 财务指标盈利能力
clear

sql_str_f1 = ['select symbol,publishdate,enddate,invenTurnover ',...
    'from yuqerdata.yq_FdmtIndiTrnovrPitGet order by endDate,publishdate'];
sql_str_f2 = ['select ticker,endDate,grossMargin from yuqerdata.yq_fdmtmainopernget_update where ',...
    'itemName = ''合计''  and classifCD=  1 order by endDate,publishdate'];
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
    sprintf('%d-%d',i,T)
end
re_rule1 = [re_rule1{:}]';
ia = datenum(re_rule1(:,1));
ia = ia>=datenum(2010,1,1) & ia<=datenum(2017,1,1);
re_rule1 = re_rule1(ia,:);


rule_validation_update(re_rule1)
setpixelposition(gcf,[430,368,1008,420]);
