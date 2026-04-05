%M_update_table
%设定更新时间
f = cell(8,1);
f{1} = 'update S29.factor_wind_com_ttm set pub_date= CONCAT(cast(year(pub_date) as char),''-05-01'') where month(pub_date) = 3';
f{2} = 'update S29.factor_wind_com_ttm set pub_date= CONCAT(cast(year(pub_date) as char),''-08-30'') where month(pub_date) = 6';
f{3} = 'update S29.factor_wind_com_ttm set pub_date= CONCAT(cast(year(pub_date) as char),''-10-31'') where month(pub_date) = 9';
f{4} = 'update S29.factor_wind_com_ttm set pub_date= CONCAT(cast(year(pub_date)+1 as char),''-04-30'') where month(pub_date) = 12';

f{5} = 'update S29.factor_wind_com set pub_date= CONCAT(cast(year(pub_date) as char),''-05-01'') where month(pub_date) = 3';
f{6} = 'update S29.factor_wind_com set pub_date= CONCAT(cast(year(pub_date) as char),''-08-30'') where month(pub_date) = 6';
f{7} = 'update S29.factor_wind_com set pub_date= CONCAT(cast(year(pub_date) as char),''-10-31'') where month(pub_date) = 9';
f{8} = 'update S29.factor_wind_com set pub_date= CONCAT(cast(year(pub_date)+1 as char),''-04-30'') where month(pub_date) = 12';

parfor i = 1:8
    exemysql(f{i});
end