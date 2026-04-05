%change data type
%{
x = fetchmysql('desc gta_web.stk_fin_incomettm',2);
T = size(x,1);
sql_str = 'alter table gta_web.stk_fin_incomettm modify %s float';
for i = 5:T
    exemysql(sprintf(sql_str,x{i,1}));
end

x = fetchmysql('desc gta_web.IAR_Rept',2);
T = size(x,1);
sql_str = 'alter table gta_web.IAR_Rept modify %s float';
for i = 9:T
    exemysql(sprintf(sql_str,x{i,1}));
end
%}

x = fetchmysql('desc gta_web.STK_FIN_CashFlowTTM',2);
T = size(x,1);
sql_str = 'alter table gta_web.STK_FIN_CashFlowTTM modify %s float';
for i = 5:T
    exemysql(sprintf(sql_str,x{i,1}));
end