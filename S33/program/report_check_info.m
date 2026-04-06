function report_check_info(OK,info)
f_str = '核查：%s，结果：%s ';
if OK
    sprintf(f_str,info,'通过')
else
    sprintf(f_str,info,'不通过!!!!!,截图，或保留记录')
end