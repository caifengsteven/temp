classdef gta_web_tool<handle
    
    methods(Static)
        function x = get_pub_date(t)
            sql_str = 'select Stkcd,Annodt from gta_web.IAR_Rept where Accper = ''%s''';
            x = fetchmysql(sprintf(sql_str,t),2);
            
            
        end
    end
    
    
    
    
end