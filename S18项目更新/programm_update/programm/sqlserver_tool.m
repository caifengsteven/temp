classdef sqlserver_tool<handle
    properties
        conna
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        function obj=sqlserver_tool(datasource,username,password,url)
            
            if eq(nargin,0)
                datasource = 'bjjszx';
                username = 'jszx';
                password = 'jszx';
                url = sprintf('jdbc:sqlserver://10.15.32.27:1433;database=%s',datasource);
            end
                        
            driver = 'com.microsoft.sqlserver.jdbc.SQLServerDriver';
            javaaddpath('D:\Program Files\MATLAB\R2018a\java\jar\mssql-jdbc-6.4.0.jre7.jar')
            obj.conna = database(datasource,username,password,driver,url);

            %obj.conna =  database('nir2008','jszx','jszx');
        end
        function delete(obj)
            close(obj.conna);
            sprintf('sqlserver_tool close')
        end
        function x = get_sql_data(obj,sqlquery_str,format_sel)
            if nargin<3
                format_sel = 1;
            end
            format_str = containers.Map([1,2],{'numeric','cellarray'});
            
            setdbprefs('DataReturnFormat',format_str(format_sel));
            x = fetch(obj.conna,sqlquery_str,1000);
        end
        function do_sql_order(obj,sqlquery_str)
            exec(obj.conna,sqlquery_str)
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods(Static)
        function sql_str = create_table(tbname,varname,typename,pkname)
            join_str = cellfun(@(x,y) [x,' ',y],varname,typename,'UniformOutput',false);
            join_str = strjoin(join_str,',');
            if nargin>3
                sql_str = sprintf('create table %s(%s,primary key(%s))',tbname,join_str,strjoin(pkname,','));
            else
                sql_str = sprintf('create table %s(%s)',tbname,join_str);
            end
            
        end
    end
end