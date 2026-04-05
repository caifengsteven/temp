function [x,OK] = fetchsqlserver(sqlquery,format,dN,usr,pass)
if nargin < 2
    format = 1;
end
if nargin < 3
    dN = 'nirserver1';
end
if nargin < 4
    usr = 'sa';
end
if nargin < 5
    pass = '352471Cf';
end
%conna = database(dN,usr,pass);
obj=sqlserver_tool(dN,usr,pass,'jdbc:sqlserver://localhost:1433;database=nirserver1');%need update
conna=obj.conna;
if eq(format,1)
    setdbprefs('DataReturnFormat','numeric');
else
    setdbprefs('DataReturnFormat','cellarray');
end
try
    if iscell(sqlquery)
        exec(conna,sqlquery{1});
        x = fetch(conna,sqlquery{2});
    else
        x = fetch(conna,sqlquery);
    end
    OK = true;
catch
    close(conna);
    OK = false;
    x = [];
end
close(conna);
