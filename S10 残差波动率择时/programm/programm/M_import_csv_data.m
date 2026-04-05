clear
[file_name,path_name] = uigetfile('*.csv');

if ~isequal(file_name,0)
    [~,~,x] = xlsread(fullfile(path_name,file_name));
    x = x(2:end,:);
    ind = cellfun(@isnumeric,x(:,2));
    x = x(ind,:);
    x1=[];    
    x1.open=cell2mat(x(:,2))';
    x1.close=cell2mat(x(:,5))';
    x1.eob = x(:,1)';
    x1.info = input('请输入指数名称(记得加引号):  ');
    fn = input('请输入保存文件名称(记得加引号)： ');
    save(fn,'x1');
end
