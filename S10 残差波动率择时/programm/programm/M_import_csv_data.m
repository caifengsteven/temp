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
    x1.info = input('������ָ������(�ǵü�����):  ');
    fn = input('�����뱣���ļ�����(�ǵü�����)�� ');
    save(fn,'x1');
end
