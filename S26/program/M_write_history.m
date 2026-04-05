clear
%dN = 'S26';
tn = 'S26.S26_result';
   
var_info = {'rule_name','tradingdate','symbol'};

load re1 
for i = 1:length(re)
    if ~isempty(re{i})
        sub_re = re{i};
        a=cellfun(@(x,y) [x,y],sub_re(:,1),sub_re(:,2),'UniformOutput',false);
        [~,ia] = unique(a,'stable');
        sub_re = sub_re(ia,:);
        
        sub_re = sub_re(:,[1,1:end]);
        sub_re(:,1) = {i};
        datainsert_adair(tn,var_info,sub_re)
    end
    
    
end