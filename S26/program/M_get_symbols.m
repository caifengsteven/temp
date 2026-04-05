%clear
%del M_rule2
method_list = {'M_rule1';'M_rule2';'M_rule3';'M_rule4';'M_rule5_update';...
    'M_rule6';'M_rule7';'M_rule8';'M_rule9';'M_rule10';'M_rule11';'M_rule12';'M_rule13'};
re = cell(13,1);
parfor i = 1:13
    re{i} = do_method(method_list{i});
end
function re_rule1 = do_method(method_name)
eval([method_name,';'])
close all
end