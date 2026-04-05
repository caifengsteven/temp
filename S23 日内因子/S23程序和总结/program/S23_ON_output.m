%S23结果输出
%中性化
%function y=S23_ON_output()
remain_num = 100;
T_pool = 3;
[y1,tc1,t_str]=S23_ON_output_fenbi();
[y2,tc2]=S23_ON_output_pareto();
[y3,tc3]=S23_ON_output_volumebig();
[y4,tc4]=S23_ON_output_volumeratio();
y=[y1,y2(:,2:end),y3(:,2:end),y4(:,2:end)];
t_c = [{'股票池'},tc1,tc2,tc3,tc4];
t_r = [1:remain_num,-1:-1:-remain_num]';
t_r = repmat(t_r,T_pool,1);
t_r = cellfun(@num2str,num2cell(t_r),'UniformOutput',false);

title_str = sprintf('%sS23选股结果',t_str);
gui_result(y,title_str,t_c,t_r)

y2 = [[{' '};t_r],[t_c;y]];
y2 = cell2table(y2);
writetable(y2,sprintf('%s.csv',title_str));
%end