%统计几何收益率曲线的参数
function [v,v_str] =cal_para_geo(y,N)

if y(end)-y(1)<0
    y =-y;
    f = -1;
else
    f = 1;
end


v_str{1} = '年化收益率';
%v(1) = ((exp(y(end)-y(1))-1)^(365/N)-1)*100;
%年收益率=[（投资内收益 / 本金）/ 投资天数] * 365 ×100%
v(1) = (y(end)-y(1))/N*365*100;
v_str{2} = '夏普比率';
temp = diff(y);
temp(isinf(temp)|isnan(temp)) = [];
v(2) = ((mean(temp)-0))/(std(temp))*sqrt(252);

v_str{3} = '年化波动率';
a = std(diff(y));
v(3)=(a*245^(1/2))*100;
if a<0
    keyboard
end


v(1:2) = v(1:2) * f;
% str = [];
% for i = 1:length(v)
%     str = [str,sprintf('%s: %0.4f',v_str{i},v(i))];
% end
% sprintf('回测曲线参数：%s \n',str)
end