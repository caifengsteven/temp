
function[v,v_str] = cal_para_math(y,N)
% y = cumprod(1+rand(1000,1)/1000);
%(AC3277/100)^(244/COUNT(AC120:AC3277))-1
%1年化收益率
v_str{1} = '年化收益率';
v(1) = (y(end)/y(1))^(365/N)-1;

v_str{2} = '夏普比率';
%v(9) = (((mean(y(2:end)./y(1:end-1)-1)-3/100/252))/std(y(2:end)./y(1:end-1)-1))*sqrt(252);
temp = y(2:end)./y(1:end-1)-1;
temp(isinf(temp)|isnan(temp)) = [];
v(2) = ((mean(temp)-0))/(std(temp))*sqrt(252);

str = [];
for i = 1:length(v)
    str = [str,sprintf('%s: %0.4f\n',v_str{i},v(i))];
end
sprintf('回测曲线参数：%s \n',str)

end