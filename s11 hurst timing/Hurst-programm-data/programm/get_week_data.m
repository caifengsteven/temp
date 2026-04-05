%获取周数据
function [tref_w,p_open_w,p_close_w] = get_week_data(tref,p_open,p_close)
week_num = weeknum(tref);
ind = find(diff(week_num));
ind = [0;ind;length(tref)];

ind = [ind(1:end-1)+1,ind(2:end)];
p_open_w = p_open(ind(:,1));
p_close_w = p_close(ind(:,2));
tref_w = tref(ind(:,2));
end