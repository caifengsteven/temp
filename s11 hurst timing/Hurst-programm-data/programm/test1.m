mod_ef = 0;

y_20 = MA(y0,20);
y_60 = MA(y0,60);
y_200 = MA(y0,120);
ind = tref>=datenum(1997,12,1) & tref<=datenum(2019,11,30);
sub_tref = tref(ind);
sub_r1 = r1(ind);
sub_close_price = close_price(ind);



y = y_200(ind);
hurst_exp = hurst_exp0(ind);

T = length(y);
ind = ones(T,1);
wid_year = 240;
ind2 = zeros(size(ind));
for i = wid_year:T
    if (y(i-1)>hurst_exp(i) &&y(i)<hurst_exp(i)) || (y(i-1)>0.5 &&y(i)<0.5)
        temp = cumprod(1+sub_r1(i-wid_year+1:i));
        if temp(end)>1.03
            if eq(mod_ef,0)
                ind(i) = 0;
            else
                ind(i) = -1;
            end
        else
            ind(i) = 1;
        end
        ind2(i) = 1;
    else
        ind(i) = ind(i-1);
    end
end
ind2 = find(ind2);

yyaxis left
plot(sub_tref,y,'LineWidth',2)
hold on
plot(sub_tref,hurst_exp,'r-','LineWidth',2)
plot(sub_tref,ind,'-','LineWidth',2);
for i = 1:length(ind2)
    line(sub_tref([ind2(i),ind2(i)]),[mod_ef,1],'Color','k','LineWidth',3);
end
yyaxis right
plot(sub_tref,sub_close_price);
my_time_label(gca,sub_tref)
box off


%back_test
r_c = zeros(size(ind));
for i = 2:length(ind)
    r_c(i) = sub_r1(i)*ind(i-1);    
end

figure;
plot(sub_tref,cumprod(1+r_c),'LineWidth',2);
hold on
plot(sub_tref,sub_close_price./sub_close_price(1),'LineWidth',2);
my_time_label(gca,sub_tref)