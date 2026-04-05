%{
def rsquare(x,y):
    ##计算拟合优度R2
    
    meanofx=x.mean()
    meanofy=y.mean()
    difofxx=x - meanofx
    difofxy=x - meanofy
    difofyy=y - meanofy 

    beta = sum(difofxx * difofyy) / sum(difofxx ** 2)
    r2 = beta**2 * sum(difofxx**2) / sum(difofyy**2)
    return r2
%}
function r2 = get_rsqure(x,y)
mx = mean(x);
my = mean(y);

x_mx = x-mx;
%x_my = x-my;
y_my = y-my;

beta = sum(x_mx.*y_my)/sum(x_mx.*x_mx);
r2 = beta*beta * sum(x_mx.*x_mx)/sum(y_my.*y_my);