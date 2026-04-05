v = input('enter vector of exchange rates')
j = 1:length(v);
x = [  ];
z = [  ];
b=[x;z];
profits = [  ];  %set counter i = start:stepsize:finish% 
for i=2:1:length(v)-1
    %finding peaks and valleys% 
    if v(i) <= v(i+1)&& v(i-1) >= v(i)||v(i) >= v(i+1)&& v(i-1)  <=  v(i) 
        %putting the peaks and valleys in a new matrix b%
        x = [x v(i)];
        z = [z j(i)];
        x=x([1,diff(x)]~=0);         
        z=z([1,diff(x)]~=0);   
        b=[x;z];    
    end
end

%set new counter n = start:stepsize:finish%
for n=3:1:length(x)-4
    %check for Elliott wave pattern   
    if x(n)< x(n-1) && x(n)>=x(n-2)&& x(n+1)>x(n-1)&& x(n+2)>x (n-1)
        if x(n+1)-x(n)>= x(n-1) - x(n-2) ||  x(n+1)-x(n)>= x    (n+3) - x(n+2)
            %check for time intervals%
            if z(n)-z(n-1)<=(0.382*(z(n-1)-z(n-2))) && z(n+1)-z(n)<=(1.618*(z(n-1)-z(n-2))) && z(n+2)-z(n+1)<=(0.382*(z(n+1)-z(n))) && z(n+3)-z(n+2)<=(1.618*(z(n+1)-z(n)))
                %check for Fibo retracement 1 hits, for other Fibo     levels replace 0.236 with other fibo ratios%
                if x(n)<= (x(n-1)-((x(n-1)-x(n-2))*0.236))*1.001 && x(n)>= (x(n-1)-((x(n-1)-x(n-2))*0.236))*0.999
                    %profits after exit signal and entry is price      exiting top of Fibonacci range, for other Fibo      levels replace 0.236 with other fibo ratios%%
                    disp(((0.999*x(n+3))-((x(n-1)-((x(n-1)-x(n-2))*0.236))*1.001))/((x(n-1)-((x(n-1)-x(n-2))*0.236)) *1.001))
                end
            end
        end
    end
end