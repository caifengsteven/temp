%example [x,y]=fmincon('fun1',rand(3,1),[],[],[],[],zeros(3,1),[],'fun2')
%{
max(fw)
st.
sl<=S(w-wb)<=sh
sl+Swb<=Sw<=sh+Swb   No need

hl<=H(w-wb)<hh 
hl+Hwb<=Hw<=hh+Hwb

wl<=w-wb<=wh
wl+wb<=w<=wh+wb

0<=w<=l
sum(w) = 1
%}
clear
f = (1:300)';

mv = [ones(200,1);ones(100,1)*2];

w0 = ones(300,1)*1/300/2;

indus_dummy = zeros(300,1);

indus_dummy(1:10,1) = 1;

indus_dummy(end-10:end,2) = 1;

fun = @(x) -f'*x;

x0 = ones(300,1);
A = ones(1,300);
b = 1;

Aeq = ones(1,300);
beq = 1;

lb = zeros(300,1);
ub = ones(300,1)*1/(300/5);

options = optimoptions('fmincon');
options.MaxFunctionEvaluations = 30000;

x = fmincon(fun,x0,[],[],Aeq,beq,lb,ub,[],options);



