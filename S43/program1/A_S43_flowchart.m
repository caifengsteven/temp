%{
策略执行分两步
1 python框架运行得到曲线
2 matlab可视化

python部分有
1 A股指数成分股 已经对接到S37，这里不再赘述
2 国内期货 M_S34_W_future.py  M_S43_W_future_paraTest.py是调参程序
3 外汇 M_S43_W_exchange.py  M_S43_W_exchange_paraTest.py是调参程序
4 道指成分股 M_S43_W_dowjones_dayly.py
5 美股 M_S43_W_american_stock.py

%}
%结果验证部分
title_str = 'S43 美股回测';
order_str = 'M_S43_american_stock';
run_program_adair(order_str,title_str);

title_str = 'S43 道指成分股回测';
order_str = 'M_S43_dowjones';
run_program_adair(order_str,title_str);

title_str = 'S43 外汇回测';
order_str = 'M_S43_exchange';
run_program_adair(order_str,title_str);

title_str = 'S43 期货回测';
order_str = 'M_S43_future';
run_program_adair(order_str,title_str);