function conna = mysql_conn()

conna = database('futuredata','root','liudehua','com.mysql.jdbc.Driver',...
    'jdbc:mysql://localhost:3306/futuredata?useSSL=false&');