import json
data_list={'mysql_para':{'user_name':'root','pass_wd':'352471Cf','port':3306,'host':'localhost'},
'sql3_usstock':r'L:\Dropbox\Dropbox\Feng Cai\sqlit',
'sql3_forex':r'L:\Dropbox\Dropbox\Feng Cai\poly_forex',
'yuqerdata_dir':r"K:\yq_data"}
with open('para.json','w',encoding='utf-8') as f:
    json.dump(data_list,f,ensure_ascii=False)
#从文件读取数据
with open('para.json','r',encoding='utf-8') as f:
    para=json.load(f)
    print(para)