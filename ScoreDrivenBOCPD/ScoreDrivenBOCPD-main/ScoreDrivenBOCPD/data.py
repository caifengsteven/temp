import numpy as np
import pandas as pd

class Priors:
    
    def __init__(self,bern_p,mean0,var0):
        
        self.bern_p = bern_p
        self.mean0  = mean0
        self.var0   = var0
        
    
    def bernoulli(self):
        
        if np.random.random() < self.bern_p:
            
            return True
        
        
    def gaussian(self):

        return np.random.normal(self.mean0, self.var0)


class Data(Priors):
    
    def __init__(self, bern_p=1/100,mean0=1,var0=3, file_name = "_",data_type =  "real"):
        super().__init__(bern_p,mean0,var0)
        
        if data_type == "real" and file_name == "_":
            raise ValueError("When the data_type is \"real\" you should also provide the file_name")
        
        self.data_type = data_type
        self.file_name = file_name
        
    
    def update_data(self):
        
        if self.data_type == "sim":
            
            return self.__sim_data()
        
        if self.data_type == "real":
            
            return self.__real_data()
        
    
    def __sim_data(self,var = 1, rho = 0.3, T = 500):
        data = []
        cps = []
        mean = super().gaussian()
        data.append(np.random.normal(mean,var))
        
        for t in range(1,T): 

            if super().bernoulli():

                mean = super().gaussian()
                
                cps.append(t)
                data.append(np.random.normal(mean,var))
                
                continue
            
            data.append(np.random.normal(mean+rho*(data[t-1]-mean),var*(1-rho**2)))
        
        return data,cps
    
    
    def __real_data(self):

        data_df = pd.read_csv(self.file_name,delimiter =',',header = 0)
        data = list(data_df[data_df.columns[0]])

        return data
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            