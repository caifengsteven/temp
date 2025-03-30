"""
The GaussianUnknownMean class updates the posterior parameters and the UPM for the 
three different models; BOCPD, MBO and MBOC as they 
are described in the paper:

[1] "Online Learning of Order Flow and Market Impact with 
Bayesian Change-Point Detection Methods" 
by Tsaknaki,Lillo,Mazzarisi, 2023

see: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4500960 
"""
import numpy as np
from   scipy.stats import norm
import ScoreDrivenBOCPD.sd_ar as sd


class GaussianModel:
    
    def __init__(self, mean0, var0, varx, init_theta, q, init_cor = 0):
        
        varx = 1e-8 + varx
        var0 = 1e-8 + var0
        
        if varx < 0:
            raise ValueError("The variance should be zero or positive")
            
        if var0 < 0:
            raise ValueError("The prior value var0 should be zero or positive")

        if init_cor > 0.9 or init_cor<-0.9:
            raise ValueError("The correlation should be a value in the interval [-0.9,0.9]")
            
        if q!=0 and q!=1/2 and q!=1:
            raise ValueError("The variable q should be one of 0, 1/2, 1")
            
        if (q == 1/2 or q == 1) and init_cor == 0:
            raise ValueError("For this model the correlation should live in the set [-0.9,0.9]\{0}")
        
        if q == 0:
            init_cor = 0
           
        self.parameters        = init_theta
        self.init_var          = varx
        self.init_cor          = init_cor
        self.cor               = init_cor
        self.mean0             = mean0
        self.var0              = var0
        self.varx              = varx
        self.var_vector        = np.array([varx])
        self.cor_vector        = np.array([0])
        self.lamda             = varx*(1-self.cor_vector**2)
        self.mean_params       = np.array([mean0])
        self.prec_params       = np.array([1/var0+1/varx])
        self.final_prec_params = np.array([1/var0])
        self.prev_element      = 0
        self.prev_cor          = 0
        self.q                 = q
        
    
    def additive_matrix(self, x, i):
        """This method is used for the update of the posterior parameters for
            each run length hypothesis. 
            See equations (21)-(22) and (31)-(32) from [1].
        """
        if i == 1:
            # the case when r_t = 0          
            self.quantity = x/self.varx
            self.additive = np.array([self.quantity])
        
        if i == 2:
            # the case when r_t = 1 
            self.quantity = ((1-self.cor)/self.lamda)*x - \
                ((self.cor-self.cor**2)/(self.lamda))*self.prev_element
            self.additive = np.append(self.additive,self.quantity)
            
            
        if i == 3:
            # the case when r_t = 2
            self.quantity = ((1-self.cor)/self.lamda)*x + \
                self.prev_element*((1-self.cor)/self.lamda)*(-self.cor)
            self.additive = np.append(self.additive,self.quantity)
            
        
        if i>3:
            # the case when r_t > 2
            self.quantity = ((1-self.cor)**2/self.lamda)*self.prev_element + \
                ((1-self.cor)/self.lamda)*x - \
                    ((1-self.cor)/self.lamda)*self.prev_element
            self.additive = np.append(self.additive,self.quantity)
    
        return self.additive
    
    
    def log_upm(self, t, x):
        """This method computes the upm function for each run length hypothesis.
            See Eq.(13),(19),(29) from [1].
        """
        post_stds  = np.sqrt(self.var_params[:t])  

        mean_params_dep = self.mean_params[1:t] + self.cor*(self.prev_element-self.mean_params[1:t])
        
        post_means = np.append(self.mean_params[0],mean_params_dep)
        
        self.mean_params_dep = post_means
        self.prev_element = x

        return norm(post_means, post_stds).logpdf(x)
    
    
    def update_params_bayes(self, t, x, q):
        """Upon observing a new datum x at time t, update the parameters for all
            run length hypotheses. 
        """
        if self.cor>1:
            self.cor = 0.9
        if self.cor<-1:
            self.cor = -0.9
        
        self.cor_vector   = np.append(self.cor_vector,self.cor)
        self.var_vector   = np.append(self.var_vector,self.varx)
       
        if t==1:
            self.final_prec_params = np.append([1/self.var0],self.prec_params)
        if t>1:
       
            new_prec_params  = self.prec_params + ((1-self.cor)**2/self.lamda)
            self.prec_params = np.append([1/self.var0 + 1/self.varx], new_prec_params)
            self.final_prec_params = np.append([1/self.var0],self.prec_params)

        for i in range(1,t+1):
            add_matrix = self.additive_matrix(x, i)
        

        new_mean_params  = (self.mean_params * self.final_prec_params[:-1] + \
                            add_matrix) / self.prec_params
    
        self.mean_params = np.append([self.mean0], new_mean_params)
    
    
    def update_params_sd(self, train_data, d):
        """ Using the module sd_ar it is filtered the time-varying correlation with 
            a Score-Driven AR(1) model and infer the variance from the static parameters.
        """
        self.cor,self.parameters  = sd.model_estimation(train_data,d,self.parameters)
        self.varx =self.parameters[-1]
        self.prev_cor = self.cor
        
        
    def initial_cond(self):
        """Initialize the variance and the correlation every time there is a new regime.
        """
        self.varx = self.init_var
        self.cor  = self.init_cor
         
        
    @property 
    def var_params(self):
        """Helper function for computing the posterior variance.
        """        
        if self.q == 0 or self.q == 1/2:
            
            return (1./self.final_prec_params)*((1-self.cor_vector)**2) + (self.var_vector)*(1-self.cor_vector**2)

        if self.q == 1:
            
            return (1./self.final_prec_params) + self.var_vector



    @property 
    def mean_dep_params(self):
        """Helper function for computing the posterior mean.
        """         
        return (1-self.cor_vector)*self.mean_params + (self.cor_vector)*self.prev_element
            