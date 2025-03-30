"""
This code, for given data, computes the run length posterior matrix and the most
likely change-points for the three different models; BOCPD, MBO and MBOC as they 
are described in the paper:

[1]"Online Learning of Order Flow and Market Impact with 
Bayesian Change-Point Detection Methods" 
by Tsaknaki,Lillo,Mazzarisi, 2023

see: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4500960

The BOCPD model was introduced in the paper

[2]"Bayesian Online Change-Point Detection" by Adams and MacKay 

and this code extends its framework (in [1]) to the dependent case when the data
within regime is Markovian.

Author: Yvonni Tsaknaki
"""
import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import numpy as np
from   scipy.special import logsumexp

class SDBocpd:
    
    def __init__(self, T, d, q):
        
        self.T       = T
        self.d       = d
        self.q       = q
        self.ml_path = np.ones(self.T+1)
        self.ml_cps  = [0]
        self.run_length_mtx = -np.inf * np.ones((self.T+1, self.T+1))
    
    
    def bocpd(self,data, model, hazard, true_cps = [], plot = True):
        """This function implements Algorithm 1 from: "Bayesian Online Changepoint Detection" 
            by R.P.Adams and D.J.C.MacKay. 
        """
        
        if self.d != 0 and self.d != 1/2:
            raise ValueError("d should be either 0 or 1/2") 
    
        if len(data) == 0:
            raise ValueError("The data set is empty")
                
        # We work in the log-space and then convert back the run-length posterior matrix
        # See section 3.5.3 from "Machine Learning: A Probabilistic Perspective" by K.P.Murphy
    
        log_run_length       = -np.inf * np.ones((self.T+1, self.T+1))
        log_run_length[0, 0] = 0             
        pmean                = np.empty(self.T)    
        pvar                 = np.empty(self.T)     
        log_message          = np.array([0])  
        argmax               = 1
        cp_t                 = 0
        new_data             = []
        
        # The enumerated steps are from Algo 1 of Adams and MacKay.
        for t in range(1, self.T+1):
    
            # 2. Observe new datum.
            x_t = data[t-1]
            
            # 9. Perform predictions.
            pmean[t-1] = np.sum(np.exp(log_run_length[t-1, :t]) * model.mean_dep_params[:t])
            pvar[t-1]  = np.sum(np.exp(log_run_length[t-1, :t]) * model.var_params[:t])
    
            # 3. Evaluate predictive probability.
            log_upms = model.log_upm(t, x_t)
    
            # 4. Calculate growth probabilities.
            log_growth_probs = log_upms + log_message + hazard.log_cnjgt_hazard(t)

            # 5. Calculate changepoint probabilities.
            log_cp_prob = logsumexp(log_upms + log_message + hazard.log_hazard(t))
    
            # 6. Calculate evidence
            new_log_joint = np.append(log_cp_prob, log_growth_probs)
    
            # 7. Determine run length distribution.
            log_run_length[t, :t+1]  = new_log_joint 
            log_run_length[t, :t+1] -= logsumexp(new_log_joint)
    
            # Find when r_t<=tol -- here we find the cps
            if log_run_length[t][argmax] == max(log_run_length[t]):
                argmax += 1
            else:
                argmax = np.where(log_run_length[t, :t+1] == np.amax(log_run_length[t, :t+1]))[0][0]
                argmax += 1
               
            if argmax<=2:
                cp_t = t
                new_data.append(0)
                self.ml_cps.append(t)
            
            if t not in self.ml_cps:
                new_data[cp_t:t+1] = (data[cp_t:t+1] - np.mean(data[cp_t:t+1]))
                # new_data[cp_t:t+1] = SubtrckToList(data[cp_t:t+1],pmean[t-1])
            if self.q == 1:
    
                if t<40:
                    model.initial_cond()
                    
                if t>=40 and t-cp_t>=10:
                    model.update_params_sd(new_data[:t],self.d)
                    
            # 8. Update sufficient statistics with Bayes
            model.update_params_bayes(t, x_t, self.q)
    
            # Pass message.
            log_message = new_log_joint
            
            # Find the most likely path
            position_max_element = np.where(log_run_length[t-1, :t] == np.amax(log_run_length[t-1, :t]))
            self.ml_path[t-1] = self.T-position_max_element[0][0]
        
        position_max_element = np.where(log_run_length[self.T, :self.T+1] == np.amax(log_run_length[self.T, :self.T+1]))
        self.ml_path[self.T] = self.T-position_max_element[0][0]
        
        self.run_length_mtx = np.exp(log_run_length)
        
        if plot == True:
            
            self.__plt_run_length_posterior(data, true_cps, pmean, pvar)
            
        return self.run_length_mtx, self.ml_cps
    
    
    def SubtrckToList(self,lista,number):
        new_lista = []
        for i in lista:
            new_lista.append(i-number)
        return new_lista
    
        
    def __plt_run_length_posterior(self, data, true_cps, pmean, pvar): 

        fig, axes = plt.subplots(2, 1, figsize=(16,8))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)
    
        ax1, ax2 = axes
        
        ax1.scatter(range(1, self.T+1), data)
        ax1.plot(range(1, self.T+1), data)
        ax1.set_xlim([0, self.T])
        _2std = 2 * np.sqrt(pvar)
        line1, = ax1.plot(range(1, self.T+1), pmean - _2std, c='black', ls='--')
        ax1.plot(range(1, self.T+1), pmean + _2std, c='black', ls='--')
        ax1.set_title("Real Data",fontsize=25)
        line2, = ax1.plot(range(1, self.T+1), pmean, c='black',linewidth=1,ls='-')
        ax1.legend([line1,line2],['pvar','pmean'],fontsize="19")
        ax1.tick_params(labelsize=22)
        ax1.set_ylabel(r'$x_t$',fontsize=26)
        ax1.margins(0)

        im = ax2.imshow(np.rot90(self.run_length_mtx,k=1), aspect='auto', cmap='gray_r',norm=LogNorm(vmin=0.0001, vmax=1))
        cax = fig.add_axes([1.0, 0.07, 0.018, 0.4])
        fig.colorbar(im, cax=cax, orientation='vertical')
        line3, = ax2.plot(self.ml_path,color='red',linewidth=2)
        
        ticks = list(range(self.T, int(min(self.ml_path)), -int(self.T/4)))
        tick_labels = list(range(0, self.T-int(min(self.ml_path)), int((self.T)/4)))
        ax2.set_ylim([self.T,int(min(self.ml_path))])
        
        ax2.set_yticks(ticks)
        ax2.set_yticklabels(tick_labels,fontsize=22)
        ax2.legend([line3], ['most likely path'],fontsize="19")
        ax2.tick_params(labelsize=22)
        ax2.set_ylabel(r'$r_t$',fontsize=26)
        ax2.set_title('Run Length Posterior of MBOC',fontsize=25)
        ax2.margins(0)
        
        for cp_f in true_cps:
            ax1.axvline(cp_f, c='red', ls='dotted',lw=2)
    
        plt.tight_layout()
        
 
class Hazard:
    
    def __init__(self,T,hazard,constant = True):
        
        if hazard < 0 or hazard > 1:
            raise ValueError("The Hazard rate should be a value in the interval [0,1]")
        
        self.T        = T
        self.hazard   = hazard
        self.constant = constant
 
    
    def log_hazard(self,t):
        
        if self.constant:
            
            return np.log(self.hazard)*np.ones(t)
    
    
    def log_cnjgt_hazard(self,t):
        
        if self.constant:

            return np.log(1-self.hazard)*np.ones(t)

    
    
    
   