import pandas as pd, numpy as np, time, math, random
from scipy.stats import t, chi2
import ipywidgets as widgets
from IPython.display import display

class two_sample_test:

    '''
    Only applicable for numeric variables !!!
    
    Methods
    -------

    self.fit(X1, X2)
    \t determine the statistical difference between two sets of samples
    
    self.autofit(X, frac=0.1, random_state=(0,100), max_round=10)
    \t given percent sample (pct_sample), it randomly determines sample
    \t that statistically provides equal mean and similar distribution 
    \t to original dataset for all variables. Neverthess, if no sampling
    \t satisfies the requirements within 'max_round', the round that 
    \t provides least number of violations, is selected.
    
    Return
    ------
    
    self.tt_result : dataframe 
    \t table is comprised of results from 
    \t (1) independent t-test compares the means of two groups in order to 
    \t determine whether there is statistical evidence that the associated 
    \t population means are significantly different 
    \t (2) chi-square test compares the distribution between sample and 
    \t original datasets whether they statistically fit with one another
    '''
    def __init__(self, tt_alpha=0.05, chi_alpha=0.05, n_interval=20):

        '''
        Parameters
        ----------

        tt_alpha : float, optional (default=0.05)
        \t two-tailed alpha for independent t-test (rejection region)
        
        chi_alpha : float, optional (default=0.05)
        \t one-tailed alpha for chi-square test (rejection region)
        
        n_interval : int, optional (defaul=20)
        \t number of intervals (bins) for chi-sqaure test
        '''
        self.tt_alpha = tt_alpha
        self.chi_alpha, self.n_interval = chi_alpha, n_interval
        self.__on__ = False

    def fit(self, X1, X2):

        '''
        Parameters
        ----------

        X1 : array-like, shape of (n_samples, n_features)
        \t original or comparable dataset
        
        X2 : array-like, shape of (n_samples, n_features)
        \t sample dataset
        '''
        features = list(set(X1.columns).intersection(X2.columns))
        columns = ['variable','t_stat','p_value_1t','crit_val','chi_p_value']
        if self.__on__==False: 
            self.__widgets()
            self.w_t1.value = 'Two-Sample Test . . . progress '
        n_features = len(features)
            
        for (n,var) in enumerate(features,1):
            
            self.w_t2.value = '( {:.0f}% ) '.format((n/n_features)*100) + var
            time.sleep(0.1)
            
            x1 = X1.loc[~np.isnan(X1[var]),var]
            x2 = X2.loc[~np.isnan(X2[var]),var]
            
            # both sets of record must not contain only missing values
            not_nan = (len(x1)>0) & (len(x2)>0)
            # remaining values (after nan elimination) must not be constant
            const = (len(np.unique(x1))>1) & (len(np.unique(x2))>1)
            
            if not_nan and const:
                t_stat, p_value = self.__ttest(x1, x2)
                crit_val, chi_p_value = self.__chi_square(x1,x2)
                p = np.array([var, t_stat, p_value, crit_val, chi_p_value]).reshape(1,-1)
            else: p = np.array([var, 0, 1, 0, 1]).reshape(1,-1)
     
            if n==1: a = p
            else: a = np.vstack((a,p))

        # Convert variables to number
        a = pd.DataFrame(a, columns=columns)
        for var in columns[1:]:
            a[var] = a[var].astype(float)

        a['reject_tt_H0'] = np.where(a['p_value_1t']<self.tt_alpha/2,True,False)
        a['reject_chi_H0'] = np.where(a['chi_p_value']<self.chi_alpha,True,False)
        self.tt_result = a
        
    def autofit(self, X, frac=0.1, random_state=(0,100), max_round=10):
        
        '''
        Parameters
        ----------
        
        X : array-like, shape of (n_samples, n_features)
        \t original or comparable dataset 
        
        frac : float, optional (default=0.1)
        \t fraction of axis items to return
        
        random_state : tuple of floats, (min_seed, max_seed), (default=(0,100))
        \t min and max seeds for the random number generator 
        
        max_round : int, optional (default=10)
        \t maximum number of iterations
        '''
        self.__widgets(); self.__on__ = True
        n_reject = len(X)*2 
        best_reject = n_reject+1
        n_round=1
        while (n_reject>0) & (n_round<=max_round):
            self.w_t1.value = 'Iteration : %d' % n_round 
            time.sleep(0.1); n_round+=1
            n = random.randint(random_state[0],random_state[1])
            rand_X = X.sample(frac=frac, replace=False, random_state=n)
            self.fit(X, rand_X)
            self.w_t2.value = ''
            n_reject = sum(self.tt_result[['reject_tt_H0','reject_chi_H0']].values.reshape(-1,1))
            if n_reject<best_reject:
                self.random_state = n
                best_reject=n_reject
                self.best_result = self.tt_result
        self.w_t1.value = 'Complete : ' 
        self.w_t2.value = 'Best Rejection = %d , random_state = %d' % (best_reject,self.random_state)
        print('Result : self.tt_result, self.best_result, and self.random_state')

    def __ttest(self, x1, x2):

        '''
        Two-sample t-test using p-value 
        Null Hypothesis (H0) : mean of two intervals are the same
        Alternative Hypothesis (HA) : mean of two intervals are different
        '''
        # calculate means
        mean1, mean2 = np.mean(x1), np.mean(x2)

        # calculate standard deviations
        std1, std2 = np.std(x1, ddof=1), np.std(x2, ddof=1)

        # calculate standard errors
        n1, n2 = len(x1), len(x2)
        se1, se2 = std1/np.sqrt(n1), std2/np.sqrt(n2)
        sed = np.sqrt(se1**2 + se2**2)

        # t-statistic (when sed=0 that means x1 and x2 are constant)
        if sed>0: t_stat = (mean1-mean2) / sed
        else: t_stat = 0

        # calculate degree of freedom
        a, b = se1**2/n1, se2**2/n2
        c, d = 1/(n1-1), 1/(n2-1)
        if (a+b>0): df = math.floor((a+b)/(a*c+b*d))
        else: df = n1 + n2 -2

        # one-tailed p-value
        p = 1.0-t.cdf(abs(t_stat), df)

        return t_stat, p

    def __chi_square(self, x1, x2):

        '''
        Using Chi-Square to test Goodness-of-Fit-Test
        In addition, the goodness of fit test is used to test if sample data fits a 
        distribution from a certain population.
        Null Hypothesis: two sampels are fit the expected population 
        '''
        a, pct = list(x1) + list(x2), np.arange(0,100.1,100/self.n_interval)
        bins = np.unique([np.percentile(a,n) for n in pct])
        bins[-1] = bins[-1] + 1
        a, _ = np.histogram(x1, bins=bins)
        b, _ = np.histogram(x2, bins=bins)
        a = a/sum(a); b = b/sum(b)
        exp = np.vstack((a,b)).sum(axis=0)
        exp[(exp==0)] = 1 # <-- denominator must not be null
        dof = max(len(a)-1,1) #<-- degree of freedoms
        crit_val = sum((a-exp)**2/exp) + sum((b-exp)**2/exp)
        p_value = 1-chi2.cdf(crit_val, df=dof)
        return crit_val, p_value
    
    def __widgets(self):

        '''
        Initialize widget i.e. progress-bar and text
        '''
        self.w_t1 = widgets.HTMLMath(value='Calculating . . . ')
        self.w_t2 = widgets.HTMLMath(value='')
        w = widgets.HBox([self.w_t1,self.w_t2])
        display(w); time.sleep(2)
        
class outliers:
  
    '''
    Each of approaches fundamentally predetermines lower and upper bounds where any point 
    that lies either below or above those points is identified as outlier. In addition,
    lower and upper bounds are kept within existing min and max values, respectively.
    Once identified, such outlier is then capped at a certain value above the upper bound 
    or floored below the lower bound.

    Methods
    -------
    
    self.fit(X)
    
    '''
    def __init__(self, method='gamma', pct_alpha=1, beta_sigma=3, beta_iqr=1.5, pct_limit=10, n_interval=100):

        '''
        Parameters
        ----------

        method : (str), optional (default='gamma')
        \t method of capping outliers i.e. 'percentile', 'sigma', 'interquartile', and 'gamma'
        
        pct_alpha : (float), optional (default=1)
        \t percentage of one-tailed alpha 
        
        beta_sigma : (float), optional (default=3)
        \t standard deviation 
        
        beta_iqr : (float), optional (default=1.5)
        \t inter-quartile-range multiplier 
        
        pct_limit : (float), optional (defalut=10)
        \t limit of percentile from both ends of distribution 
        
        n_interval : (int), optional (default=100)
        \t number of intervals within per_limit 
        '''
        self.method = method
        self.pct_alpha, self.beta_sigma, self.beta_iqr = pct_alpha, beta_sigma, beta_iqr
        self.pct_limit, self.n_interval = pct_limit, n_interval
        self.mid_index = int(n_interval*0.5)
        self.low_pct = int(n_interval*pct_limit/100)

    def __delta_gamma(self, X, delta_asec=True, gamma_asec=True):

        '''
        Determine rate of change

        Parameters
        ----------

        X : dataframe, array-like shape of (n_sample, n_feature)
        
        delta_asec : boolean, optional (default=True)
        \t Whether or not to rearrange delta (change in values) in ascending manner
        
        gamma_asec : boolean, optional (default=True)
        \t Whether or not to rearrange gamma (change in deltas) in ascending manner
        '''
        # Slope (Delta)
        diff_X = np.diff(X)
        divisor = np.where(np.array(X)==0,1,abs(np.array(X)))
        divisor = np.where(delta_asec==True,divisor[:len(diff_X)],-1*divisor[1:])
        delta = diff_X/divisor

        # Change in slope (Gamma)
        divisor = np.where(delta==0,1,abs(delta))
        divisor = np.where(gamma_asec==True,divisor[:len(delta)-1],-1*divisor[1:])
        gamma = np.diff(delta)/divisor
        return delta, gamma
  
    def __percentile(self, X):

        '''
        List of nth percentiles arranged in increasing manner
        '''
        n_range = np.arange(self.n_interval+1)
        p_range = [n/self.n_interval*100 for n in n_range]
        return [np.percentile(X,p) for p in p_range]
  
    def __gamma_cap(self, X):

        # Copy and eliminate all nans
        a = self.__percentile(X.copy())

        # Low side delta and gamma. Gamma is arranged in reversed order 
        # as change is determined towards the lower number.
        delta, gamma = self.__delta_gamma(a, gamma_asec=False)
        slope, chg_rate = delta[:self.mid_index], gamma[:(self.mid_index-1)]

        # Low cut-off and index of maximum change (one before)
        min_index = np.argmax(chg_rate[:self.low_pct]) + 1
        low_cut = min_index/self.n_interval*100 #<-- percentile
        low = np.percentile(a, low_cut)

        # Recalculate for high-side delta and gamma (ascending order)
        delta, gamma = self.__delta_gamma(a)
        slope, chg_rate = delta[self.mid_index:], gamma[self.mid_index:]

        # High cut-off and index of maximum change (one before)
        max_index = np.argmax(chg_rate[-self.low_pct:])-1
        max_index = self.mid_index + max_index - self.low_pct
        high_cut = (max_index/self.n_interval*100)+50 #<-- percentile
        high = np.percentile(a, high_cut)

        # Compare result against IQR
        iqr_low, iqr_high = self.__iqr_cap(a)
        low, high = min(low, iqr_low), max(high, iqr_high)
        return low, high

    def __iqr_cap(self, a):

        q1, q3 = np.percentile(a,25), np.percentile(a,75)
        low = q1 - (q3-q1)*self.beta_iqr
        high = q3 + (q3-q1)*self.beta_iqr
        return low, high
  
    def __sigma_cap(self, a):

        mu, sigma = np.mean(a), np.std(a)*self.beta_sigma
        low, high = mu-sigma, mu+sigma
        return low, high
  
    def __pct_cap(self, a):

        low = np.percentile(a,self.pct_alpha)
        high = np.percentile(a,100-self.pct_alpha)
        return low, high
  
    def fit(self, X):
        
        '''
        Parameters
        ----------
        
        X : array-like shape of (n_sample, n_feature)
        
        Return
        ------
        
        self.cap_df : dataframe, array-like shape of (n_feature, ['variable','lower','upper'])
        \t table of lower and upper limits

        self.capped_X : dataframe, array-like shape of (n_sample, n_feature)
        \t X variables that are capped and floored
        
        self.complete_nan : list of str
        \t list of variables that contain only missing values
        '''
        # Convert data into dataframe
        self.__widgets()
        self.capped_X = self.__to_df(X)

        # Lower and Upper bound dataframe
        self.cap_df = pd.DataFrame(columns=['variable','lower','upper'])
        self.cap_df['variable'] = self.field_names
        
        self.w_t1.value = self.method.upper() + ' ==> '
        n_features = len(self.field_names)
        self.complete_nan = list()
        
        for (n,var) in enumerate(self.field_names):
            
            self.w_t2.value = '( {:.0f}% ) '.format(((n+1)/n_features)*100) + var
            time.sleep(0.1)
            
            # eliminate np.nan
            a = self.capped_X[var].copy()
            nonan = a.loc[~np.isnan(a)]
            
            if len(nonan) > 0:
                if self.method == 'gamma':
                    low, high = self.__gamma_cap(nonan)
                elif self.method == 'interquartile':
                    low, high = self.__iqr_cap(nonan)
                elif self.method == 'sigma':
                    low, high = self.__sigma_cap(nonan)
                elif self.method == 'percentile':
                    low, high = self.__pct_cap(nonan)
                else: low, high = self.__pct_cap(nonan)
                # cap values in dataframe
                low = max(low, np.nanmin(a))
                high = min(high, np.nanmax(a))
                self.capped_X.loc[(~np.isnan(a)) & (a<low),var] = low
                self.capped_X.loc[(~np.isnan(a)) & (a>high),var] = high
            else: 
                low, high = np.nan, np.nan
                self.complete_nan.append(var)
                
            self.cap_df.iloc[n,1] = low
            self.cap_df.iloc[n,2] = high
        
        self.w_t1.value = 'Complete : '
        n_nan = ', no. of variables w/ complete NaN = %s' % '{:,.0f}'.format(len(self.complete_nan))
        self.w_t2.value = 'no. of variables = %s' % '{:,.0f}'.format(n_features) + n_nan
        print('Result : self.cap_df, self.capped_X, and self.complete_nan')
  
    def __to_df(self, X):

        a = X.copy()
        if isinstance(a,list) | (isinstance(a,np.ndarray) & (a.size==len(a))):
            self.field_names = ['X_1']
            a = np.array(a).reshape(len(a),1)
        elif isinstance(a, pd.core.series.Series):    
            self.field_names = a.name
            a = np.array(a).reshape(len(a),1)
        elif isinstance(a, pd.core.frame.DataFrame):
            self.field_names = a.columns
            a = np.array(a)
        elif isinstance(a, np.ndarray) & (a.size!=len(a)):
            self.field_names = self.__field_names(a.shape[1])
            a = np.array(a)
        return pd.DataFrame(a, columns=self.field_names)
  
    def __field_names(self, n):

        digit = 10**math.ceil(np.log(n)/np.log(10))
        return ['X_' + str(digit+n)[1:] for n in range(n)]
    
    def __widgets(self):

        '''
        Initialize widget i.e. progress-bar and text
        '''
        self.w_t1 = widgets.HTMLMath(value='Calculating . . . ')
        self.w_t2 = widgets.HTMLMath(value='')
        w = widgets.HBox([self.w_t1,self.w_t2])
        display(w); time.sleep(2)
