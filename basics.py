import pandas as pd, numpy as np, time, math
from scipy.stats import t, chi2

#@markdown **_class_** : two_sample_test

class two_sample_test:

  '''
  Methods
  -------

  \t self.fit(X)
  \t **Return**
  \t - self.tt_result : (dataframe), reuslts of t-test whether it rejects
  \t null hypothesis or not for each respective variables
  '''
  def __init__(self, tt_alpha=0.05, chi_alpha=0.05, n_interval=100):

    '''
    Parameters
    ----------

    \t tt_alpha : (float), two-tailed alpha e.g. 1% = 0.01 (rejection region)
    \t chi_alpha : (float), one-tailed alpha for chi-square test (default=0.05)
    \t n_interval : (int), number of intervals for chi-sqaure test (defaul=100)
    '''
    self.tt_alpha = tt_alpha
    self.chi_alpha, self.n_interval = chi_alpha, n_interval

  def fit(self, X1, X2):

    '''
    Parameters
    ----------
    
    \t X1, X2 : (array-like, sparse matrix), shape = [n_samples, n_features]
    '''
    features = list(set(X1.columns).intersection(X2.columns))
    columns = ['variable','n_X1','n_X2','t_stat','p_value_1t','crit_val','chi_p_value']
    
    for (n,var) in enumerate(features):
      x1, x2 = X1.loc[X1[var].notna(),var], X2.loc[X2[var].notna(),var]
      n_x1, n_x2 = x1.shape[0], x2.shape[0]
      # check whether two means are the same
      t_stat, p_value = self.__ttest(x1, x2)
      # check whether the proportion in respective bins are the same
      crit_val, chi_p_value = self.__chi_square(x1,x2)
      p = np.array([var, n_x1, n_x2, t_stat, p_value, crit_val, chi_p_value]).reshape(1,-1)
      if n==0: a = p
      else: a = np.vstack((a,p))

    # Convert variables to number
    a = pd.DataFrame(a, columns=columns)
    for var in columns[1:]:
      a[var] = pd.to_numeric(a[var], errors='ignore')
    a['reject_tt_H0'] = False
    a.loc[a['p_value_1t']<self.tt_alpha/2,'reject_tt_H0'] = True
    a['reject_chi_H0'] = False
    #a.loc[a['chi_p_value']<self.chi_alpha,'reject_chi_H0'] = True
    self.t_result = a

  def __ttest(self, x1, x2):
    
    '''
    Two-sample t-test using p-value 
    Null Hypothesis (H0) : mean of two intervals are the same
    Alternative (Ha) : mean of two intervals are different
    '''
    # calculate means
    mean1, mean2 = np.mean(x1), np.mean(x2)
    
    # calculate standard deviations
    std1, std2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    
    # calculate standard errors
    n1, n2 = len(x1), len(x2)
    se1, se2 = std1/np.sqrt(n1), std2/np.sqrt(n2)
    sed = np.sqrt(se1**2 + se2**2)
   
    # t-statistic
    #(when sed=0 that means x1 and x2 are constant)
    if sed>0: t_stat = (mean1-mean2) / sed
    else: t_stat = float('Inf')
    
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
    dof = len(a) - 1 #<-- degree of freedoms
    crit_val = sum((a-exp)**2/exp) + sum((b-exp)**2/exp)
    p_value = 1-chi2.cdf(crit_val, df=dof)
    return crit_val, p_value

#@markdown **_class_** : outliers
  
class outliers:
  
  '''
  Each of approaches fundamentally predetermines lower and upper bounds where any point 
  that lies either below or above those points is identified as outlier. In addition,
  lower and upper bounds are kept within existing min and max values, respectively.
  Once identified, such outlier is then capped at a certain value above the upper bound 
  or floored below the lower bound.

  Methods
  -------

  \t self.fit(X)
  \t **Return**
  \t - self.cap_df : (dataframe), table of lower and upper limits for each respective variables
  \t - self.capped_X : (dataframe), X variables that are capped and floored
  '''
  def __init__(self, method='gamma', pct_alpha=1, beta_sigma=3, 
               beta_iqr=1.5, pct_limit=10, n_interval=100):
    
    '''
    Parameters
    ----------

    \t method : (str), method of capping outliers i.e. 'percentile', 'sigma', 'interquartile', and 'gamma'
    \t pct_alpha : (float), percentage of one-tailed alpha (default=1)
    \t beta_sigma : (float), standard deviation (default=3)
    \t beta_iqr : (float), inter-quartile-range multiplier (default=1.5)
    \t pct_limit : (float), limit of percentile from both ends of distribution (defalut=10)
    \t n_interval : (int), number of intervals within per_limit (default=100)
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
    
    \t X : (dataframe), array-like shape(n_sample, n_feature)
    \t delta_asec : (boolean), True = ascendingly ordered delta
    \t gamma_asec : (boolean), True = ascendingly ordered gamma
    '''
    # Slope (Delta)
    diff_X, divisor = np.diff(X), abs(np.array(X.copy()))
    divisor[divisor==0] = 1
    if delta_asec == True: divisor = divisor[:len(diff_X)]
    else: divisor = -1*divisor[1:]
    delta = diff_X/divisor

    # Change in slope (Gamma)
    diff_del = np.diff(delta)
    divisor = abs(np.array(delta))
    divisor[divisor==0] = 1
    if gamma_asec == True: divisor = divisor[:len(diff_del)]
    else: divisor = -1*divisor[1:]
    gamma = diff_del/divisor
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
    a = self.__percentile(self.__nonan(X))
    
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

  def __iqr_cap(self, X):
    
    a = self.__nonan(X)
    q1, q3 = np.percentile(a,25), np.percentile(a,75)
    low = q1 - (q3-q1)*self.beta_iqr
    high = q3 + (q3-q1)*self.beta_iqr
    return low, high
  
  def __sigma_cap(self, X):
    
    a = self.__nonan(X)
    mu, sigma = np.mean(a), np.std(a)*self.beta_sigma
    low, high = mu-sigma, mu+sigma
    return low, high
  
  def __pct_cap(self, X):
    
    a = self.__nonan(X)
    low = np.percentile(a,self.pct_alpha)
    high = np.percentile(a,100-self.pct_alpha)
    return low, high
  
  def fit(self, X):
    
    # Convert data into dataframe
    self.capped_X = self.__to_df(X)

    # Lower and Upper bound dataframe
    self.cap_df = pd.DataFrame(columns=['variable','lower','upper'])
    self.cap_df['variable'] = self.field_names
    
    for n,var in enumerate(self.field_names):
      a = self.capped_X[var]
      if self.method == 'gamma':
        low, high = self.__gamma_cap(a)
      elif self.method == 'interquartile':
        low, high = self.__iqr_cap(a)
      elif self.method == 'sigma':
        low, high = self.__sigma_cap(a)
      elif self.method == 'percentile':
        low, high = self.__pct_cap(a)

      # cap values in dataframe
      low = max(low, np.nanmin(a))
      high = min(high, np.nanmax(a))
      self.capped_X.loc[(a.notna()) & (a<low),var] = low
      self.capped_X.loc[(a.notna()) & (a>high),var] = high
      self.cap_df.iloc[n,1], self.cap_df.iloc[n,2] = low, high
  
  def __to_df(self, X):
    
    '''
    Convert all input to an array
    '''
    a = X.copy()
    if isinstance(a, (pd.core.series.Series, list)):
      self.field_names = ['X_1']
      a = np.array(a).reshape(len(a),1)
    elif isinstance(a, pd.core.frame.DataFrame):
      self.field_names = a.columns
      a = np.array(a)
    elif isinstance(a, np.ndarray) & (a.size==len(a)):
      self.field_names = ['X_1']
      a = np.array(a).reshape(len(a),1)
    elif isinstance(a, np.ndarray) & (a.size!=len(a)):
      self.field_names = self.__field_names(a.shape[1])
      a = np.array(a)
    return pd.DataFrame(a, columns=self.field_names)
  
  def __field_names(self, n):
    
    digit = 10**math.ceil(np.log(n)/np.log(10))
    return ['X_' + str(digit+n)[1:] for n in range(n)]
    
  def __nonan(self, X):
    
    '''
    Convert X into array and eliminate all missing values
    '''
    a = np.array(X.copy())
    return a[~np.isnan(a)]
