import pandas as pd, numpy as np, time, math
from scipy.stats import t, chi2

class outliers:
  
  '''
  Each of approaches fundamentally predetermines (from likelihood) lower and upper bounds 
  where any point that lies either below or above those points is identified as outlier. 
  Once identified, such outlier is then capped at a certain value above the upper bound 
  or floored below the lower bound.

  Methods
  -------

  \t self.fit(X)
  \t **Return**
  \t - self.cap_df : (dataframe), table of lower and upper limits for each respective variables
  \t - self.capped_X : (dataframe), X variables that are capped and floored
  '''
  def __init__(self, method='gamma', pct_alpha=1, beta_sigma=3, beta_iqr=1.5, pct_limit=10, n_interval=100):
    
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
    diff_X = np.diff(X)
    divisor = abs(np.array(X.copy()))
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
    List of nth percentiles 
    arranged in increasing manner
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
    return low, high

  def __iqr_cap(self, X):
    
    a = self.__nonan(X)
    q1, q3 = np.percentile(a,25), np.percentile(a,75)
    low = q1 - (q3-q1)*self.beta_iqr
    high = q3 + (q3-q1)*self.beta_iqr
    return low, high
  
  def __sigma_cap(self, X):
    
    a = self.__nonan(X)
    mu, sigma = np.mean(a), np.std(a)
    low = mu - sigma*self.beta_sigma
    high = mu + sigma*self.beta_sigma
    return low, high
  
  def __pct_cap(self, X):
    
    a = self.__nonan(X)
    low = np.percentile(a,self.pct_alpha)
    high = np.percentile(a,100-self.pct_alpha)
    return low, high
  
  def fit(self, X):
    
    # Convert data into array
    a = self.__to_array(X)
    self.capped_X = a.copy()
    
    # Lower and Upper bound dataframe
    columns= ['variable','lower','upper']
    self.cap_df = pd.DataFrame(columns=columns)
    self.cap_df['variable'] = self.field_names
    
    for n in range(a.shape[1]):
      if self.method == 'gamma':
        low1, high1 = self.__gamma_cap(a[:,n])
        low2, high2 = self.__iqr_cap(a[:,n])
        low = min(low1, low2)
        high = max(high1, high2)
      elif self.method == 'interquartile':
        low, high = self.__iqr_cap(a[:,n])
      elif self.method == 'sigma':
        low, high = self.__sigma_cap(a[:,n])
      elif self.method == 'percentile':
        low, high = self.__pct_cap(a[:,n])
      nonan = ~np.isnan(a[:,n])
      low, high = max(low, min(a[:,n])), min(high, max(a[:,n]))
      args = tuple((a[:,n], low , high))
      self.capped_X[nonan,n] = self.__cap_outlier(*args)
      self.cap_df.iloc[n,1], self.cap_df.iloc[n,2] = low, high
    self.capped_X = pd.DataFrame(self.capped_X, columns=self.field_names)
      
  def __cap_outlier(self, a, low, high):
    
    '''
    Replace values that exceed low or high
    '''
    c = a[~np.isnan(a)].copy()
    c[(c>high)], c[(c<low)] = high, low
    a[~np.isnan(a)] = c
    return a
  
  def __to_array(self, X):
    
    '''
    Convert all input to an array
    '''
    a = X.copy()
    if isinstance(a, (pd.core.series.Series, list)):
      self.field_names = ['X_1']
      return np.array(a).reshape(-1,1)
    elif isinstance(a, pd.core.frame.DataFrame):
      self.field_names = a.columns
      return np.array(a.values).reshape(a.shape)
    elif isinstance(a, np.ndarray) & (a.size==len(a)):
      self.field_names = ['X_1']
      return np.array(a).reshape(-1,1)
    elif isinstance(a, np.ndarray) & (a.size!=len(a)):
      self.field_names = self.__field_names(a.shape[1])
      return np.array(a)
  
  def __field_names(self, n):
    
    digit = 10**math.ceil(np.log(n)/np.log(10))
    return ['X_' + str(digit+n)[1:] for n in range(n)]
    
  def __nonan(self, X):
    
    '''
    Convert X into array and eliminate all missing values
    '''
    a = np.array(X.copy())
    return a[~np.isnan(a)]
