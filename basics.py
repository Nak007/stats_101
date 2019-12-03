import pandas as pd, numpy as np, time, math
from scipy.stats import t, chi2
  
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
    a.loc[a['chi_p_value']<self.chi_alpha,'reject_chi_H0'] = True
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
