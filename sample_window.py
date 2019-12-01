# Library
import pandas as pd, numpy as np, calendar
import matplotlib.pylab as plt

#@markdown **_class_** : vintage_analysis

class vintage_analysis:

  '''
  Methods
  -------

  \t self.__init__(a, prefix='m_')
  \t **Return**
  \t - self.vintage
  
  \t self.fit(per_dpd=90, obs_mth=12, per_mth=12)
  \t **Return**
  \t - self.per_dpd : (int), ever delinquent (dpd>per_dpd) within performance period
  \t - self.obs_mth : (int), observation period (month)
  \t - self.per_mth : (int), performance period (month)
  \t - self.waterfall : (dataframe), summary of excluded observations
  \t - self._by_vintage : (dataframe), summary of BADs by vintage
  \t - self._all_ : (dataframe), sum of BADs from all vintages
  
  \t self.plot(n_col=2, width=0.6, height=4.5, fname=None)
  \t **Return**
  \t - plot cumulative ever-delinquent by vintage
  
  \t self.plot_all(width=0.5, fname=None)
  \t **Return**
  \t - plot total cumulative ever-delinquent
  '''
  def __init__(self, a, prefix='m_'):
    
    '''
    a : (dataframe)
    ===========================================
    |   | cif_no	| max_dpd	|   mdata	  | mob |
    -------------------------------------------
    | 0	| YX13KX	|    1	  | 31JAN2017 |	 0  |
    | 1	| OX123C	|    1	  | 31JAN2017 |	 0  |
    ===========================================
    - cif_no : (str), 12-digit integer
    - max_dpd : (int), maximum dpd within a month
    - mdata : (str), date of record (format=DDMMYYYY)
    - mob : (int): Month-on-Book
    '''
    self.prefix = prefix
    self.__vintage_df(a)
    
  def __vintage_df(self, a):

    '''
    Rearrange data into vintage-based format
    '''
    self.digit = 10**max(int(len(str(a['mob'].max()))),2)
    self.n_mob = np.unique(a['mob'])
    for (n,mob) in enumerate(self.n_mob):
      b = a.loc[a['mob']==mob,['cif_no','max_dpd']]
      b = b.rename(columns={'max_dpd':'m_'+str(n+self.digit)[1:]}).astype(int)
      if n==0: vintage = b
      else: vintage = vintage.merge(b, on='cif_no', how='outer')

    # convert mdata to actual date as well as create yyyymm field
    b = a.loc[a['mob'].astype(int)==0,['cif_no','mdata']]
    b = self.__yyyymm(b.rename(columns={'mdata':'date'}))
    b = b.merge(vintage, how='outer', on='cif_no')
    
    # convert cif_no into 12-digit number
    b['cif_no'] = ((10**12 + b['cif_no']).astype(str)).str[1:]
    self.vintage = b

  def __yyyymm(self, a):

    '''
    Convert date (str) into datetime format as well as create yyyymm field
    '''
    dt = a.copy()
    u_months = dict((v.upper(),k) for k,v in enumerate(calendar.month_abbr))
    dt['day'] = dt['date'].str[:2].astype(str)
    dt['year'] = dt['date'].str[5:].astype(str)
    dt['month'] = dt['date'].str[2:5].apply(lambda n: u_months[n]).astype(str)
    self.dt_var, self.yymm = 'date_' + self.prefix + '00', 'yyyymm_' + self.prefix + '00'
    dt[self.dt_var] = dt['day'] + '/' + dt['month'] + '/' + dt['year']
    dt[self.dt_var] = pd.to_datetime(dt[self.dt_var], format='%d/%m/%Y')
    dt[self.yymm] = dt['year'].astype(int).mul(100) + dt['month'].astype(int)
    return dt[['cif_no', self.dt_var, self.yymm]]

  def __mth(self, mob):

    '''
    Parameters
    ----------

    \t mob : (array-like), list of month-on-book (int)
    \t **Return**
    \t - list of month-on-book columns with predefined prefix
    '''
    return [self.prefix + str(n+self.digit)[1:] for n in mob]
  
  def fit(self, per_dpd=90, obs_mth=12, per_mth=12):

    '''
    day-past-due (dpd) is a number of days that loan payment has not 
    been made as of its due date

    Parameters
    ----------

    \t per_dpd : (int), ever delinquent (dpd>per_dpd) within performance period
    \t obs_mth : (int), observation period (month)
    \t per_mth : (int), performance period (month)

    (1) must not contain np.nan throughout whole period
    (2) must not have delinquency at the very first month of performance period
    '''
    # exclusion table
    item, amount = ['samples','incomplete','delinquent','total'], list()
    
    # take samples from predefined window
    columns = ['cif_no', self.dt_var, self.yymm]
    n_period = self.__mth(range(obs_mth + per_mth))
    a = self.vintage[columns + n_period].copy() 
    amount.append(len(self.vintage)) #<-- number of samples

    # contain no NaN throughout observation and performance periods
    nonan = (np.isnan(a[n_period]).astype(int).sum(axis=1)==0)
    a = a.loc[nonan].reset_index(drop=True)
    amount.append(-sum(~nonan.astype(bool))) #<-- number of missings

    # not delinquent at the biginning of performance period
    not_delq = ((a[n_period[obs_mth]]>per_dpd).astype(int)==0)
    a = a.loc[not_delq].reset_index(drop=True)
    amount.append(-sum(~not_delq.astype(bool))) #<-- number of delinquent cifs

    # total number of samples
    amount.append(len(a))
    self.waterfall = pd.DataFrame({'item':item,'amount': amount})

    # performance window
    perf = a.iloc[:,-per_mth:].copy()
    perf_var = list()
    for (n, period) in enumerate(perf.columns):
      p_var = 'p_' + period; perf_var.append(p_var)
      perf[p_var], c_delq = 0, (perf[period]>per_dpd).values
      if n==0: perf.loc[c_delq, p_var] = 1
      else: perf.loc[(perf.iloc[:,-2]==1) | c_delq, p_var] = 1

    # defualt flag
    ever_delq = (perf.iloc[:,-per_mth:].values.sum(axis=1)>0)
    perf['ever_' + str(per_dpd) + 'p'] = ever_delq
    kwargs = dict(left_index=True, right_index=True)
    self.perform = a[columns].merge(perf.iloc[:,-(per_mth+1):], **kwargs)

    # summary by vintage
    self.__by_vintage(perf_var)
    self.per_dpd = per_dpd
    self.obs_mth, self.per_mth = obs_mth, per_mth

  def __by_vintage(self, perf_var):

    '''
    Group performance by vintage
    '''
    a = self.perform[[self.yymm] + perf_var].groupby([self.yymm]).agg(['sum'])
    columns = ['asof'] + list([n[0] for n in a.columns])
    b = pd.DataFrame(a.reset_index().values, columns=columns)
    c = self.perform[[self.yymm,'cif_no']].groupby([self.yymm]).agg(['count'])
    c = pd.DataFrame(c.reset_index().values, columns=['asof','count']) 
    self._by_vintage = c.merge(b, on='asof', how='left')
    c = self._by_vintage.copy(); c['asof'] = 1
    c = c.groupby('asof').agg(['sum'])
    self._all_ = pd.DataFrame(c.values, columns=[n[0] for n in c.columns])

  def plot(self, n_col=2, width=0.6, height=4.5, fname=None):

    a = self._by_vintage.copy()
    columns = np.sort([n for n in a if str(n).find('p_')>-1]).tolist()
    fig, axis = plt.subplots(1,1,figsize=(width*a.shape[1],height))
    n_mth, n_asof = len(columns), len(a)
    x_ticks, bad_pct = range(n_mth), np.full(n_asof,0.0)
    n_pop = np.full(n_asof,0.0)

    for (n,asof) in enumerate(a['asof'].astype(int)):
      y = a.loc[a['asof']==asof, ['count'] + columns]
      n_pop[n] = float(y['count'])
      y = ((y[columns]/n_pop[n])*100).values[0]
      bad_pct[n] = max(y)
      label = '%d (%0.2f%%)' % (asof, bad_pct[n])
      axis.plot(x_ticks, y, label=label, lw=1, ls='--')
    avg_pct = sum(bad_pct*n_pop/sum(n_pop))

    axis.set_facecolor('white')
    axis.set_ylabel('Cummulative Bad (%)', fontsize=10)
    axis.set_xlabel('Performance (months)', fontsize=10)
    axis.set_xticks(x_ticks)
    axis.set_xlim(-0.5,len(x_ticks)-0.5)
    axis.axhline(avg_pct, lw=1, color='r')
    s = r'Weighted avg. BAD: $\bf{%0.2f}$%%' % avg_pct
    axis.text(0, avg_pct*0.98, s, va='top')
    t = 'Vintage Analysis (by as-of) \n observation = %d, performance = %d, dpd = %d+' 
    axis.set_title(t % (self.obs_mth, self.per_mth, self.per_dpd), fontsize=12)
    axis.set_xticklabels(['M'+str(n) for n in x_ticks], fontsize=10)
    kwargs = dict(loc='upper left', fontsize=10, facecolor='white', 
                  edgecolor='k', ncol=n_col, fancybox=True, shadow=True, 
                  bbox_to_anchor=(1,1))
    axis.legend(**kwargs)
    axis.grid(False)
    
    if fname is not None: plt.savefig(fname)
    plt.tight_layout()
    plt.show()
  
  def plot_all(self, width=0.5, height=4.5, fname=None):

    a = self._all_.copy()
    columns = np.sort([n for n in a if str(n).find('p_')>-1]).tolist()
    fig, axis = plt.subplots(1,1,figsize=(width*a.shape[1],height))
    n_total = float(a['count'].values)
    y = (a.iloc[:,1:]/n_total*100).values[0]
    x_ticks = range(len(y))
    kwargs = dict(color='#67e6dc', alpha=0.6, lw=1)
    axis.fill_between(x_ticks,y,**kwargs)

    bbx_kw = dict(boxstyle='round',facecolor='white',alpha=0)
    s = tuple(('Sample = {:,.0f}'.format(n_total),
               '%BAD = {:,.2f}% ({:,.0f})'.format(max(y),max(y)*n_total/100)))
    axis.text(0.05, 0.95, '\n'.join(s), transform=axis.transAxes, 
              fontsize=12, va='top', ha='left', bbox=bbx_kw) 

    axis.set_facecolor('white')
    axis.set_ylabel('Cummulative Bad (%)', fontsize=10)
    axis.set_xlabel('Performance (months)', fontsize=10)
    axis.set_xticks(x_ticks)
    axis.set_xlim(-0.5,len(x_ticks)-0.5)
    t = 'Vintage Analysis (All) \n observation = %d, performance = %d, dpd = %d+' 
    axis.set_title(t % (self.obs_mth, self.per_mth, self.per_dpd), fontsize=12)
    axis.set_xticklabels(['M'+str(n) for n in x_ticks], fontsize=10)
    axis.grid(False)
    
    if fname is not None: plt.savefig(fname)
    plt.tight_layout()
    plt.show()
