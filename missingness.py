import pandas as pd, numpy as np
import matplotlib.pylab as plt

#@markdown **_class_** : overview

class overview:
  
  '''
  Methods
  -------

  \t self.fit(X)
  \t **Return**
  \t - plot overview of missingness from three different aspects
  \t - self.missing_pct_ : (dataframe), table of missingness (%) by variable
  \t - self.n_records : (float), number of records
  '''
  def __init__(self):
    
    '''
    No initial input is required
    '''
    self.bbx_kw = dict(boxstyle='round', facecolor='white', edgecolor='#4b4b4b', alpha=1, pad=0.6)
    self.bar_kw = dict(alpha=1, width=0.7, color='#d1ccc0', lw=1, align='center', 
                       edgecolor='#4b4b4b', hatch='////')
    self.txt_kw = dict(ha='center',va='bottom', rotation=0, color='#4b4b4b',fontsize=10)
    self.pie_kw = dict(explode=(0,0), shadow=True, startangle=90, labels=None, 
                       colors=['#fff200','#d1ccc0'], textprops=dict(color='#4b4b4b',fontsize=12))
    
  def fit(self, X, threshold=0.05, cutoff=0.05, 
          figsize=[(12,4),(12,5.5),(12,4)], folder=None):
    
    '''
    Parameters
    ----------

    \t X : (dataframe), shape=(n_samples, n_features)
    \t threshold : (float), percentage of missingness cutoff
    \t cutoff : (float), number of records cutoff
    \t figsize : (list), list of figsize tuples

    Return
    ------

    \t self.missing_pct_ : (dataframe), table of missingness (%) by variable
    \t self.n_records : (float), number of records
    '''
    ch_name, fname = ['pie_chart.png','by_variable.png','by_record.png'], np.full(3,None)
    if isinstance(X, pd.core.frame.DataFrame):
      if folder!=None: fname = [folder+n for n in ch_name]
      self.threshold, self.cutoff = threshold, cutoff
      self.n_records = len(X)
      self.__pie_charts(X, figsize[0])
      self.__by_variable(X, figsize[1])
      self.__by_record(X, figsize[2])

  def __pie_charts(self, X, figsize=(12,4), fname=None):

    '''
    There are three pie charts illustrating
    (1) Number of variables (with and without nan)
    (2) Number of records (with and without nan)
    (3) Number of data points (with and without nan)
    '''
    fig = plt.figure(figsize=figsize)
    axis = [plt.subplot2grid((1,3),(0,n)) for n in range(3)]
    kwargs = dict(title='Variables', note='Number of variables')
    self.__plot_pie(axis[0], self.__missing_var(X), ['w/o missing','w/ missing'], **kwargs)
    kwargs = dict(title='Records', note='Number of records')
    self.__plot_pie(axis[1], self.__complete_records(X), ['complete','incomplete'], **kwargs)
    kwargs = dict(title='Data points', note='Number of data points')
    self.__plot_pie(axis[2], self.__missing_data(X), ['values','missing'], **kwargs)
    plt.tight_layout()
    if fname != None: plt.savefig(fname)
    plt.show()

  def __missing_var(self, X): 
    
    '''
    how many variables that have missing data?
    '''
    n_features = len(X.columns)
    a = np.isnan(X.values).astype(int)
    n_missing = sum([n if n==0 else 1 for n in sum(a).tolist()])
    return [n_features-n_missing, n_missing]

  def __complete_records(self, X):
    
    '''
    how many records that are complete?
    '''
    n_complete = len(X.dropna(how='any'))
    return [n_complete, len(X)-n_complete]

  def __missing_data(self, X):
    
    '''
    how many data points are missing?
    '''
    n_points = X.values.size
    n_missing = sum(sum(np.isnan(X.values).astype(int)))
    return [n_points-n_missing, n_missing]

  def __plot_pie(self, axis, sizes, labels, title=None, note=None):
    
    self.pie_kw['labels'] = labels
    self.pie_kw['autopct'] = lambda pct : self.__func_(pct, sizes)
    axis.pie(sizes, **self.pie_kw)
    axis.set_facecolor('white')
    axis.set_title('%s' % title, fontsize=14)
    s = '\n'.join((note,'(n = {:,d})'.format(sum(sizes))))
    axis.text(0.5, -0.02, s, transform=axis.transAxes, fontsize=11, 
              va='top', ha='center', wrap=True, bbox=self.bbx_kw)
  
  def __func_(self, pct, value):
      absolute = int(pct/100.*np.sum(value))
      return "{:.0f}%\n({:,d})".format(pct, absolute)
    
  def __by_variable(self, X, figsize=(12,5.5), fname=None):
    
    '''
    Sort variables by their missingness
    '''
    self.missing_pct_ = self.__missing_pct(X)
    a = self.missing_pct_
    a = a.loc[a['missing_pct']>=self.threshold]
    fig, axis = plt.subplots(1,1,figsize=figsize)
    self.__plot_bar(axis, a['missing_pct']*100, a['features'])
    plt.tight_layout()
    if fname != None: plt.savefig(fname)
    plt.show()

  def __missing_pct(self, X):
    
    a = np.array(X.columns).reshape(-1,1)
    b = sum(np.isnan(X.values).astype(int))/len(X)
    b = np.array(b).reshape(-1,1)
    c = np.concatenate((a,b),axis=1)
    c = pd.DataFrame(c,columns=['features','missing_pct'])
    return c.sort_values(by=['missing_pct'],ascending=False).reset_index(drop=True)
  
  def __by_record(self, X, figsize=(12,4), fname=None):
    
    '''
    Accummulate number of records by number of missingness
    '''
    bins, hist, cum_hist = self.__missing_records(X)
    fig, axis = plt.subplots(1,1,figsize=figsize)
    self.__plot_bar_line(axis, hist, cum_hist, bins)    
    plt.tight_layout()
    if fname != None: plt.savefig(fname)
    plt.show()

  def __missing_records(self, X):
    
    '''
    numpy.histogram --> range: lower <= X < upper
    '''
    a = np.isnan(X.values).astype(int).sum(axis=1)
    bins = np.unique(a)
    hist, _ = np.histogram(a, bins=bins)
    cum_hist = np.cumsum(hist)/len(X)
    return bins[:-1], hist, cum_hist

  def __plot_bar(self, axis, y, X):
    
    # plot missing data point
    axis.bar(X, y, **self.bar_kw)
    axis.set_facecolor('white')
    axis.set_ylabel('Missing data point (%)')
    axis.set_xlabel('Variable')
    axis.set_xticks(np.arange(len(X)))
    axis.set_xticklabels(X, fontsize=10, color='#4b4b4b', rotation=90)
    title = '\n'.join(('Missing data point (by variable)',
                       'Missingness threshold $\geq$ %d%%' % (self.threshold*100))) 
    axis.set_title(title, fontsize=14)
    ylim = axis.get_ylim()
    axis.set_ylim(ylim[0],ylim[1]+0.05)
    axis.set_xlim(-0.5,len(X)-0.5)
    axis.grid(False)

  def __plot_bar_line(self, axis, y1, y2, X):
    
    axis.bar(np.arange(len(X)), y1, **self.bar_kw)
    axis.set_facecolor('white')
    axis.set_ylabel('Number of records')
    axis.set_xlabel('Number of missing data points')
    axis.set_xticks(np.arange(len(X)))
    axis.set_xticklabels(X, fontsize=10, color='#4b4b4b')
    title = '\n'.join(('Number of records (by missingness)',
                       'Record cutoff $\leq$ %d%%' % ((1-self.cutoff)*100))) 
    axis.set_title(title, fontsize=14)
    axis.set_xlim(-0.5,len(X)-0.5)
    axis.grid(False)
    n = sum((y2<=1-self.cutoff).astype(int))-1
    tw_axis = axis.twinx()
    tw_axis.plot(np.arange(len(X)), y2*100, color='#ff793f', lw=1.5, ls='--',label='cum. perentage')
    tw_axis.axvline(n, lw=1, ls='--', color='#84817a',label='cut-off')
    tw_axis.set_ylabel('Cum. percent record (%)')
    tw_axis.legend(loc='center right')
