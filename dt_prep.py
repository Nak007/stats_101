import pandas as pd, numpy as np, os, calendar
from datetime import datetime

def dt9_to_dt(date):
    
    '''
    Parameters : date : DATE9 i.e. 29FEB2020
    Return : Timestamp('2020-02-29 00:00:00')
    '''
    mth = dict([(m.lower(),str(n)) for (n,m) in enumerate(calendar.month_abbr)])
    dt = date[:2] + mth[date[2:5].lower()] + date[-4:]
    return pd.to_datetime(dt,format='%d%m%Y')

def roomnight(booking, scheme=['2020-07-01','2020-11-01'], return_date=False):
    
    '''
    This function finds roomnight given `booking` and
    `scheme` dates. When `booking` is not overlapping `scheme`,
    it returns array of naughts (`np.zeros`).
    
    Parameters
    ----------
    booking : list of `datetime64`
    \t List of starting and ending of booking dates.
    \t Date format must be '2020-07-01'.
    
    scheme : list of `datetime64`, optional
    \t List of starting and ending dates of the 
    \t program e.g. ['2020-07-01','2020-11-01'].
    
    return_date : `bool`, optional, (default:False)
    \t If `True`, also returns the date-array from beginning
    \t until the end of scheme.
    
    Returns
    -------
    - 1D-Binary array, where `1` represents roomnight, 
      otherwise `0`.
    - The date-array of scheme. Only provided if `return_date` 
      is `True`. 
      
    Examples
    --------
    >>> import numpy as np
    >>> booking = ['2020-10-01','2021-10-05']
    # This also returns date-array.
    >>> roomnight(booking, return_date=True)
    '''
    # Datetime functions.
    date = lambda dt : np.datetime64(dt)
    days = lambda dt : max(((date(dt[1])-date(dt[0]))).astype(int),0)
    period = lambda t : [days(t[n:n+2]) for n in range(3)]
    datetime = lambda t : np.arange(t[0],t[1],dtype='datetime64[D]')
    
    # Booking period must overlap scheme.
    b = [None,None]
    conditon = ((booking[0]>scheme[1]) | (booking[1]<scheme[0]))
    if conditon==False:
        b[0], b[1] = max(booking[0],scheme[0]), min(booking[1],scheme[1])
        timeframe = period(np.array([scheme[0]]+b+[scheme[1]]))
        a = np.hstack([np.zeros(t) if n!=1 else np.ones(t) 
                       for n,t in enumerate(timeframe)])
    else: return np.zeros(days(scheme))
    # Return with/without dates.
    if return_date: return a, datetime(scheme)
    else: return a

def find_files(folder=None, comp=None):
    
    '''
    Find files that are contained within `folder` and 
    sorted by their modified date.
    
    Parameters
    ----------
    folder : `str`, optional, (default:None)
    \t `folder` is a path where searching is executed, 
    \t Its format must follow typical pattern e.g. 
    \t '../example/'. If `None`, `os.getcwd()` is 
    \t deployed instead (current directory).
    
    comp : list of `str`, optional, (default:None)
    \t List of `str` components that are used in 
    \t searching e.g. ['example','.txt']. All elements
    \t within `comp` are joined togather with `OR`
    \t operation.
    
    Returns
    -------
    Array of shape (n_found, [modified_date, file_path])
    \t `n_found` is number of files found.
    
    Examples
    --------
    >>> import os, numpy as np
    # Find "*.txt" within current directory.
    >>> find_files(comp=['.txt'])
    '''
    try:
        # Find file(s) in defined path.
        if folder is None: folder = os.getcwd()
        files = pd.Index(os.listdir(folder))
        join = lambda s : '|'.join(s) if len(s)>1 else s[0]
        if comp is not None: files = files[files.str.contains(join(comp))]
            
        # Find last modified date and sort by such date accordingly.
        date = lambda d : datetime.fromtimestamp(d).strftime('%Y-%m-%d %H:%M:%S')
        files = [(date(os.path.getmtime(folder+f)),folder+f)  for f in files]
        files.sort(reverse=True, key=lambda x : x[0])
        return np.array(files)
    except: return None
