import pandas as pd, numpy as np, calendar

def dt9_to_dt(date):
    
    '''
    Parameters : date : DATE9 i.e. 29FEB2020
    Return : Timestamp('2020-02-29 00:00:00')
    '''
    mth = dict([(m.lower(),str(n)) for (n,m) in enumerate(calendar.month_abbr)])
    dt = date[:2] + mth[date[2:5].lower()] + date[-4:]
    return pd.to_datetime(dt,format='%d%m%Y')
