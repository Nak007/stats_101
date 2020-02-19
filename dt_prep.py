import pandas as pd, numpy as np, calendar

mth = dict([(m.lower(),str(n)) for (n,m) in enumerate(calendar.month_abbr)])

def dt9_to_dt(date):

  dt = date[:2] + mth[date[2:5].lower()] + date[-4:]
  return pd.to_datetime(dt,format='%d%m%Y')
