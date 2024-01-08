import pandas as pd
import numpy as np
import time
def fact(n):
    if n==0 or n==1:
        return 1
    return n*fact(n-1)
temperatures = [5, 2, 3, 4, 4, 1]
ser1 = pd.Series(data = temperatures)
def f(n):
    if n>=ser1.agg('mean'):
        return 'High'
    else:
        return 'Low'
# ser1.apply(f)
# ser1.select([ser1 > serq.agg('mean'), ser1 < ser1.agg('mean')], ['High', 'Low'])
ser1.clip(lower=ser1.quantile(0.1), upper=ser1.quantile(0.9))
