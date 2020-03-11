import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

# mondiale_df = pd.read_csv("who_covid_19_sit_rep_time_series.csv",parse_dates=['data'],
#                                 index_col=['data'])
mondiale_df = pd.read_csv("who_covid_19_sit_rep_time_series.csv")

print(mondiale_df.head(5))
print(mondiale_df.columns)


totale_casi = np.array(mondiale_df['totale_casi'])
casi_attivi = totale_casi - mondiale_df['dimessi_guariti']

data =  mondiale_df.index.values
# convert date to unix time for fit use
dates = pd.to_numeric(data)

# totale_casi_fit = np.polyfit(dates, np.log(totale_casi), 1)





def exp_func(x, a, b ):
    return a*np.exp(b*x)
def log_func(t,a,b):
    return a+b*np.log(t)

# using an array of integer to calculate fit to avoid overflow
x = np.arange(0,len(totale_casi))

popt1, pcov1 = curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  totale_casi)
popt2, pcov2 = curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  casi_attivi)



plt.figure()
plt.plot(dates, totale_casi, 'ko', label="Original Data")
plt.plot(dates, exp_func(x, *popt1), 'r-', label="Fitted Curve")


plt.legend()
# plt.show()

plt.figure()
plt.plot(mondiale_df.index.values,mondiale_df['totale_casi'],label='totale casi',marker='x')
plt.plot(mondiale_df.index.values,casi_attivi,label='casi attivi')
plt.plot(mondiale_df.index.values, exp_func(x, *popt1), 'r-', label="Fitted Curve - casi totali")
plt.plot(mondiale_df.index.values, exp_func(x, *popt2), 'r-', label="Fitted Curve - casi attivi")

plt.plot(mondiale_df.index.values,mondiale_df['isolamento_domiciliare'],label='isolamento_domiciliare',marker='>')
plt.plot(mondiale_df.index.values,mondiale_df['deceduti'],label='deceduti',marker='o')
plt.plot(mondiale_df.index.values,mondiale_df['dimessi_guariti'],label='dimessi_guariti',marker='s')
# plt.plot(date, totale_casi_fit,label='casi totali - fit', marker='^')
plt.xticks(rotation=15, ha="right")
plt.legend(loc='best')
plt.show()

# print(popt,pcov)