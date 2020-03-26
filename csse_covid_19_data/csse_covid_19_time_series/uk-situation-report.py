#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().magic('load_ext autoreload')
# get_ipython().magic('autoreload 2')
#
# get_ipython().magic('matplotlib inline')


# In[2]:
import itertools
import numpy as np
import sys
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings('ignore')
from datetime import date
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14, 9)
plt.rcParams['font.size'] = 16
import pandas as pd
import seaborn as sns
# sns.set_style('whitegrid')

import covid19
def pre_process_database(datatemplate, fields):
    dfs = dict()
    for field in fields:
        dfs[field] = pd.read_csv(datatemplate.format(field))

    # print(dfs['Confirmed'].head())

    # loop on the dataframe dictionary
    for field, df in dfs.items():
        # group by country, to sum on states
        df = df.groupby("Country/Region", as_index=False).sum()
        # turn each measurement column into a separate line,
        # and store the results in a new dataframe
        df = df.melt(id_vars=["Country/Region", "Lat", "Long"], value_name="counts")
        # keep track of the quantity that is measured
        # either Confirmed, Deaths, or Recovered
        df["quantity"] = field
        # change column names
        df.columns = ["country", "lat", "lon", "date", "counts", "quantity"]
        # replace the dataframe in the dictionary
        dfs[field] = df

    dfall = pd.concat(dfs.values())
    dfall["date"] = pd.to_datetime(dfall["date"])
    return dfall


def select_database(database, country, field):
    sel = database[(database["country"] == country) & (database["quantity"] == field)]
    return sel, sel["date"], sel["counts"]



# ## data

# In[3]:

today = date.today()
country = "United Kingdom"
# dd/mm/YY
today = today.strftime("%d-%m-%Y")
# datatemplate = "time_series_19-covid-{}.csv"
datatemplate = "time_series_covid19_{}_global.csv"
# fields = ["Confirmed", "Deaths", "Recovered"]
fields = ["confirmed", "deaths", "recovered"]
dataframe_all_countries = pre_process_database(datatemplate, fields)
# data_italy_path = 'dpc-covid19-ita-andamento-nazionale.csv'
# data_italy_path = covid19.data.download('andamento-nazionale')
dataframe, x, y = select_database(
    dataframe_all_countries, country, "confirmed"
)
# dataframe.reset_index()
# # In[4]:
#
#
# data_italy = pd.read_csv(dataframe, parse_dates=['date'], index_col=['date'])
# data_italy.index = data_italy.index.normalize()
data_italy = dataframe.copy()
for column in ['counts']:
    data_italy['variazione_' + column] = data_italy[column].diff(1)

data_italy.tail()

#
# # ## situation report
#
# # In[5]:
#
#
START_FIT = '2020-02-23'
STOP_FIT = '2020-03-25'
EXTRAPOLTATE = ('2020-02-23', '2020-03-26')
#
#
# # In[6]:
#
#
data_italy.set_index('date', inplace=True)
fits = {}
fits['counts'] = covid19.fit.ExponentialFit.from_frame('counts', data_italy, start=START_FIT, stop=STOP_FIT)
ylim_df = data_italy['counts'].iloc[-1]*1.05
#
# # In[7]:
#
#
_, ax = plt.subplots(subplot_kw={'yscale': 'log', 'ylim': (50, ylim_df)})
# # _ = covid19.plot.add_events(ax, linestyle=':', offset=11, color='grey')
#

covid19.plot.plot_fit(ax, fits['counts'], color=sns.color_palette()[2])
for kind, color in zip(['counts'], sns.color_palette()):
    covid19.plot.plot(ax, data_italy[kind], fits[kind], label=kind.replace('_', ' '), extrapolate=EXTRAPOLTATE, color=color, date_interval=3)
#
_ = ax.set(title=r'COVID-19 data in the UK. Modelli $f(t) = 2 ^ \frac{t - t_0}{T_d}$, w $T_d$ doubling time')
_ = ax.yaxis.grid(color='lightgrey', linewidth=0.5)
_ = ax.xaxis.grid(color='lightgrey', linewidth=0.5)
_ = ax.yaxis.tick_right()
_ = ax.legend(loc='upper left')
#
#
# # In[8]:
#
#
_, ax = plt.subplots(subplot_kw={'yscale': 'linear', 'ylim': (80, ylim_df)})
# # _ = covid19.plot.add_events(ax, linestyle=':', offset=17, color='grey')
#

covid19.plot.plot_fit(ax, fits['counts'], color=sns.color_palette()[2])
for kind, color in zip(['counts'], sns.color_palette()):
    covid19.plot.plot(ax, data_italy[kind], fits[kind], label=kind.replace('_', ' '), extrapolate=EXTRAPOLTATE, color=color, date_interval=2)
#
_ = ax.set(title=r'COVID-19 data in the UK. Modelli $f(t) = 2 ^ \frac{t - t_0}{T_d}$, w $T_d$ doubling time')
_ = ax.yaxis.grid(color='lightgrey', linewidth=0.5)
_ = ax.xaxis.grid(color='lightgrey', linewidth=0.5)
_ = ax.yaxis.tick_right()
#
#
# # In[9]:
#
#
_, ax = plt.subplots(subplot_kw={'yscale': 'log', 'ylim': (5, ylim_df)})
kind = 'counts'
covid19.plot.plot(ax, data_italy[kind], fits[kind], label=kind, extrapolate=EXTRAPOLTATE, color=color)
#
#
# # ## estimates
#
# # In[10]:
#
#
kinds = ['counts']
datetime_expected = '2020-03-25'
expected_values = []
for kind in kinds:
    expected_values.append(int(round(fits[kind].predict(datetime_expected))))
print(', '.join(f'{k}: {v}' for v, k in zip(expected_values, kinds)))
#
#
# # In[11]:
#
#
for key, value in list(fits.items()):
    print(f'{key} {" " * (26 - len(key))}{str(value)}')
#
#
# # In[ ]:
#
#
#
#


DAY = np.timedelta64(24 * 60 * 60, 's')

START_FIT = None
CHANGE_FIT_1 = np.datetime64('2020-03-05')
CHANGE_FIT_2 = np.datetime64('2020-03-11')
CHANGE_FIT_3 = np.datetime64('2020-03-15')
STOP_FIT = None

EXTRAPOLTATE = ('2020-02-23', '2020-03-26')

REGIONS_FIT_PARAMS = {
    'UK': {
        'exponential_fits': [('2020-02-24', '2020-03-10'), ('2020-03-13', None)],
        # 'exponential_fits': [(None, '2020-03-11'), ('2020-03-12', None)],
    }
}

DELAY = 12 * DAY
PALETTE_ONE = list(sns.color_palette())
PALETTE = itertools.cycle(PALETTE_ONE)

# In[6]:


fits = {}
for region, params in REGIONS_FIT_PARAMS.items():
    for kind in ['counts']:
        exponential_fits = params.get('exponential_fits',
                                      [(START_FIT, CHANGE_FIT_1), (CHANGE_FIT_1 + DAY, CHANGE_FIT_2),
                                       (CHANGE_FIT_2 + DAY, CHANGE_FIT_3), (CHANGE_FIT_3 + DAY, STOP_FIT)])
        fits[region, kind] = []
        for start, stop in exponential_fits:
            try:

                fits[region, kind] += [
                    covid19.fit.ExponentialFit.from_frame(kind, data_italy, start=start, stop=stop)]
            except:
                print('skipping:', region, start, stop)

# In[7]:
UK_EVENTS = [
{'x': '2020-03-17', 'label': 'first advice to stay home'},
{'x': '2020-03-23', 'label': 'Lockdown in UK'},
]

for region in REGIONS_FIT_PARAMS:
    # select = (data_italy_regions['denominazione_regione'] == region)
    for kind in ['counts']:
        _, ax = plt.subplots(subplot_kw={'yscale': 'log', 'ylim': (9, ylim_df)}, figsize=(14, 8))
        _ = ax.yaxis.grid(color='lightgrey', linewidth=0.5)
        _ = covid19.plot.add_events(ax,events=UK_EVENTS, linestyle=':', offset=0, color='grey')
        if len(fits[region, kind]) == 0:
            print('No data for', region)
            continue
        try:
            for fit, color in zip(fits[region, kind], PALETTE_ONE[1:]):
                covid19.plot.plot_fit(ax, fit, label=kind.split('_')[0].title(), extrapolate=EXTRAPOLTATE, color=color)
            covid19.plot.plot_data(ax, data_italy[kind], label=kind.split('_')[0].title(),
                                   color=PALETTE_ONE[0], date_interval=3)
            ax.set_title(f'{region}')
            _ = ax.yaxis.grid(color='lightgrey', linewidth=0.5)
            _ = ax.yaxis.tick_right()
        except:
            pass

# _ = ax.set(title=r'COVID-19 "severe" cases in Italy. Fit is $f(t) = 2 ^ \frac{t - t_0}{T_d}$, with $T_d$ doubling time and $t_0$ reference date')


# ## Estimate of the initial / uncontined doubling time

# In[8]:


for key, value in list(fits.items()):
    if len(value):
        print(f'{key[0]}:{" " * (14 - len(key[0]))} {str(value[0])}')

# In[9]:


for key, value in list(fits.items()):
    if len(value):
        print(f'{key[0]}:{" " * (14 - len(key[0]))} {str(value[-1])}')

plt.show()