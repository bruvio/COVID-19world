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
import glob
import os

sys.path.insert(0, "..")
import warnings

warnings.filterwarnings("ignore")
from datetime import date
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (14, 9)
plt.rcParams["font.size"] = 16
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
    # dfall["date"] = pd.to_datetime(dfall["date"], errors='coerce', format='%m/%d/%Y')
    dfall["date"] = pd.to_datetime(dfall["date"])
    return dfall


def select_database(database, country, field):
    sel = database[(database["country"] == country) & (database["quantity"] == field)]
    return sel, sel["date"], sel["counts"]


# ## data

# In[3]:

today = date.today()
countrylist = []
countrylist.append("Italy")
# countrylist.append("Australia")
countrylist.append("Germany")
# countrylist.append("China")
countrylist.append("US")
countrylist.append("Finland")
# countrylist.append("France")
# countrylist.append("Korea, South")
# countrylist.append("Switzerland")
countrylist.append("United Kingdom")
# countrylist.append("Japan")

# dd/mm/YY
today = today.strftime("%d-%m-%Y")
# datatemplate = "time_series_19-covid-{}.csv"
datatemplate = "./csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{}_global.csv"
fields = ["confirmed", "deaths", "recovered"]

dataframe_all_countries = pre_process_database(datatemplate, fields)
# data_italy_path = 'dpc-covid19-ita-andamento-nazionale.csv'
# data_italy_path = covid19.data.download('andamento-nazionale')

list_of_files = glob.glob('worldmeter_data/*')  # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)

dataframe_all_countries_last_update = pd.read_csv(
    latest_file)


field = 'confirmed'
# field = 'recovered'
# field = 'deaths'
# fields = []
# field.append('confirmed')
# field.append('deaths')
# field.append('recovered')

for country in countrylist:
    print("\n" + country + "\n")
    dataframe, x, y = select_database(dataframe_all_countries, country, field)
    # dataframe.reset_index()
    # # In[4]:
    #
    #
    # data_italy = pd.read_csv(dataframe, parse_dates=['date'], index_col=['date'])
    # data_italy.index = data_italy.index.normalize()
    # data_italy = dataframe.copy()

    # for field in fields:
    dataframe_all_countries_last_update1 = dataframe_all_countries_last_update[
        ["Country/Region", field, "Last Update"]]

    dataframe_all_countries_last_update2 = pd.DataFrame(
        {'Country/Region': dataframe_all_countries_last_update1['Country/Region'],
         field: dataframe_all_countries_last_update1[field],
         'date': dataframe_all_countries_last_update1['Last Update'],
         })
    dataframe_all_countries_last_update2.reset_index()
    # df = pd.pivot_table(dataframe_all_countries_last_update2, values = 'confirmed', index=['date'], columns = 'confirmed').reset_index()
    # df = dataframe_all_countries_last_update2.pivot(values = 'confirmed', columns = 'Country/Region')
    df_last_update = dataframe_all_countries_last_update2.pivot(index="date", columns="Country/Region",
                                                                values=field)

    df_country = df_last_update[[country]]

    # dataframe.append(pd.Series(name=df_country.index[0]))
    if field == 'confirmed':
        dataframe = dataframe.append(pd.Series(), ignore_index=True)
        dataframe['date'].iloc[-1] = pd.to_datetime(df_country.index[0])
        dataframe['country'].iloc[-1] = country
        dataframe['quantity'].iloc[-1] = field
        dataframe['counts'].iloc[-1] = df_country[country].values[0]
        x.append(pd.Series(pd.to_datetime(df_country.index[0])))
        y.append(pd.Series(df_country[country].values[0]))
    if field == 'deaths':
        dataframe_deaths = dataframe_deaths.append(pd.Series(), ignore_index=True)
        dataframe_deaths['date'].iloc[-1] = pd.to_datetime(df_country.index[0])
        dataframe_deaths['country'].iloc[-1] = country
        dataframe_deaths['quantity'].iloc[-1] = field
        dataframe_deaths['counts'].iloc[-1] = df_country[country].values[0]
        x_deaths.append(pd.Series(pd.to_datetime(df_country.index[0])))
        y_deaths.append(pd.Series(df_country[country].values[0]))
    if field == 'recovered':
        dataframe_recovered = dataframe_recovered.append(pd.Series(), ignore_index=True)
        dataframe_recovered['date'].iloc[-1] = pd.to_datetime(df_country.index[0])
        dataframe_recovered['country'].iloc[-1] = country
        dataframe_recovered['quantity'].iloc[-1] = field
        dataframe_recovered['counts'].iloc[-1] = df_country[country].values[0]
        x_recovered.append(pd.Series(pd.to_datetime(df_country.index[0])))
        y_recovered.append(pd.Series(df_country[country].values[0]))

    data_italy = dataframe.copy()
    for column in ["counts"]:
        data_italy["variazione_" + column] = data_italy[column].diff(1)

    # data_italy.tail()

    #
    # # ## situation report
    #
    # # In[5]:
    #
    #
    START_FIT = "2020-02-23"
    STOP_FIT = "2020-03-30"
    EXTRAPOLTATE = ("2020-02-23", "2020-04-01")
    #
    #
    # # In[6]:
    #
    #
    data_italy.set_index("date", inplace=True)
    fits = {}
    fits["counts"] = covid19.fit.ExponentialFit.from_frame(
        "counts", data_italy, start=START_FIT, stop=STOP_FIT
    )

    ylim_df = data_italy["counts"].iloc[-1] * 1.20
    #
    # # In[7]:
    #
    #
    _, ax = plt.subplots(subplot_kw={"yscale": "log", "ylim": (50, ylim_df)})
    # # _ = covid19.plot.add_events(ax, linestyle=':', offset=11, color='grey')
    #

    covid19.plot.plot_fit(ax, fits["counts"], color=sns.color_palette()[2])
    for kind, color in zip(["counts"], sns.color_palette()):
        covid19.plot.plot(
            ax,
            data_italy[kind],
            fits[kind],
            label=kind.replace("_", " "),
            extrapolate=EXTRAPOLTATE,
            color=color,
            date_interval=3,
        )
    #
    _ = ax.set(
        title=r"COVID-19 data in "
        + country
        + r". Model $f(t) = 2 ^ \frac{t - t_0}{T_d}$, w $T_d$ doubling time"
    )
    _ = ax.yaxis.grid(color="lightgrey", linewidth=0.5)
    _ = ax.xaxis.grid(color="lightgrey", linewidth=0.5)
    _ = ax.yaxis.tick_right()
    _ = ax.legend(loc="upper left")

    plt.savefig("./Figures/"
                        +country + "COVID-19 Model-log-{}.png".format(today), dpi=400)
    #
    #
    # # In[8]:
    #
    #
    _, ax = plt.subplots(subplot_kw={"yscale": "linear", "ylim": (80, ylim_df)})
    # # _ = covid19.plot.add_events(ax, linestyle=':', offset=17, color='grey')
    #

    covid19.plot.plot_fit(ax, fits["counts"], color=sns.color_palette()[2])
    for kind, color in zip(["counts"], sns.color_palette()):
        covid19.plot.plot(
            ax,
            data_italy[kind],
            fits[kind],
            label=kind.replace("_", " "),
            extrapolate=EXTRAPOLTATE,
            color=color,
            date_interval=2,
        )
    #
    _ = ax.set(
        title=r"COVID-19 data in "
        + country
        + r". Model $f(t) = 2 ^ \frac{t - t_0}{T_d}$, w $T_d$ doubling time"
    )
    _ = ax.yaxis.grid(color="lightgrey", linewidth=0.5)
    _ = ax.xaxis.grid(color="lightgrey", linewidth=0.5)
    _ = ax.yaxis.tick_right()

    plt.savefig("./Figures/"
                        +country + "COVID-19 Model-{}.png".format(today), dpi=400)
    #
    #
    # # In[9]:
    #
    #
    # _, ax = plt.subplots(subplot_kw={'yscale': 'log', 'ylim': (5, 50000)})
    # kind = 'counts'
    # covid19.plot.plot(ax, data_italy[kind], fits[kind], label=kind, extrapolate=EXTRAPOLTATE, color=color)
    # #
    #
    # # ## estimates
    #
    # # In[10]:
    #
    #
    kinds = ["counts"]
    datetime_expected = "2020-04-10"
    expected_values = []
    for kind in kinds:
        expected_values.append(int(round(fits[kind].predict(datetime_expected))))
    print(", ".join(f"{k}: {v}" for v, k in zip(expected_values, kinds)))
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
#
#
# DAY = np.timedelta64(24 * 60 * 60, 's')
#
# START_FIT = None
# CHANGE_FIT_1 = np.datetime64('2020-03-05')
# CHANGE_FIT_2 = np.datetime64('2020-03-11')
# CHANGE_FIT_3 = np.datetime64('2020-03-15')
# STOP_FIT = None
#
# EXTRAPOLTATE = ('2020-02-23', '2020-03-24')
#
# REGIONS_FIT_PARAMS = {
#     'UK': {
#         'exponential_fits': [('2020-02-24', '2020-03-10'), ('2020-03-13', None)],
#         # 'exponential_fits': [(None, '2020-03-11'), ('2020-03-12', None)],
#     }
# }
#
# DELAY = 12 * DAY
# PALETTE_ONE = list(sns.color_palette())
# PALETTE = itertools.cycle(PALETTE_ONE)
#
# # In[6]:
#
#
# fits = {}
# for region, params in REGIONS_FIT_PARAMS.items():
#     for kind in ['counts']:
#         exponential_fits = params.get('exponential_fits',
#                                       [(START_FIT, CHANGE_FIT_1), (CHANGE_FIT_1 + DAY, CHANGE_FIT_2),
#                                        (CHANGE_FIT_2 + DAY, CHANGE_FIT_3), (CHANGE_FIT_3 + DAY, STOP_FIT)])
#         fits[region, kind] = []
#         for start, stop in exponential_fits:
#             try:
#
#                 fits[region, kind] += [
#                     covid19.fit.ExponentialFit.from_frame(kind, data_italy, start=start, stop=stop)]
#             except:
#                 print('skipping:', region, start, stop)
#
# # In[7]:
# UK_EVENTS = [
# {'x': '2020-03-17', 'label': 'first advice to stay home'},
# {'x': '2020-03-23', 'label': 'Lockdown in UK'},
# ]
#
# for region in REGIONS_FIT_PARAMS:
#     # select = (data_italy_regions['denominazione_regione'] == region)
#     for kind in ['counts']:
#         _, ax = plt.subplots(subplot_kw={'yscale': 'log', 'ylim': (9, 15000)}, figsize=(14, 8))
#         _ = ax.yaxis.grid(color='lightgrey', linewidth=0.5)
#         _ = covid19.plot.add_events(ax,events=UK_EVENTS, linestyle=':', offset=0, color='grey')
#         if len(fits[region, kind]) == 0:
#             print('No data for', region)
#             continue
#         try:
#             for fit, color in zip(fits[region, kind], PALETTE_ONE[1:]):
#                 covid19.plot.plot_fit(ax, fit, label=kind.split('_')[0].title(), extrapolate=EXTRAPOLTATE, color=color)
#             covid19.plot.plot_data(ax, data_italy[kind], label=kind.split('_')[0].title(),
#                                    color=PALETTE_ONE[0], date_interval=3)
#             ax.set_title(f'{region}')
#             _ = ax.yaxis.grid(color='lightgrey', linewidth=0.5)
#             _ = ax.yaxis.tick_right()
#         except:
#             pass
#
# # _ = ax.set(title=r'COVID-19 "severe" cases in Italy. Fit is $f(t) = 2 ^ \frac{t - t_0}{T_d}$, with $T_d$ doubling time and $t_0$ reference date')
#
#
# # ## Estimate of the initial / uncontined doubling time
#
# # In[8]:
#
#
# for key, value in list(fits.items()):
#     if len(value):
#         print(f'{key[0]}:{" " * (14 - len(key[0]))} {str(value[0])}')
#
# # In[9]:
#
#
# for key, value in list(fits.items()):
#     if len(value):
#         print(f'{key[0]}:{" " * (14 - len(key[0]))} {str(value[-1])}')
#
plt.show()
