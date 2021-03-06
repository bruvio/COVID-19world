#!/Users/bruvio/python37/bin python
# -*- coding: utf-8 -*-

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




today = date.today()
countrylist = []
countrylist.append("Italy")
# countrylist.append("Australia")
countrylist.append("Germany")
# countrylist.append("China")
# countrylist.append("US")
# countrylist.append("Finland")
countrylist.append("France")
countrylist.append("Spain")
# countrylist.append("Korea, South")
# countrylist.append("Switzerland")
countrylist.append("United Kingdom")
# countrylist.append("Japan")

# dd/mm/YY
today = today.strftime("%d-%m-%Y")
# datatemplate = "time_series_19-covid-{}.csv"
# datatemplate = "./csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{}_global.csv"
# fields = ["Confirmed", "Deaths", "Recovered"]
#
# dataframe_all_countries = pre_process_database(datatemplate, fields)
# # data_italy_path = 'dpc-covid19-ita-andamento-nazionale.csv'
# # data_italy_path = covid19.data.download('andamento-nazionale')
#
# list_of_files = glob.glob('worldmeter_data/*')  # * means all if need specific format then *.csv
# latest_file = max(list_of_files, key=os.path.getctime)
#
# dataframe_all_countries_last_update = pd.read_csv(
#     latest_file)


field = 'Confirmed'
# field = 'Recovered'
# field = 'Deaths'
# fields = []
# field.append('confirmed')
# field.append('deaths')
# field.append('recovered')

DAY = np.timedelta64(24 * 60 * 60, 's')

START_FIT = None
CHANGE_FIT_1 = np.datetime64('2020-03-05')
CHANGE_FIT_2 = np.datetime64('2020-03-11')
CHANGE_FIT_3 = np.datetime64('2020-03-15')
STOP_FIT = None

EXTRAPOLTATE = ('2020-02-23', '2020-04-05')

REGIONS_FIT_PARAMS = {
    'United Kingdom': {
        'exponential_fits': [('2020-02-24', '2020-03-23'), ('2020-03-24', None)],
        # 'exponential_fits': [(None, '2020-03-11'), ('2020-03-12', None)],
    },
    'Spain': {
        'exponential_fits': [('2020-02-24', '2020-03-15'), ('2020-03-15', None)],
        # 'exponential_fits': [(None, '2020-03-11'), ('2020-03-12', None)],
    },
    'France': {
        'exponential_fits': [('2020-02-24', '2020-03-17'), ('2020-03-18', '2020-03-28'), ('2020-03-29', None)],
        # 'exponential_fits': [(None, '2020-03-11'), ('2020-03-12', None)],
    },
    'Germany': {
        'exponential_fits': [('2020-02-24', '2020-03-17'), ('2020-03-13', '2020-03-18'), ('2020-03-19', None)],
        # 'exponential_fits': [(None, '2020-03-11'), ('2020-03-12', None)],
    },
    'Italy': {
        'exponential_fits': [('2020-02-24', '2020-03-10'), ('2020-03-13', '2020-03-17'), ('2020-03-18', '2020-03-27'), ('2020-03-28', None)],
        # 'exponential_fits': [(None, '2020-03-11'), ('2020-03-12', None)],
    }
}

DELAY = 12 * DAY
PALETTE_ONE = list(sns.color_palette())
PALETTE = itertools.cycle(PALETTE_ONE)

for country in countrylist:
    print("\n" + country + "\n")

    data_italy = pd.read_csv('./country_data/{}.csv'.format(country))
    for column in [field]:
        data_italy["variazione_" + column] = data_italy[column].diff(1)


    if country =='United Kingdom':
        START_FIT = "2020-02-23"
        STOP_FIT = "2020-04-12"
        EXTRAPOLTATE = ("2020-02-23", "2020-04-12")
    # else country =='Italy':
    else:
        START_FIT = '2020-02-23'
        STOP_FIT = '2020-04-12'
        EXTRAPOLTATE = ('2020-02-23', '2020-04-12')


    data_italy.set_index("date", inplace=True)
    data_italy.index = pd.to_datetime(data_italy.index)
    fits = {}
    fits[field] = covid19.fit.ExponentialFit.from_frame(
        field, data_italy, start=START_FIT, stop=STOP_FIT
    )

    ylim_df = data_italy[field].iloc[-1] * 1.20

    _, ax = plt.subplots(subplot_kw={"yscale": "log", "ylim": (50, ylim_df)})



    covid19.plot.plot_fit(ax, fits[field], color=sns.color_palette()[2])
    for kind, color in zip([field], sns.color_palette()):
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
    plt.close()

    _, ax = plt.subplots(subplot_kw={"yscale": "linear", "ylim": (80, ylim_df)})
    # # _ = covid19.plot.add_events(ax, linestyle=':', offset=17, color='grey')
    #

    covid19.plot.plot_fit(ax, fits[field], color=sns.color_palette()[2])
    for kind, color in zip([field], sns.color_palette()):
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
    plt.close()

    kinds = [field]
    datetime_expected = '2020-03-31'
    expected_values = []
    for kind in kinds:
        expected_values.append(int(round(fits[kind].predict(datetime_expected))))
    print(", ".join(f"{k}: {v}" for v, k in zip(expected_values, kinds)))

    for key, value in list(fits.items()):
        print(f'{key} {" " * (26 - len(key))}{str(value)}')


    fits = {}
    for region, params in REGIONS_FIT_PARAMS.items():
        if region == country:
            for kind in [field]:
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

    ITALY_EVENTS = [
        # {'x': '2020-02-19', 'label': 'First alarm'},
        {'x': '2020-02-24', 'label': 'Chiusura scuole al nord'},
        {'x': '2020-03-01', 'label': 'Lockdown parziale al nord'},
        {'x': '2020-03-05', 'label': 'Chiusura scuole in Italia'},
        {'x': '2020-03-08', 'label': 'Lockdown al nord'},
        {'x': '2020-03-10', 'label': 'Lockdown in Italia'},
    ]
    country_EVENTS = []

    Spain_EVENTS = [
        {'x': '2020-03-15', 'label': 'Lockdown in Spain'}
    ]
    France_EVENTS = [
        {'x': '2020-03-17', 'label': 'Lockdown in France'}
    ]

    for region in REGIONS_FIT_PARAMS:
        if region == country:
        # select = (data_italy_regions['denominazione_regione'] == region)
            for kind in [field]:
                _, ax = plt.subplots(subplot_kw={'yscale': 'log', 'ylim': (9, ylim_df)}, figsize=(14, 8))
                _ = ax.yaxis.grid(color='lightgrey', linewidth=0.5)
                if region == 'United Kingdom':
                    events = UK_EVENTS
                elif region == 'Italy':
                    events = ITALY_EVENTS
                elif region == 'Spain':
                    events = Spain_EVENTS
                elif region == 'France':
                    events = France_EVENTS
                else:
                    events = country_EVENTS
                _ = covid19.plot.add_events(ax,events=events, linestyle=':', offset=0, color='grey')
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
                    plt.savefig("./Figures/"
                                + region + "COVID-19 Model-change-fit{}.png".format(today), dpi=400)
                except:
                    pass

    for key, value in list(fits.items()):
        if len(value):
            print(f'{key[0]}:{" " * (14 - len(key[0]))} {str(value[0])}')

    # In[9]:


    for key, value in list(fits.items()):
        if len(value):
            print(f'{key[0]}:{" " * (14 - len(key[0]))} {str(value[-1])}')
#
# plt.show()
