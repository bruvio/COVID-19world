import pandas as pd
from pandas.plotting import register_matplotlib_converters
import subprocess

register_matplotlib_converters()
import logging
import os
import glob
import plotly.graph_objects as go
import plotly
import warnings
import datetime
import pdb
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import matplotlib.colors as mc
import colorsys
from random import randint
import re
from datetime import date



# from IPython.display import HTML
label_size = 8
mpl.rcParams["xtick.labelsize"] = label_size
from scipy.optimize import curve_fit
import pycountry as pc

# do not print unnecessary warnings during curve_fit()
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
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


def get_times(dataframe, y, prediction_days):

    date = dataframe.index.values

    start = dataframe["date"].iloc[0]

    x = np.arange(0, len(y))

    end = dataframe["date"].iloc[-1]

    real_days = len(y)
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    t_real_asnumber = np.linspace(start.value, end.value, real_days)
    t_real = np.asarray(pd.to_datetime(t_real_asnumber))

    prediction = start + datetime.timedelta(days=prediction_days)

    # end = pd.Timestamp('2020-03-31')
    start = pd.Timestamp(start)
    prediction = pd.Timestamp(prediction)
    t_prediction_asnumber = np.linspace(start.value, prediction.value, prediction_days)
    t_prediction = np.asarray(pd.to_datetime(t_prediction_asnumber))
    t_prediction_asnumber_plot = np.linspace(
        start.value, prediction.value, int(prediction_days / 3)
    )
    t_plot = np.asarray(pd.to_datetime(t_prediction_asnumber_plot))

    days = len(t_prediction)

    start_prediction_date = date
    stop_prediction_date = days
    date_prediction = np.arange(0, stop_prediction_date)

    return t_real, t_prediction, x, start, prediction, days, t_plot


today = date.today()

# dd/mm/YY
today = today.strftime("%d-%m-%Y")
# datatemplate = "time_series_19-covid-{}.csv"
datatemplate = "./csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{}_global.csv"
# fields = ["confirmed", "deaths", "recovered"]
fields = []
fields.append('Confirmed')
fields.append('Deaths')
fields.append('Recovered')

dataframe_all_countries = pre_process_database(datatemplate, fields)
list_of_files = glob.glob('worldmeter_data/*')  # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)

dataframe_all_countries_last_update = pd.read_csv(
    latest_file)

countrylist_df = list(set(dataframe_all_countries["country"]))
countrylist = []
# countrylist.append("Italy")
# countrylist.append("Australia")
# countrylist.append("Germany")
# countrylist.append("China")
# countrylist.append("Australia")
# countrylist.append("US")
# countrylist.append("France")
# countrylist.append("Spain")
# countrylist.append("Korea, South")
# countrylist.append("Switzerland")
# countrylist.append("United Kingdom")
# countrylist.append("Japan")
# countrylist.append("Romania")

for country in countrylist_df:
            if country in countrylist_df:
                try:
                    print(country)
                    databasename = "Confirmed cases"
                    # databasename = "Recovered cases"
                    # databasename = "Deaths cases"





                    dataframe, x, y = select_database(
                        dataframe_all_countries, country, "Confirmed"
                    )
                    dataframe_deaths, x_deaths, y_deaths = select_database(
                        dataframe_all_countries, country, "Deaths"
                    )
                    dataframe_recovered, x_recovered, y_recovered = select_database(
                        dataframe_all_countries, country, "Recovered"
                    )
                    #updating JH data base with latest daily data scraped from web
                    for field in fields:
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
                        df_last_update = dataframe_all_countries_last_update2.pivot(index="date", columns="Country/Region", values=field)

                        df_country = df_last_update[[country]]





                        # dataframe.append(pd.Series(name=df_country.index[0]))
                        if field =='Confirmed':
                            dataframe = dataframe.append(pd.Series(), ignore_index=True)
                            dataframe['date'].iloc[-1] = pd.to_datetime(df_country.index[0])
                            dataframe['country'].iloc[-1] = country
                            dataframe['quantity'].iloc[-1] = 'Confirmed'
                            dataframe['counts'].iloc[-1] = df_country[country].values[0]
                            x.append(pd.Series(pd.to_datetime(df_country.index[0])))
                            y.append(pd.Series(df_country[country].values[0]))
                        if field == 'Deaths':
                            dataframe_deaths = dataframe_deaths.append(pd.Series(), ignore_index=True)
                            dataframe_deaths['date'].iloc[-1] = pd.to_datetime(df_country.index[0])
                            dataframe_deaths['country'].iloc[-1] = country
                            dataframe_deaths['quantity'].iloc[-1] = 'Deaths'
                            dataframe_deaths['counts'].iloc[-1] = df_country[country].values[0]
                            x_deaths.append(pd.Series(pd.to_datetime(df_country.index[0])))
                            y_deaths.append(pd.Series(df_country[country].values[0]))
                        if field == 'Recovered':
                            dataframe_recovered = dataframe_recovered.append(pd.Series(), ignore_index=True)
                            dataframe_recovered['date'].iloc[-1] = pd.to_datetime(df_country.index[0])
                            dataframe_recovered['country'].iloc[-1] = country
                            dataframe_recovered['quantity'].iloc[-1] = 'Recovered'
                            dataframe_recovered['counts'].iloc[-1] = df_country[country].values[0]
                            x_recovered.append(pd.Series(pd.to_datetime(df_country.index[0])))
                            y_recovered.append(pd.Series(df_country[country].values[0]))


                    country_df = pd.DataFrame({'date': dataframe['date'],
                                            'Confirmed': dataframe['counts'],
                                            'Deaths': dataframe_deaths['counts'],
                                            'Recovered': dataframe_recovered['counts'],
                                            })
                    country_df.to_csv('./country_data/{}.csv'.format(country), index=False)

                except:
                    print('failed to run for {}'.format(country))