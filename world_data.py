#!/Users/bruvio/python37/bin python
# -*- coding: utf-8 -*-

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


def Insert_row_(row_number, df, row_value):
    # Slice the upper half of the dataframe
    df1 = df[0:row_number]

    # Store the result of lower half of the dataframe
    df2 = df[row_number:]

    # Inser the row in the upper half dataframe
    df1.loc[row_number] = row_value

    # Concat the two dataframes
    df_result = pd.concat([df1, df2])

    # Reassign the index labels
    df_result.index = [*range(df_result.shape[0])]

    # Return the updated dataframe
    return df_result


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


def derivative(f, a, method="central", h=0.01):
    """Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h
    """
    if method == "central":
        return (f(a + h) - f(a - h)) / (2 * h)
    elif method == "forward":
        return (f(a + h) - f(a)) / h
    elif method == "backward":
        return (f(a) - f(a - h)) / h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


def fourPL(x, A, B, C, D):
    return (A - D) / (1.0 + ((x / C) ** (B))) + D


def exp_func(x, a, b):
    return a * np.exp(b * x)


def expo_func(x, a, b):
    return np.exp(a * (x - b))


def exp_func1(x, b):
    return np.exp(b * x)


def Hill_sigmoidal_func(x, a, b, c):  # Hill sigmoidal equation from zunzun.com
    return a * np.power(x, b) / (np.power(c, b) + np.power(x, b))


def func1(x, a, b, c, d):
    return d + ((a - d) / (1 + (x / c) ** b))


def func2(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))


def sigmoidal_func(x, a, b, c):
    return a / (1 + np.exp(-c * (x - b)))


def fit_data(x, y, func, p0=None):
    # these are the same as the scipy defaults
    # initialParameters = np.array([1.0, 1.0, 1.0])

    # curve fit the test data
    # fittedParameters, pcov = curve_fit(func, x, totale_casi, initialParameters, maxfev=5000)
    try:
        if p0:
            fittedParameters, pcov = curve_fit(func, x, y, maxfev=5000, p0=p0)
        else:
            fittedParameters, pcov = curve_fit(func, x, y, maxfev=5000)
        modelPredictions = func(y, *fittedParameters)

        absError = modelPredictions - y

        SE = np.square(absError)  # squared errors
        MSE = np.mean(SE)  # mean squared errors
        RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (np.var(absError) / np.var(y))

        print("Parameters:", fittedParameters)
        print("RMSE:", RMSE)
        print("R-squared:", Rsquared)

        # create data for the fitted equation plot
        # xModel = np.arange(min(x), max(x))
        xModel = x
        yModel = func(xModel, *fittedParameters)

        # plt.legend(loc='best')
        # plt.xticks(rotation=15, ha="right")

        return xModel, yModel, fittedParameters, Rsquared
    except:
        print("failed to fit data")
        return [], [], [], []


def plot_data(Xdata, Ydata, country, label, color, logscale=False):

    # first the raw data as a scatter plot
    if logscale:
        plt.yscale("log")
        # plt.ylim(1, 1e4)
    plt.scatter(
        Xdata,
        Ydata,
        marker="D",
        label="Original Data - " + country + " - " + label,
        c=color,
    )


def plot_model(
    xModel, yModel, country, label, color, marker, logscale=False, linewidths=None
):

    if logscale:
        plt.yscale("log")

        # plt.ylim(1, 1e4)
    plt.scatter(
        xModel,
        yModel,
        label="Fitted Data - " + country + " - " + label,
        c=color,
        marker=marker,
        linewidths=linewidths,
    )

    # plt.legend(loc='best')
    # plt.xticks(rotation=15, ha="right")


def plot(dataframe, countries, xrange, dtype="Confirmed", yrange=None, yscale="linear"):
    """plot the covid-19 data with an exponential fit.
    - countries: list of countries
    - xrange: fit range, e.g. (30,55)
    - yscale: log or linear
    - yrange: well, y range, useful in log mode.
    """

    xmin, xmax = xrange
    linx = np.linspace(xmin, xmax, 101)
    colors = ["blue", "red", "orange", "green"]
    for i, country in enumerate(countries):
        color = colors[i]
        sel = dataframe[
            (dataframe["country"] == country) & (dataframe["quantity"] == dtype)
        ]
        yp = sel["counts"][xmin : xmax + 1]
        xp = np.arange(len(yp)) + xmin

        plt.scatter(xp, yp, label=country, alpha=0.7, marker=".", color=color)
        pars, cov = curve_fit(expo_func, xp, yp)
        f = expo_func(linx, *pars)
        plt.plot(linx, f, color=color, alpha=0.3)
    plt.legend(loc="upper left")
    plt.xlabel("days since beginning of the year")
    plt.ylabel("confirmed cases")
    plt.yscale(yscale)
    if yrange:
        plt.ylim(*yrange)
    plt.grid()



import random

def hex_code_colors():
    a = hex(random.randrange(0,256))
    b = hex(random.randrange(0,256))
    c = hex(random.randrange(0,256))
    a = a[2:]
    b = b[2:]
    c = c[2:]
    if len(a)<2:
        a = "0" + a
    if len(b)<2:
        b = "0" + b
    if len(c)<2:
        c = "0" + c
    z = a + b + c
    return "#" + z.upper()


def transform_color(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def main(plot_fits, plot_bar_plot, plot_bar_plot_video):
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
    countrylist.append("Italy")
    countrylist.append("Australia")
    countrylist.append("Germany")
    countrylist.append("China")
    countrylist.append("US")
    countrylist.append("France")
    countrylist.append("Spain")
    # countrylist.append("Korea, South")
    countrylist.append("Switzerland")
    countrylist.append("United Kingdom")
    countrylist.append("Japan")
    # countrylist.append("Romania")

    exception_list = []
    exception_list.append("Australia")
    exception_list.append("China")
    exception_list.append("Australia")
    # exception_list.append("Italy")
    exception_list.append("US")
    exception_list.append("France")
    exception_list.append("Switzerland")
    # exception_list.append("United Kingdom")
    exception_list.append("Japan")
    # logscale= True
    logscale = False
    if plot_fits:
        for country in countrylist:
            if country in countrylist_df:
                try:
                    print(country)
                    databasename = "Confirmed cases"
                    # databasename = "Recovered cases"
                    # databasename = "Deaths cases"


                    dataframe = pd.read_csv('./country_data/{}.csv'.format(country))
                    y = dataframe['Confirmed']




                    # dataframe,x,y = dataframe_deaths, x_deaths, y_deaths
                    prediction_dates = 96
                    day_to_use_4_fit = 4
                    t_real, t_prediction, x, start, prediction, days, t_plot = get_times(
                        dataframe, y, prediction_dates
                    )

                    # FITTING THE DATA ###########################
                    if country == "Italy":
                        xModel_fit, yModel_fit, fittedParameters, Rsquared = fit_data(
                            x, y, sigmoidal_func
                        )
                        text_fit = "${} / (1 + exp^{{(-{} * (x - {}))}})$".format(
                            float("{0:.2f}".format(fittedParameters[0])),
                            float("{0:.2f}".format(fittedParameters[1])),
                            float("{0:.2f}".format(fittedParameters[2])),
                        )
                    elif country == "United Kingdom":
                        xModel_fit, yModel_fit, fittedParameters, Rsquared = fit_data(
                            x, y, expo_func
                        )
                        text_fit = "$ exp^{{ {} \cdot x}}$".format(
                            float("{0:.2f}".format(fittedParameters[-1]))
                        )
                    else:
                        xModel_fit, yModel_fit, fittedParameters, Rsquared = fit_data(
                            x, y, exp_func1
                        )
                        text_fit = "$ exp^{{ {} \cdot x}}$".format(
                            float("{0:.2f}".format(fittedParameters[-1]))
                        )
                    xModel_date = xModel_fit.astype(datetime.datetime)

                    ###########################



                    ################ FITTING LAST 10 DAYS OF THE DATA ###########################

                    if country == "Italy":
                        print("\n last {} days fit data\n".format(day_to_use_4_fit))
                        xModel, yModel, fittedParameters_10, Rsquared = fit_data(
                            x[-day_to_use_4_fit:], y[-day_to_use_4_fit:], exp_func1
                        )
                        text_10days_fit = "$ exp^{{ {} \cdot x}}$".format(
                            float("{0:.3}".format(fittedParameters_10[-1]))
                        )
                    elif country == "US":
                        print("\n last {} days fit data\n".format(day_to_use_4_fit))
                        xModel, yModel, fittedParameters_10, Rsquared = fit_data(
                            x[-day_to_use_4_fit:], y[-day_to_use_4_fit:], exp_func1
                        )
                        text_10days_fit = "$ exp^{{ {} \cdot x}}$".format(
                            float("{0:.3}".format(fittedParameters_10[-1]))
                        )
                    else:
                        print("\n last {} days fit data\n".format(day_to_use_4_fit))
                        xModel, yModel, fittedParameters_10, Rsquared = fit_data(
                            x[-day_to_use_4_fit:], y[-day_to_use_4_fit:], exp_func1
                        )
                        text_10days_fit = "$ exp^{{ {} \cdot x}}$".format(
                            float("{0:.3f}".format(fittedParameters_10[-1]))
                        )

                    ################ PLOTTING DATA & FIT ###############
                    plt.figure(num=country + "_fit")
                    # ax1 = plt.subplot(211)

                    plot_data(t_real, y, country, "confirmed cases", "r", logscale=logscale)

                    plot_model(
                        t_real,
                        yModel_fit,
                        country,
                        " data fit - " + databasename + "  -  " + text_fit,
                        "b",
                        marker="x",
                        logscale=logscale,
                    )
                    plt.savefig(
                        "./Figures/"
                        + country
                        + "_"
                        + databasename
                        + "_fitted_log-{}.png".format(today),
                        dpi=100,
                    )
                    # plt.ylim(1, 1e5)
                    plt.xticks(rotation=15, ha="right")
                    # figure  = plt.figure(num=country,figsize=(11, 8))
                    figure = plt.figure(num=country, figsize=(11, 8))
                    ax1 = plt.subplot(211)

                    plot_data(t_real, y, country, databasename, "r", logscale=logscale)

                    # plot_model(
                    #     t_real,
                    #     yModel_fit,
                    #     country,
                    #     " data fit - " + databasename + "  -  " + text_fit,
                    #     "b",
                    #     marker="x",
                    #     logscale=logscale,
                    # )

                    plot_model(
                        t_real[-day_to_use_4_fit:],
                        yModel[-day_to_use_4_fit:],
                        country,
                        " 11days-fit - " + databasename + "  -  " + text_10days_fit,
                        "g",
                        marker="x",
                        logscale=logscale,
                    )

                    ax1.set_ylim([min(y), max(y)*1.05])
                    # plt.show()
                    ###########################

                    ################ PREDICTION ###########################
                    print("prediction from {} to {}".format(start, prediction))
                    #

                    day1 = dataframe["date"].iloc[0]

                    day_of_the_year_start = datetime.datetime.strptime(day1, '%Y-%m-%d %H:%M:%S').day
                    # x = np.arange(day_of_the_year_start, len(y) + day_of_the_year_start)
                    x_prediction = np.arange(0, len(y) + day_of_the_year_start)
                    if country in exception_list:
                        coefs = np.poly1d(np.polyfit(x, y, 5))
                        modelPredictions = np.polyval(coefs, x_prediction)
                        absError = modelPredictions[0 : len(y)] - y
                        #
                        # print('\nsigmoidal fit data\n')
                        SE = np.square(absError)  # squared errors
                        MSE = np.mean(SE)  # mean squared errors
                        RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
                        Rsquared = 1.0 - (np.var(absError) / np.var(y))
                        #
                        # print("Parameters:", fittedParameters_prediction)
                        print("RMSE:", RMSE)
                        print("R-squared:", Rsquared)

                    else:
                        (
                            xModel_Predictions,
                            yModel_Predictions,
                            fittedParameters_prediction,
                            Rsquared,
                        ) = fit_data(x, y, sigmoidal_func, p0=[1, fittedParameters_10[-1], 1])

                        modelPredictions = sigmoidal_func(
                            x_prediction, *fittedParameters_prediction
                        )
                        absError = modelPredictions[0 : len(y)] - y
                        #
                        print("\nsigmoidal fit data\n")
                        SE = np.square(absError)  # squared errors
                        MSE = np.mean(SE)  # mean squared errors
                        RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
                        Rsquared = 1.0 - (np.var(absError) / np.var(y))
                        #
                        print("Parameters:", fittedParameters_prediction)
                        print("RMSE:", RMSE)
                        print("R-squared:", Rsquared)
                    #


                    xModel_prediction = np.arange(0, days)
                    if country in exception_list:
                        yModel_prediction = np.polyval(coefs, xModel_prediction)
                        f = np.poly1d(coefs)
                        text = str(f)
                    else:

                        yModel_prediction = sigmoidal_func(
                            xModel_prediction, *fittedParameters_prediction
                        )
                        text = "${} / (1 + exp^{{(-{} * (x - {}))}})$".format(
                            float("{0:.2f}".format(fittedParameters_prediction[0])),
                            float("{0:.2f}".format(fittedParameters_prediction[1])),
                            float("{0:.2f}".format(fittedParameters_prediction[2])),
                        )

                    plot_model(
                        t_prediction,
                        yModel_prediction,
                        country,
                        "predictions - " + databasename + "  -  " + text,
                        "m",
                        marker=".",
                        logscale=logscale,
                    )
                    # plt.ylim(1, 1e4)
                    plt.xticks(rotation=15, ha="right")

                    plt.xlabel("days since it started")  # X axis data label
                    plt.ylabel(databasename)  # Y axis data label


                    ###########################
                    ################ DAILY PREDICTION ###########################
                    daily = np.diff(y)
                    a = 0
                    daily = np.concatenate([[a], daily])
                    # yy = sigmoidal_func(x, *fittedParameters_prediction)

                    if country in exception_list:
                        f = np.poly1d(coefs)
                    else:
                        a, b, c = fittedParameters_prediction
                        f = lambda x: a / (1 + np.exp(-c * (x - b)))
                    # yy = f(x)
                    ydx = derivative(f, xModel_prediction, method="forward", h=0.001)
                    # ydx1 = derivative(f, xModel_prediction, method="backward", h=0.01)
                    # ydx2 = derivative(f, xModel_prediction, method="central", h=0.01)

                    ax2 = plt.subplot(212, sharex=ax1)

                    plot_model(
                        t_prediction,
                        ydx,
                        country,
                        "daily predictions - " + databasename,
                        "b",
                        marker="o",
                        logscale=logscale,
                    )

                    plot_data(
                        t_real,
                        daily,
                        country,
                        "daily " + databasename,
                        "r",
                        logscale=logscale,
                    )
                    # plt.ylim(1, 1e4)
                    # plt.legend(loc="best", fontsize="6")
                    # plt.legend(loc=9, bbox_to_anchor=(0.5,-0.02), fontsize=8)
                    plt.xticks(t_plot, rotation=15, ha="right")

                    ax1.legend(
                        bbox_to_anchor=(0.6, 0.8),
                        fancybox=True,
                        shadow=True,
                        ncol=1,
                        fontsize=8,
                    )
                    ax2.legend(
                        bbox_to_anchor=(0.4, 0.8),
                        fancybox=True,
                        shadow=True,
                        ncol=1,
                        fontsize=8,
                    )
                    figure.tight_layout()
                    if logscale:
                        plt.savefig(
                            "./Figures/"
                            + country
                            + "_"
                            + databasename
                            + "_fitted_log-{}.png".format(today),
                            dpi=100,
                        )
                        plt.ylim(1, 1e5)
                    else:
                        plt.savefig(
                            "./Figures/"
                            + country
                            + "_"
                            + databasename
                            + "_fitted-{}.png".format(today),
                            dpi=100,
                        )



                    # fig_rate = go.Figure()
                    # tickList = list(
                    #     np.arange(
                    #         0,
                    #         (dataframe["Deaths"] / dataframe["Confirmed"] * 100).max()
                    #         + 0.2,
                    #         0.5,
                    #     )
                    # )
                    # dataframe['date'] = pd.to_datetime(dataframe['date'], infer_datetime_format=True)
                    # fig_rate.add_trace(
                    #     go.Scatter(
                    #         x=dataframe["date"],
                    #         y=dataframe["Deaths"]  / dataframe["Confirmed"] * 100,
                    #         mode="lines+markers",
                    #         line_shape="spline",
                    #         name=country,
                    #         line=dict(color="#626262", width=4),
                    #         marker=dict(
                    #             size=4, color="#f4f4f2", line=dict(width=1, color="#626262")
                    #         ),
                    #         text=[
                    #             datetime.datetime.strftime(d, "%b %d %Y AEDT")
                    #             for d in dataframe["date"]
                    #         ],
                    #         hovertext=[
                    #             country + " death rate (%) <br>{:.2f}%".format(i)
                    #             for i in dataframe["Deaths"]
                    #             / dataframe["Confirmed"]
                    #             * 100
                    #         ],
                    #         hovertemplate="<b>%{text}</b><br></br>"
                    #         + "%{hovertext}"
                    #         + "<extra></extra>",
                    #     )
                    # )
                    #
                    # fig_rate.update_layout(
                    #     # title=country + " death rate (%)",
                    #     xaxis_title="days",
                    #     yaxis_title=country + " death rate (%)",
                    #     margin=go.layout.Margin(l=10, r=10, b=10, t=5, pad=0),
                    #     yaxis=dict(
                    #         showline=False,
                    #         linecolor="#272e3e",
                    #         zeroline=False,
                    #         # showgrid=False,
                    #         gridcolor="rgba(203, 210, 211,.3)",
                    #         gridwidth=0.1,
                    #         tickmode="array",
                    #         # Set tick range based on the maximum number
                    #         tickvals=tickList,
                    #         # Set tick label accordingly
                    #         ticktext=["{:.1f}".format(i) for i in tickList],
                    #     ),
                    #     #    yaxis_title="Total Confirmed Case Number",
                    #     xaxis=dict(
                    #         showline=False,
                    #         linecolor="#272e3e",
                    #         showgrid=False,
                    #         gridcolor="rgba(203, 210, 211,.3)",
                    #         gridwidth=0.1,
                    #         zeroline=False,
                    #     ),
                    #     xaxis_tickformat="%b %d",
                    #     hovermode="x",
                    #     legend_orientation="h",
                    #     # legend=dict(x=.02, y=.95, bgcolor="rgba(0,0,0,0)",),
                    #     plot_bgcolor="#f4f4f2",
                    #     paper_bgcolor="#cbd2d3",
                    #     font=dict(color="#292929"),
                    # )
                    # # fig_rate.write_image("Figures/death_rates_" + country )
                    # plotly.offline.plot(
                    #     fig_rate,
                    #     filename="Figures/death_rates_" + country ,
                    #     auto_open=False,
                    # )

                    y1 = 100 * (1.10) ** (x - 1)  # 15% growth rate
                    y2 = 100 * (1.08) ** (x - 1)  # 8% growth rate
                    y3 = 100 * (1.07) ** (x - 1)  # 7% growth rate
                    y4 = 100 * (1.05) ** (x - 1)  # 5% growth rate

                    plt.figure(num=country + "growth-rate")
                    # ax1 = plt.subplot(211)

                    plot_data(t_real, y, country, "confirmed cases", "r", logscale=logscale)

                    plot_model(
                        t_real,
                        yModel_fit,
                        country,
                        " data fit - " + databasename + "  -  " + text_fit,
                        "b",
                        marker="x",
                        logscale=logscale,
                    )

                    plot_model(
                        t_real,
                        y1,
                        country,
                        " 10% growth rate - " + databasename,
                        "b",
                        marker="^",
                        logscale=logscale,
                        linewidths=0.1,
                    )

                    plt.ylim([min(y), max(y)*1.05])
                    plt.legend(loc="best", fontsize=8)


                except:
                    print('failed to run for {}'.format(country))
    # plt.show()
    if plot_bar_plot:

        fields = []
        fields.append("Confirmed")
        fields.append("Deaths")
        fields.append("Recovered")



        for field in fields:
            # df = pd.read_csv(datatemplate.format(field))
            list_of_files = glob.glob('cumulative_data/*')  # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            date_suffix = latest_file[-14:-4]
            df = pd.read_csv(
                './cumulative_data/cumulative_{}-{}.csv'.format(field,date_suffix))

            for p in range(3):
                i = 0
                while i < len(df.columns):
                    try:
                        a = np.array(df.iloc[:, i + 1])
                        b = np.array(df.iloc[:, i + 2])
                        c = (a + b) / 2
                        df.insert(
                            i + 2,
                            str(df.iloc[:, i + 1].name) + "^" + str(len(df.columns)),
                            c,
                        )
                    except:
                        print(f"\n  Interpolation No. {p + 1} done...")
                    i += 2

            df = pd.melt(df, id_vars="country", var_name="date")

            df.rename(columns={"value": "counts"}, inplace=True)
            # frames_list = df["date"].unique()
            df.to_csv('test_database_after_interp.csv')

            frames_list = df["date"].unique().tolist()
            # for i in range(10):
            #     frames_list.append(df["date"].iloc[-1])

            all_names = df["country"].unique().tolist()
            random_hex_colors = []
            for i in range(len(all_names)):
                # random_hex_colors.append("#" + "%06X" % randint(0, 0xFFFFFF))
                random_hex_colors.append(hex_code_colors())

            rgb_colors = [transform_color(i, 1) for i in random_hex_colors]
            rgb_colors_opacity = [
                rgb_colors[x] + (0.825,) for x in range(len(rgb_colors))
            ]
            rgb_colors_dark = [transform_color(i, 1.12) for i in random_hex_colors]

            fig, ax = plt.subplots(figsize=(11, 8))
            # fig, ax = plt.subplots()

            num_of_elements = 10

            def draw_barchart(Time):
                df_frame = (
                    df[df["date"].eq(Time)]
                    .sort_values(by="counts", ascending=True)
                    .tail(num_of_elements)
                )
                ax.clear()

                normal_colors = dict(zip(df["country"].unique(), rgb_colors_opacity))
                dark_colors = dict(zip(df["country"].unique(), rgb_colors_dark))

                ax.barh(
                    df_frame["country"],
                    df_frame["counts"],
                    color=[normal_colors[x] for x in df_frame["country"]],
                    height=0.8,
                    edgecolor=([dark_colors[x] for x in df_frame["country"]]),
                    linewidth="6",
                )

                dx = float(df_frame["counts"].max()) / 200

                for i, (value, name) in enumerate(
                    zip(df_frame["counts"], df_frame["country"])
                ):
                    ax.text(
                        value + dx,
                        i + (num_of_elements / 50),
                        "    " + name,
                        size=14,
                        weight="bold",
                        ha="left",
                        va="center",
                        fontdict={"fontname": "Trebuchet MS"},
                    )
                    ax.text(
                        value + dx * 10,
                        i - (num_of_elements / 50),
                        f"    {value:,.0f}",
                        size=14,
                        ha="left",
                        va="center",
                    )

                time_unit_displayed = re.sub(r"\^(.*)", r"", str(Time))
                ax.text(
                    1.0,
                    1.14,
                    time_unit_displayed,
                    transform=ax.transAxes,
                    color="#666666",
                    size=14,
                    ha="right",
                    weight="bold",
                    fontdict={"fontname": "Trebuchet MS"},
                )
                # ax.text(-0.005, 1.06, 'Number of confirmed cases', transform=ax.transAxes, size=14, color='#666666')
                ax.text(
                    -0.005,
                    1.14,
                    "Number of {} cases ".format(field),
                    transform=ax.transAxes,
                    size=14,
                    weight="bold",
                    ha="left",
                    fontdict={"fontname": "Trebuchet MS"},
                )

                ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
                ax.xaxis.set_ticks_position("top")
                ax.tick_params(axis="x", colors="#666666", labelsize=12)
                ax.set_yticks([])
                ax.set_axisbelow(True)
                ax.margins(0, 0.01)
                ax.grid(which="major", axis="x", linestyle="-")

                plt.locator_params(axis="x", nbins=4)
                plt.box(False)
                plt.subplots_adjust(
                    left=0.075,
                    right=0.75,
                    top=0.825,
                    bottom=0.05,
                    wspace=0.2,
                    hspace=0.2,
                )

            draw_barchart(frames_list[-1])
            plt.savefig(
                "./Figures/Racing Bar Chart-{}-{}.png".format(field, frames_list[-1]),
                dpi=100,
            )
            # plt.savefig(
            #     "./Figures/Racing Bar Chart-{}.png".format(field),
            #     dpi=100,
            # )
            # plt.show()
            if plot_bar_plot_video:
                animator = animation.FuncAnimation(fig, draw_barchart, frames=frames_list)
                animator.save(
                    "./Figures/Racing Bar Chart-{}-{}.mp4".format(field, today),
                    fps=30,
                    dpi=100,
                )

            # animator.save("./Figures/Racing Bar Chart-{}.mp4".format(field), dpi=100,bitrate=30,fps=1.4)
        # subprocess.run(["open", "-a", "/Applications/QuickTime Player.app", "Racing Bar Chart-{}.mp4".format(field)])


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    debug_map = {
        0: logging.INFO,
        1: logging.WARNING,
        2: logging.DEBUG,
        3: logging.ERROR,
    }
    logging.root.setLevel(level=debug_map[0])


    # main(plot_fits=True, plot_bar_plot=True, plot_bar_plot_video=True)
    main(plot_fits=True, plot_bar_plot=False, plot_bar_plot_video=False)

