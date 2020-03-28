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
        # fittedParameters, pcov = curve_fit(func2, x[:max_size], totale_casi[:max_size], maxfev=5000)
        # fittedParameters[2]=90
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
        plt.ylim(1, 1e4)
    # plt.plot(Xdata, Ydata, 'D', label="Original Data - " + country + ' - ' + label,color=color)
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
        plt.ylim(1, 1e4)

    # plt.plot(xModel, yModel, label="Fitted Data - " + country + ' - ' + label, color=color,marker=marker)
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
    datatemplate = "time_series_covid19_{}_global.csv"
    fields = ["confirmed", "deaths", "recovered"]
    dataframe_all_countries = pre_process_database(datatemplate, fields)
    list_of_files = glob.glob('worldmeter_data/*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    # for filename in os.listdir('worldmeter_data'):
    #     file = filename.replace('_webData.csv','')
    #     date_filename = datetime.datetime.strptime(file.split(" ")[0], '%m_%d_%Y_%H_%M')
    #     if date_filename < datetime.datetime.now() :

    dataframe_all_countries_last_update = pd.read_csv(
                                              latest_file)



    countrylist_df = list(set(dataframe_all_countries["country"]))



    # countrynotinlist = []
    # i = 0
    # for region in df.columns:
    #     if region in list(set(dataframe_all_countries["country"])):
    #         print(region)
    #         i = i + 1
    #     else:
    #         countrynotinlist.append(region)

    countrylist = []
    countrylist.append("Italy")
    # countrylist.append("Australia")
    countrylist.append("Germany")
    # countrylist.append("China")
    # countrylist.append("Australia")
    countrylist.append("US")
    # countrylist.append("France")
    # countrylist.append("Korea, South")
    # countrylist.append("Switzerland")
    countrylist.append("United Kingdom")
    # countrylist.append("Japan")
    # countrylist.append("Romania")

    # countrylist = [
    #     "United Kingdom",
    #     "US",
    #     "Germany",
    #     "Italy",
    #     "China",
    #     "Singapore",
    #     "Australia",
    #     "France",
    #     "Switzerland",
    #     "Iran",
    #     "Korea, South",
    #     "Romania",
    #     "Colombia",
    # ]
    # countrylist = ["Italy"]
    # countrylist = ['United Kingdom']
    # countrylist = ['Iran']
    # countrylist = ['Singapore']
    # countrylist = ['US']
    # countrylist = ['Switzerland']
    # countrylist = ['Germany']
    # countrylist = ['France']
    # countrylist = ['Japan']
    # countrylist = ['China']
    # countrylist = ['Russia']
    # countrylist = ['Australia']
    exception_list = []
    exception_list.append("Australia")
    exception_list.append("China")
    exception_list.append("Australia")
    # exception_list.append("Italy")
    exception_list.append("US")
    exception_list.append("France")
    exception_list.append("Switzerland")
    exception_list.append("United Kingdom")
    exception_list.append("Japan")
    # logscale= True
    logscale = False
    if plot_fits:
        for country in countrylist:
            if country in countrylist_df:
                print(country)
                databasename = "Confirmed cases"
                # databasename = "Recovered cases"
                # databasename = "Deaths cases"





                dataframe, x, y = select_database(
                    dataframe_all_countries, country, "confirmed"
                )
                dataframe_deaths, x_deaths, y_deaths = select_database(
                    dataframe_all_countries, country, "deaths"
                )
                dataframe_recovered, x_recovered, y_recovered = select_database(
                    dataframe_all_countries, country, "recovered"
                )
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
                    df = dataframe_all_countries_last_update2.pivot(index="date", columns="Country/Region", values=field)

                    df_country = df[[country]]





                    # dataframe.append(pd.Series(name=df_country.index[0]))
                    if field =='confirmed':
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

                # italy Parameters: [1.06353071e+05 5.88260356e+01 2.08552443e-01]
                # UK Parameters: [0.00114855 0.26418431]

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
                # plt.figure(num=country + "_fit")
                # # ax1 = plt.subplot(211)
                #
                # plot_data(t_real, y, country, "confirmed cases", "r", logscale=logscale)
                #
                # plot_model(
                #     t_real,
                #     yModel_fit,
                #     country,
                #     " data fit - " + databasename + "  -  " + text_fit,
                #     "b",
                #     marker="x",
                #     logscale=logscale,
                # )

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
                # plt.legend(loc='best', fontsize=8)
                # plt.legend(loc=9, bbox_to_anchor=(0.5,-0.02), fontsize=8)
                # ax1.legend(bbox_to_anchor=(0.5, 1.1),
                #           fancybox=True, shadow=True, ncol=1, fontsize=8)
                ax1.set_ylim([min(y), max(y)*1.05])
                # plt.show()
                ###########################

                ################ PREDICTION ###########################
                print("prediction from {} to {}".format(start, prediction))
                #
                # fittedParameters_prediction, pcov = curve_fit(
                #     sigmoidal_func, x, y, maxfev=5000, p0=[1, fittedParameters_10[-1], 1]
                # )

                # start = datetime.datetime.strptime(
                #     dataframe["date"].loc[0], "%m/%d/%y"
                # )
                start = dataframe["date"].iloc[0]

                day_of_the_year_start = start.day
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

                # UK Parameters: [2.15051089e+08 9.82481446e+01 2.64184732e-01]

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

                # plt.text(0.0, 0.1, 'matplotlib', horizontalalignment='center',
                # verticalalignment = 'center',
                # transform = ax1.transAxes)
                # fig.tight_layout(rect=[0, 0.1, 1, 0.95])
                # plt.legend(loc='best', bbox_to_anchor=(0.5, -1.02), fontsize=8)
                # plt.legend(loc="best", fontsize=8)

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

                # y = dataframe['counts']
                # x = np.arange(len(y))
                #
                #
                #
                #
                # plt.figure()
                # plt.scatter(x, y)
                # results = curve_fit(expo_func, x, y)
                # # results
                # plt.plot(x, expo_func(x, *results[0]))
                #
                #
                # linx = np.linspace(0, 50, 101)
                # fields = ['Confirmed', 'Deaths', 'Recovered']
                # plt.figure()
                # dataframe, x, Confirmed = select_database(dataframe_all_countries, country, 'Confirmed')
                # plt.scatter(x, Confirmed,label = 'Confirmed')
                # dataframe, x, Deaths = select_database(dataframe_all_countries, country, 'Deaths')
                # plt.scatter(x, Deaths,label = 'Deaths')
                # dataframe, x, Recovered = select_database(dataframe_all_countries, country, 'Recovered')
                # plt.scatter(x, Recovered,label = 'Recovered')
                # plt.legend(loc='best', fontsize = 8)
                # plt.figure()
                # dataframe, x, Confirmed = select_database(dataframe_all_countries, country, 'Confirmed',)
                # plt.plot(x, Confirmed,label = 'Confirmed',marker ='x')
                # dataframe, x, Deaths = select_database(dataframe_all_countries, country, 'Deaths')
                # plt.plot(x, Deaths,label = 'Deaths',marker ='x')
                # dataframe, x, Recovered = select_database(dataframe_all_countries, country, 'Recovered')
                # plt.plot(x, Recovered,label = 'Recovered',marker ='x')
                # plt.legend(loc='best', fontsize = 8)
                # # plt.plot(linx, expo_func(linx, *results[0]))

                # The Q-Q plot, or quantile-quantile plot, is a graphical tool
                # to help us assess if a set of data plausibly came from some theoretical distribution
                # from statsmodels.graphics.gofplots import qqplot
                # plt.figure()
                # qqplot(ydx, line='s')
                # plt.show()



                try:
                    plt.figure(dpi=90, figsize=(8, 4))
                    plot(
                        dataframe_all_countries,
                        [country],
                        dtype="confirmed",
                        xrange=(30, 56),
                        yscale="log",
                    )
                    # fields = ['Confirmed', 'Deaths', 'Recovered']
                    plt.savefig(
                        "./Figures/"
                        + country
                        + "Confirmed_expo_fit_log_scale-{}.png".format(today),
                        dpi=100,
                    )
                except:
                    print(
                        "unable to fit {} data with {}".format(
                            country, expo_func.__name__
                        )
                    )

                # plt.figure()
                # # the previous fit:
                # linx = np.linspace(0, 50, 101)
                # plt.scatter(x, y)
                # plt.plot(linx, expo_func(linx, *results[0]))
                # # multiplying the last count by 1.2:
                # y_p = y.copy()
                # y_p.iloc[-1] = y.iloc[-1] * 1.2
                # plt.scatter(x, y_p)
                # results_p = curve_fit(expo_func, x, y_p)
                # plt.plot(linx, expo_func(linx, *results_p[0]))
                # plt.xlim(30, 50)

                fig_rate = go.Figure()
                tickList = list(
                    np.arange(
                        0,
                        (dataframe_deaths["counts"] / dataframe["counts"] * 100).max()
                        + 0.2,
                        0.5,
                    )
                )

                fig_rate.add_trace(
                    go.Scatter(
                        x=dataframe_deaths["date"],
                        y=dataframe_deaths["counts"] / dataframe["counts"] * 100,
                        mode="lines+markers",
                        line_shape="spline",
                        name=country,
                        line=dict(color="#626262", width=4),
                        marker=dict(
                            size=4, color="#f4f4f2", line=dict(width=1, color="#626262")
                        ),
                        text=[
                            datetime.datetime.strftime(d, "%b %d %Y AEDT")
                            for d in dataframe_deaths["date"]
                        ],
                        hovertext=[
                            country + " death rate (%) <br>{:.2f}%".format(i)
                            for i in dataframe_deaths["counts"]
                            / dataframe["counts"]
                            * 100
                        ],
                        hovertemplate="<b>%{text}</b><br></br>"
                        + "%{hovertext}"
                        + "<extra></extra>",
                    )
                )

                fig_rate.update_layout(
                    # title=country + " death rate (%)",
                    xaxis_title="days",
                    yaxis_title=country + " death rate (%)",
                    margin=go.layout.Margin(l=10, r=10, b=10, t=5, pad=0),
                    yaxis=dict(
                        showline=False,
                        linecolor="#272e3e",
                        zeroline=False,
                        # showgrid=False,
                        gridcolor="rgba(203, 210, 211,.3)",
                        gridwidth=0.1,
                        tickmode="array",
                        # Set tick range based on the maximum number
                        tickvals=tickList,
                        # Set tick label accordingly
                        ticktext=["{:.1f}".format(i) for i in tickList],
                    ),
                    #    yaxis_title="Total Confirmed Case Number",
                    xaxis=dict(
                        showline=False,
                        linecolor="#272e3e",
                        showgrid=False,
                        gridcolor="rgba(203, 210, 211,.3)",
                        gridwidth=0.1,
                        zeroline=False,
                    ),
                    xaxis_tickformat="%b %d",
                    hovermode="x",
                    legend_orientation="h",
                    # legend=dict(x=.02, y=.95, bgcolor="rgba(0,0,0,0)",),
                    plot_bgcolor="#f4f4f2",
                    paper_bgcolor="#cbd2d3",
                    font=dict(color="#292929"),
                )
                # fig_rate.write_image("Figures/death_rates_" + country )
                plotly.offline.plot(
                    fig_rate,
                    filename="Figures/death_rates_" + country ,
                    auto_open=False,
                )
                # fig_rate.show()
                #
                # # Pseduo data for logplot
                #
                # daysOutbreak = (dataframe_confirmed['date'].iloc[0] -
                #                 datetime.datetime.strptime('12/31/2019', '%m/%d/%Y')).days
                # pseduoDay = np.arange(1, daysOutbreak + 1)
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
                #
                # plot_model(
                #     t_real,
                #     y2,
                #     country,
                #     " 8% growth rate - " + databasename ,
                #     "g",
                #     marker="o",
                #     logscale=logscale,linewidths = 0.1
                # )
                # plot_model(
                #     t_real,
                #     y3,
                #     country,
                #     " 7% growth rate - " + databasename ,
                #     "y",
                #     marker=">",
                #     logscale=logscale,linewidths = 0.1
                # )
                # plot_model(
                #     t_real,
                #     y4,
                #     country,
                #     " 5% growth rate - " + databasename ,
                #     "r",
                #     marker="<",
                #     logscale=logscale,linewidths = 0.1
                # )
                plt.ylim([min(y), max(y)*1.05])
                plt.legend(loc="best", fontsize=8)
                # # Read cumulative data of a given region from ./cumulative_data folder
                # # dfs_curve = pd.read_csv('./lineplot_data/dfs_curve.csv')
                #
                # # Create empty figure canvas
                # fig_curve = go.Figure()
                # fig_curve.add_trace(go.Scatter(x=pseduoDay,
                #                                y=y1,
                #                                line=dict(color='#613262', width=4),
                #                                marker=dict(size=4, color='#f4f4f2',
                #                                            line=dict(width=1, color='#626262')),
                #                                text=[
                #                                    '85% growth rate' for i in pseduoDay],
                #                                hovertemplate='<b>%{text}</b><br></br>' +
                # '%{hovertext}' +
                # '<extra></extra>'
                #                                )
                #                     )
                # fig_curve.add_trace(go.Scatter(x=pseduoDay,
                #                                y=y2,
                #                                line=dict(color='#625462', width=4),
                #                                marker=dict(size=4, color='#f4f4f2',
                #                                            line=dict(width=1, color='#626262')),
                #                                text=[
                #                                    '35% growth rate' for i in pseduoDay],
                #                                hovertemplate='<b>%{text}</b><br></br>' +
                # '%{hovertext}' +
                # '<extra></extra>'
                #                                )
                #                     )
                # fig_curve.add_trace(go.Scatter(x=pseduoDay,
                #                                y=y3,
                #                                line=dict(color='#625462', width=4),
                #                                marker=dict(size=4, color='#f4f4f2',
                #                                            line=dict(width=1, color='#626262')),
                #                                text=[
                #                                    '15% growth rate' for i in pseduoDay],
                #                                hovertemplate='<b>%{text}</b><br></br>' +
                # '%{hovertext}' +
                # '<extra></extra>'
                #                                )
                #                     )
                # fig_curve.add_trace(go.Scatter(x=pseduoDay,
                #                                y=y4,
                #                                line=dict(color='#626262', width=4),
                #                                marker=dict(size=4, color='#f4f4f2',
                #                                            line=dict(width=1, color='#626262')),
                #                                text=[
                #                                    '5% growth rate' for i in pseduoDay],
                #                                hovertemplate='<b>%{text}</b><br></br>' +
                # '%{hovertext}' +
                # '<extra></extra>'
                #                                )
                #                     )
                # fig_curve.add_trace(go.Scatter(x=dfs_curve.loc[dfs_curve['Region'] == regionName]['DayElapsed'],
                #                                y=dfs_curve.loc[dfs_curve['Region'] == regionName]['Confirmed'],
                #                                mode='lines',
                #                                line_shape='spline',
                #                                name=regionName,
                #                                # opacity=0.5,
                #                                line=dict(color='rgba(0,0,0,.3)', width=1.5),
                #                                text=[
                #                                    i for i in
                #                                    dfs_curve.loc[dfs_curve['Region'] == regionName]['Region']],
                #                                hovertemplate='<b>%{text}</b><br>' +
                #                                              '<br>%{x} days after 100 cases<br>' +
                #                                              'with %{y:,d} cases<br>'
                #                                              '<extra></extra>'
                #                                )
                #                     )
                # fig_curve.add_trace(go.Scatter(x=dfs_curve.loc[dfs_curve['Region'] == Region]['DayElapsed'],
                #                            y=dfs_curve.loc[dfs_curve['Region'] == Region]['Confirmed'],
                #                            mode='lines',
                #                            line_shape='spline',
                #                            name=Region,
                #                            line=dict(color='#d7191c', width=3),
                #                            text=[
                #                                i for i in dfs_curve.loc[dfs_curve['Region'] == Region]['Region']],
                #                            hovertemplate='<b>%{text}</b><br>' +
                #                                          '<br>%{x} days after 100 cases<br>' +
                #                                          'with %{y:,d} cases<br>'
                #                                          '<extra></extra>'
                #                            )
                #                 )

    plt.show(block=True)
    # Customise layout

    # fig_rate.show()

    # fig_curve.show()

    if plot_bar_plot:
        # field = 'Deaths'
        # field = 'Confirmed'
        # field = 'Recovered'
        # fields = ["Confirmed", "Deaths", "Recovered"]
        # datatemplate = "time_series_covid19_{}_global.csv"
        fields = ["confirmed", "deaths", "recovered"]
        for field in fields:
            # df = pd.read_csv(datatemplate.format(field))

            df = dataframe_all_countries[(dataframe_all_countries["quantity"] == field)]
            df = df.reset_index()

            df = df[["country", "date", "counts"]]

            df = df.pivot(index="country", columns="date", values="counts")
            df = df.reset_index()
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
            frames_list = df["date"].unique().tolist()
            # for i in range(10):
            #     frames_list.append(df["date"].iloc[-1])

            all_names = df["country"].unique().tolist()
            random_hex_colors = []
            for i in range(len(all_names)):
                random_hex_colors.append("#" + "%06X" % randint(0, 0xFFFFFF))

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
    # main(plot_fits=True, plot_bar_plot=True, plot_bar_plot_video=False)
    # main(plot_fits=False, plot_bar_plot=True, plot_bar_plot_video=False)
    # main(plot_fits=True, plot_bar_plot=False, plot_bar_plot_video=True)
    main(plot_fits=True, plot_bar_plot=False, plot_bar_plot_video=False)
