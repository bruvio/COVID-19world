import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
import logging
import warnings
import datetime
import pdb
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.animation as animation
# from IPython.display import HTML
label_size = 8
mpl.rcParams["xtick.labelsize"] = label_size
from scipy.optimize import curve_fit
import pycountry as pc

# do not print unnecessary warnings during curve_fit()
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
datatemplate = 'time_series_19-covid-{}.csv'
fields = ['Confirmed', 'Deaths', 'Recovered']

def pre_process_database(datatemplate,fields):
    dfs = dict()
    for field in fields:
        dfs[field] = pd.read_csv(datatemplate.format(field))

    # print(dfs['Confirmed'].head())

    # loop on the dataframe dictionary
    for field, df in dfs.items():
        # group by country, to sum on states
        df = df.groupby('Country/Region', as_index=False).sum()
        # turn each measurement column into a separate line,
        # and store the results in a new dataframe
        df = df.melt(id_vars=['Country/Region', 'Lat', 'Long'],
                     value_name='counts')
        # keep track of the quantity that is measured
        # either Confirmed, Deaths, or Recovered
        df['quantity'] = field
        # change column names
        df.columns = ['country', 'lat', 'lon', 'date', 'counts', 'quantity']
        # replace the dataframe in the dictionary
        dfs[field] = df

    dfall = pd.concat(dfs.values())
    dfall['date'] = pd.to_datetime(dfall['date'])
    return dfall

def select_database(database,country,field):
    sel = database[(database['country'] == country ) &
                (database['quantity'] == field)]
    return sel,sel['date'],sel['counts']


def get_times(dataframe, y, prediction_days):

    date = dataframe.index.values

    # start = pd.Timestamp(dataframe['Date'].loc[0])
    start = dataframe["date"].iloc[0]
    # start = start.to_pydatetime()

    # start = datetime.datetime.strptime(
    #     start, "%m/%d/%y"
    # )
    # day_of_the_year_start = int(start[0:2])
    # x = np.arange(day_of_the_year_start, len(y)+day_of_the_year_start)
    x = np.arange(0, len(y))
    # start = datetime.datetime.strptime(start, "%d/%m/%y")

    end = dataframe["date"].iloc[-1]
    # end = datetime.datetime.strptime(end, "%d/%m/%y")

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
    days = len(t_prediction)

    start_prediction_date = date
    stop_prediction_date = days
    date_prediction = np.arange(0, stop_prediction_date)

    return t_real, t_prediction, x, start, prediction, days


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


def exp_func(x, a, b):
    return a * np.exp(b * x)

def expo_func(x,a,b):
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


def fit_data(x, y, func):
    # these are the same as the scipy defaults
    # initialParameters = np.array([1.0, 1.0, 1.0])

    # curve fit the test data
    # fittedParameters, pcov = curve_fit(func, x, totale_casi, initialParameters, maxfev=5000)
    try:
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


def plot_model(xModel, yModel, country, label, color, marker, logscale=False):

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
    )

    # plt.legend(loc='best')
    # plt.xticks(rotation=15, ha="right")
def plot(dataframe,countries, xrange,
             dtype='Confirmed',
             yrange=None,
             yscale='linear'):
        '''plot the covid-19 data with an exponential fit.
        - countries: list of countries
        - xrange: fit range, e.g. (30,55)
        - yscale: log or linear
        - yrange: well, y range, useful in log mode.
        '''
        xmin, xmax = xrange
        linx = np.linspace(xmin, xmax, 101)
        colors = ['blue', 'red', 'orange', 'green']
        for i, country in enumerate(countries):
            color = colors[i]
            sel = dataframe[(dataframe['country'] == country) &
                        (dataframe['quantity'] == dtype)]
            yp = sel['counts'][xmin:xmax + 1]
            xp = np.arange(len(yp)) + xmin
            plt.scatter(xp, yp, label=country,
                        alpha=0.7, marker='.', color=color)
            pars, cov = curve_fit(expo_func, xp, yp)
            f = expo_func(linx, *pars)
            plt.plot(linx, f,
                     color=color, alpha=0.3)
        plt.legend(loc='upper left')
        plt.xlabel('days')
        plt.ylabel('confirmed cases')
        plt.yscale(yscale)
        if yrange:
            plt.ylim(*yrange)
        plt.grid()




def main():
    dataframe_all_countries = pre_process_database(datatemplate, fields)
    countrylist_df = list(set(dataframe_all_countries["country"]))

    countrylist = [
        "United Kingdom",
        "US",
        "Germany",
        "Italy",
        "China",
        "Singapore",
        "Australia",
        "France",
        "Switzerland",
        "Iran",
        "Korea, South",
        "Romania",
    ]
    countrylist = ["Italy"]
    # countrylist = ['United Kingdom']
    # countrylist = ['Iran']
    # logscale= True
    logscale = False
    if 1:
        for country in countrylist:
            if country in countrylist_df:
                print(country)
                databasename = "confirmed cases"
                dataframe,x,y = select_database(dataframe_all_countries, country, 'Confirmed')
                prediction_dates = 75
                day_to_use_4_fit = 5
                t_real, t_prediction, x, start, prediction, days = get_times(
                    dataframe, y, prediction_dates
                )


                if country == "Italy":
                    xModel_fit, yModel_fit, fittedParameters, Rsquared = fit_data(
                        x, y, sigmoidal_func
                    )
                    text_fit = text = '${} / (1 + exp^{{(-{} * (x - {}))}})$'.format(float(fittedParameters[0]),float(fittedParameters[1]),float(fittedParameters[2]))

                elif country == "United Kingdom":
                    xModel_fit, yModel_fit, fittedParameters, Rsquared = fit_data(
                        x, y, exp_func1
                    )
                    text_fit = '$ exp^{{ {} \cdot x}}$'.format(
                        float(fittedParameters[-1]))
                else :
                    xModel_fit, yModel_fit, fittedParameters, Rsquared = fit_data(
                        x, y, exp_func1
                    )
                    text_fit = '$ exp^{{ {} \cdot x}}$'.format(
                        float(fittedParameters[-1]))
                xModel_date = xModel_fit.astype(datetime.datetime)
                # italy Parameters: [1.06353071e+05 5.88260356e+01 2.08552443e-01]
                # UK Parameters: [0.00114855 0.26418431]

                plt.figure(num=country)
                if country == "Italy":
                    print('\n last {} days fit data\n'.format(day_to_use_4_fit))
                    xModel, yModel, fittedParameters_10, Rsquared = fit_data(
                        x[-day_to_use_4_fit:], y[-day_to_use_4_fit:], exp_func1
                    )
                    text_10days_fit = '$ exp^{{ {} \cdot x}}$'.format(
                        float(fittedParameters_10[-1]))
                elif country == "US":
                    print('\n last {} days fit data\n'.format(day_to_use_4_fit))
                    xModel, yModel, fittedParameters_10, Rsquared = fit_data(
                        x[-day_to_use_4_fit:], y[-day_to_use_4_fit:], exp_func1
                    )
                    text_10days_fit = '$ exp^{{ {} \cdot x}}$'.format(
                        float(fittedParameters_10[-1]))
                else:
                    print('\n last {} days fit data\n'.format(day_to_use_4_fit))
                    xModel, yModel, fittedParameters_10, Rsquared = fit_data(
                        x[-day_to_use_4_fit:], y[-day_to_use_4_fit:], exp_func1
                    )
                    text_10days_fit = '$ exp^{{ {} \cdot x}}$'.format(
                        float(fittedParameters_10[-1]))

                ax1 = plt.subplot(211)

                plot_data(t_real, y, country, "confirmed cases", "r", logscale=logscale)

                text = '$ exp^{{ {} \cdot x}}$'.format(
                    float(fittedParameters_10[-1]))
                plot_model(
                    t_real,
                    yModel_fit,
                    country,
                    " data fit - " + databasename + "  -  "+text_fit,
                    "b",
                    marker="x",
                    logscale=logscale,
                )

                plot_model(
                    t_real[-day_to_use_4_fit:],
                    yModel[-day_to_use_4_fit:],
                    country,
                    " 11days-fit - " + databasename + "  -  "+text_10days_fit,
                    "g",
                    marker="x",
                    logscale=logscale,
                )
                plt.ylim([min(y),max(y)])


                print("prediction from {} to {}".format(start, prediction))
                #
                fittedParameters_prediction, pcov = curve_fit(
                    sigmoidal_func, x, y, maxfev=5000, p0=[1, -fittedParameters_10[-1], 1]
                )

                # start = datetime.datetime.strptime(
                #     dataframe["date"].loc[0], "%m/%d/%y"
                # )
                start = dataframe["date"].iloc[0]

                day_of_the_year_start = start.day
                x = np.arange(day_of_the_year_start, len(y) + day_of_the_year_start)
                x_prediction = np.arange(0, len(y) + day_of_the_year_start)
                modelPredictions = sigmoidal_func(
                    x_prediction, *fittedParameters_prediction
                )
                #
                absError = modelPredictions[0 : len(y)] - y
                #
                print('\nsigmoidal fit data\n')
                SE = np.square(absError)  # squared errors
                MSE = np.mean(SE)  # mean squared errors
                RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
                Rsquared = 1.0 - (np.var(absError) / np.var(y))
                #
                print("Parameters:", fittedParameters_prediction)
                print("RMSE:", RMSE)
                print("R-squared:", Rsquared)
                # UK Parameters: [2.15051089e+08 9.82481446e+01 2.64184732e-01]

                xModel_prediction = np.arange(0, days)
                yModel_prediction = sigmoidal_func(
                    xModel_prediction, *fittedParameters_prediction
                )

                text = '${} / (1 + exp^{{(-{} * (x - {}))}})$'.format(float(fittedParameters_prediction[0]),float(fittedParameters_prediction[1]),float(fittedParameters_prediction[2]))
                plot_model(
                    t_prediction,
                    yModel_prediction,
                    country,
                    "predictions - "
                    + databasename
                    + "  -  "+text,
                    "m",
                    marker=".",
                    logscale=logscale,
                )
                # plt.ylim(1, 1e4)
                plt.xticks(rotation=15, ha="right")

                plt.xlabel("days since it started")  # X axis data label
                plt.ylabel("confirmed cases")  # Y axis data label
                plt.legend(loc="best", fontsize="6")
                if logscale:
                    plt.savefig("./Figures/" + country + "_fitted_log.png", dpi=100)
                    plt.ylim(1, 1e5)
                else:
                    plt.savefig("./Figures/" + country + "_fitted.png", dpi=100)

                daily = np.diff(y)
                a = 0
                daily = np.concatenate([[a], daily])
                # yy = sigmoidal_func(x, *fittedParameters_prediction)
                a, b, c = fittedParameters_prediction
                f = lambda x: a / (1 + np.exp(-c * (x - b)))
                # yy = f(x)
                ydx = derivative(f, xModel_prediction)

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
                    t_real, daily, country, "daily confirmed cases", "r", logscale=logscale
                )
                # plt.ylim(1, 1e4)
                plt.legend(loc="best", fontsize="6")
                plt.xticks(rotation=15, ha="right")






                y = dataframe['counts']
                x = np.arange(len(y))




                plt.figure()
                plt.scatter(x, y)
                results = curve_fit(expo_func, x, y)
                # results
                plt.plot(x, expo_func(x, *results[0]))


                linx = np.linspace(0, 50, 101)
                plt.figure()
                plt.scatter(x, y)
                plt.plot(linx, expo_func(linx, *results[0]))


                plt.figure(dpi=90, figsize=(8, 4))
                plot(dataframe_all_countries,['Italy'],
                     dtype='Confirmed',
                     xrange=(30, 56),
                     yscale='log')

                plt.figure()
                # the previous fit:
                linx = np.linspace(0, 50, 101)
                plt.scatter(x, y)
                plt.plot(linx, expo_func(linx, *results[0]))
                # multiplying the last count by 1.2:
                y_p = y.copy()
                y_p.iloc[-1] = y.iloc[-1] * 1.2
                plt.scatter(x, y_p)
                results_p = curve_fit(expo_func, x, y_p)
                plt.plot(linx, expo_func(linx, *results_p[0]))
                plt.xlim(30, 50)

                plt.show(block=True)

if __name__ == "__main__":
    main()
