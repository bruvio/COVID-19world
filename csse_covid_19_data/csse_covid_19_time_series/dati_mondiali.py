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
from IPython.display import HTML
label_size = 8
mpl.rcParams["xtick.labelsize"] = label_size
from scipy.optimize import curve_fit
import pycountry as pc

# do not print unnecessary warnings during curve_fit()
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def country_name_check():
    pycntrylst = list(pc.countries)
    alpha_2 = []
    alpha_3 = []
    name = []
    common_name = []
    official_name = []
    invalid_countrynames = []
    tobe_deleted = [
        "IRAN",
        " SOUTH KOREA",
        "NORTH KOREA",
        "SUDAN",
        "MACAU",
        "REPUBLIC OF IRELAND",
    ]
    for i in pycntrylst:
        alpha_2.append(i.alpha_2)
        alpha_3.append(i.alpha_3)
        name.append(i.name)
        if hasattr(i, "common_name"):
            common_name.append(i.common_name)
        else:
            common_name.append("")
        if hasattr(i, "official_name"):
            official_name.append(i.official_name)
        else:
            official_name.append("")
    for j in input_country_list:
        if (
            j not in map(str.upper, alpha_2)
            and j not in map(str.upper, alpha_3)
            and j not in map(str.upper, name)
            and j not in map(str.upper, common_name)
            and j not in map(str.upper, official_name)
        ):
            invalid_countrynames.append(j)
    invalid_countrynames = list(set(invalid_countrynames))
    invalid_countrynames = [
        item for item in invalid_countrynames if item not in tobe_deleted
    ]
    return print(invalid_countrynames)


confirmed_df = pd.read_csv("time_series_19-covid-Confirmed.csv")
deaths_df = pd.read_csv("time_series_19-covid-Deaths.csv")
recovered_df = pd.read_csv("time_series_19-covid-Recovered.csv")


input_country_list = list(confirmed_df["Country/Region"])
input_country_list = [element.upper() for element in input_country_list]
#
country_name_check()


print(confirmed_df.columns)


confirmed_df_reshaped = confirmed_df.melt(
    id_vars=["Province/State", "Country/Region", "Lat", "Long"],
    var_name="Date",
    value_name="cases",
)
# confirmed_df_reshaped.set_index('Date', inplace=True)
deaths_df_reshaped = deaths_df.melt(
    id_vars=["Province/State", "Country/Region", "Lat", "Long"],
    var_name="Date",
    value_name="cases",
)
# deaths_df_reshaped.set_index('Date', inplace=True)
recovered_df_reshaped = recovered_df.melt(
    id_vars=["Province/State", "Country/Region", "Lat", "Long"],
    var_name="Date",
    value_name="cases",
)
# recovered_df_reshaped.set_index('Date', inplace=True)

active_df_reshaped = confirmed_df_reshaped.copy()
active_cases = confirmed_df_reshaped["cases"] - recovered_df_reshaped["cases"]
active_df_reshaped["cases"] = active_cases


# for country in countrylist:
#     if 'Korea' in country.split():
#         print(country)


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


def extract_database(country, dataframe):
    country_database = dataframe[dataframe["Country/Region"] == country]
    country_database = country_database.reset_index(drop=True)
    df_bydate_grouper = (
        country_database.groupby("Date", sort=False)["cases"]
        .sum()
        .reset_index(name="Total Cases")
    )

    totale_casi = np.array(df_bydate_grouper["Total Cases"])

    # data = df_bydate_grouper.index.values

    return df_bydate_grouper, totale_casi


def print_Data_by_country(country, dataframe, label, show_fit=False, logscale=False):
    try:
        # dataframe_bycountry = dataframe.groupby(dataframe['Country/Region'])
        country_database = dataframe[dataframe["Country/Region"] == country]
        country_database = country_database.reset_index(drop=True)
        df_bydate_grouper = (
            country_database.groupby("Date", sort=False)["cases"]
            .sum()
            .reset_index(name="Total Cases")
        )

        totale_casi = np.array(df_bydate_grouper["Total Cases"])

        # data = df_bydate_grouper.index.values

        x = np.arange(0, len(totale_casi))

        if logscale:
            plt.yscale("log")
        popt1, pcov1 = curve_fit(lambda t, a, b: a * np.exp(b * t), x, totale_casi)
        plt.scatter(
            data,
            totale_casi,
            marker="o",
            label="Original Data - " + country + " - " + label,
        )
        if show_fit:
            plt.plot(
                data,
                exp_func(x, *popt1),
                marker="-",
                label="Fitted Curve -" + country + " - " + label,
            )
        plt.xticks(rotation=15, ha="right")
        plt.legend(loc="best")

        return df_bydate_grouper
    except:
        logger.info("country {} not in database".format(country))
        return 0


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


def get_times(dataframe, y, prediction_days):

    date = dataframe.index.values

    # start = pd.Timestamp(dataframe['Date'].loc[0])
    start = datetime.datetime.strptime(dataframe["Date"].loc[0], "%m/%d/%y").strftime(
        "%d/%m/%y"
    )
    day_of_the_year_start = int(start[0:2])
    # x = np.arange(day_of_the_year_start, len(y)+day_of_the_year_start)
    x = np.arange(0, len(y))
    start = datetime.datetime.strptime(start, "%d/%m/%y")

    end = datetime.datetime.strptime(dataframe["Date"].iloc[-1], "%m/%d/%y").strftime(
        "%d/%m/%y"
    )
    end = datetime.datetime.strptime(end, "%d/%m/%y")

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


# graphics output section


# countrylist = ['Italy']
# for country in countrylist:
#     logger.info('plotting {} data'.format(country))
#     plt.figure()
#     df_bydate_grouper_country = print_Data_by_country(country, deaths_df_reshaped, 'death-cases',
#                                                       show_fit=True,logscale=False)
#     df_bydate_grouper_country = print_Data_by_country(country, confirmed_df_reshaped, 'confirmed-cases',
#                                                       show_fit=True,logscale=False)
#     df_bydate_grouper_country = print_Data_by_country(country, recovered_df_reshaped, 'recovered-cases',
#                                                       show_fit=True,logscale=False)
#     logscale = False
#     if logscale:
#         plt.savefig('./Figures/' + country + '_fitted_log.png', dpi=100)
#     else:
#         plt.savefig('./Figures/' + country + '_fitted.png', dpi=100)
#
# plt.show()


def main():
    countrylist_df = list(set(confirmed_df_reshaped["Country/Region"]))

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
    # countrylist = ["Italy"]
    # countrylist = ['United Kingdom']
    # countrylist = ['Iran']
    if 0:
        for country in countrylist:
            if country in countrylist_df:
                logger.info("plotting {} data".format(country))
                # try:
                print(country)
                # logscale= True
                logscale = False
                databasename = "confirmed cases"
                dataframe, y = extract_database(country, confirmed_df_reshaped)
                y=y[0:-1]
                prediction_dates = 75
                day_to_use_4_fit = 6
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

                start = datetime.datetime.strptime(
                    dataframe["Date"].loc[0], "%m/%d/%y"
                ).strftime("%d/%m/%y")
                day_of_the_year_start = int(start[0:2])
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



    # plt.show(block=True)
    plt.close('all')


    plt.style.use('ggplot')

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    fig, ax = plt.subplots(figsize=(15, 8))
    def dataframe_plot(day,pos):
        ax.bar(pos, y[day], color='blue')
        ax.set_ylabel("Total cases")

        # ax.set_xticks(0)
        # ax.set_xticklabels(countrylist,rotation=15, ha="right")

    dataframe, y = extract_database('Italy', confirmed_df_reshaped)
    dataframe.index = list(dataframe.index)
    # range_dates = dataframe['Date']
    animator = animation.FuncAnimation(fig, dataframe_plot,frames=len(y),fargs = (0,),
                              interval=100
                                       )

    plt.show()

    # x_pos = [i for i, _ in enumerate(countrylist )]
    #
    #     # ax = plt.subplot()
    #     # fig, ax = plt.subplots(figsize=(15, 8))
    #
    # for j,country in enumerate(countrylist):
    #     if country in countrylist_df:
    #         logger.info("plotting {} data".format(country))
    #
    #
    #         dataframe, y = extract_database(country, confirmed_df_reshaped)
    #         dataframe.index = list(dataframe.index)
    #         range_dates = dataframe['Date']
    #         animator = animation.FuncAnimation(fig, dataframe_plot,
    #                                            frames=len(y))
    #             #
    #             # dataframe_plot(-1)
    #     # ax.bar(x_pos[j], y[-1], color='blue')
    #     # ax.set_ylabel("Total cases")
    #     #
    #     # ax.set_xticks(x_pos)
    #     # ax.set_xticklabels(countrylist,rotation=15, ha="right")
    # plt.show()
    # HTML(animator.to_jshtml())




if __name__ == "__main__":
    main()
