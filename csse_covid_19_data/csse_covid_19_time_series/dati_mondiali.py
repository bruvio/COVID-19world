import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import logging
import warnings
import pdb
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
label_size = 8
mpl.rcParams['xtick.labelsize'] = label_size
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
    tobe_deleted = ['IRAN',' SOUTH KOREA', 'NORTH KOREA','SUDAN','MACAU','REPUBLIC OF IRELAND']
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
        if j not in map(str.upper,alpha_2) and j not in map(str.upper,alpha_3) and j not in map(str.upper,name) and j not in map(str.upper,common_name) and j not in map(str.upper,official_name):
            invalid_countrynames.append(j)
    invalid_countrynames = list(set(invalid_countrynames))
    invalid_countrynames = [item for item in invalid_countrynames if item not in tobe_deleted]
    return print(invalid_countrynames)

confirmed_df = pd.read_csv("time_series_19-covid-Confirmed.csv")
deaths_df = pd.read_csv("time_series_19-covid-Deaths.csv")
recovered_df = pd.read_csv("time_series_19-covid-Recovered.csv")


input_country_list = list(confirmed_df['Country/Region'])
input_country_list = [element.upper() for element in input_country_list]
#
country_name_check()


print(confirmed_df.columns)


confirmed_df_reshaped = confirmed_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],var_name="Date", value_name='cases')
# confirmed_df_reshaped.set_index('Date', inplace=True)
deaths_df_reshaped = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],var_name="Date", value_name='cases')
# deaths_df_reshaped.set_index('Date', inplace=True)
recovered_df_reshaped = recovered_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],var_name="Date", value_name='cases')
# recovered_df_reshaped.set_index('Date', inplace=True)

active_df_reshaped = confirmed_df_reshaped
active_df_reshaped['cases'] = confirmed_df_reshaped['cases'] - recovered_df_reshaped['cases']

def exp_func(x, a, b ):
    return a*np.exp(b*x)

def func(x, a, b, c): # Hill sigmoidal equation from zunzun.com
    return  a * np.power(x, b) / (np.power(c, b) + np.power(x, b))


def print_Data_by_country(country,dataframe,label,show_fit=False,logscale=False):
    try:
        # dataframe_bycountry = dataframe.groupby(dataframe['Country/Region'])
        country_database = dataframe[dataframe['Country/Region']==country]
        country_database = country_database.reset_index(drop=True)
        df_bydate_grouper = country_database.groupby('Date', sort=False)["cases"].sum().reset_index(name='Total Cases')

        totale_casi = np.array(df_bydate_grouper['Total Cases'])

        data = df_bydate_grouper.index.values

        x = np.arange(0, len(totale_casi))
        if logscale:
            plt.yscale('log')
        popt1, pcov1 = curve_fit(lambda t, a, b: a * np.exp(b * t), x, totale_casi)
        plt.scatter(data, totale_casi,marker='o', label="Original Data - "+country+' - '+label)
        if show_fit:
            plt.plot(data, exp_func(x, *popt1), marker='-', label="Fitted Curve -"+country+' - '+label)
        plt.xticks(rotation=15, ha="right")
        plt.legend(loc='best')

        return df_bydate_grouper
    except:
        logger.info('country {} not in database'.format(country))
        return 0

def ModelAndScatterPlot(country,dataframe,label,graphWidth, graphHeight,logscale):
    # these are the same as the scipy defaults
    # initialParameters = np.array([1.0, 1.0, 1.0])

    country_database = dataframe[dataframe['Country/Region'] == country]
    country_database = country_database.reset_index(drop=True)
    df_bydate_grouper = country_database.groupby('Date', sort=False)["cases"].sum().reset_index(name='Total Cases')

    totale_casi = np.array(df_bydate_grouper['Total Cases'])

    data = df_bydate_grouper.index.values

    x = np.arange(0, len(totale_casi))

    # curve fit the test data
    # fittedParameters, pcov = curve_fit(func, x, totale_casi, initialParameters, maxfev=5000)
    try:
        fittedParameters, pcov = curve_fit(func, x, totale_casi, maxfev=5000)
        modelPredictions = func(x, *fittedParameters)

        absError = modelPredictions - totale_casi

        SE = np.square(absError)  # squared errors
        MSE = np.mean(SE)  # mean squared errors
        RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (np.var(absError) / np.var(totale_casi))

        print('Parameters:', fittedParameters)
        print('RMSE:', RMSE)
        print('R-squared:', Rsquared)

        # first the raw data as a scatter plot
        if logscale:
            plt.yscale('log')
        plt.plot(data, totale_casi, 'D', label="Original Data - " + country + ' - ' + label)

        # create data for the fitted equation plot
        xModel = np.linspace(min(x), max(x))
        yModel = func(xModel, *fittedParameters)

        # now the model as a line plot
        plt.plot(xModel, yModel, label="Fitted Data - " + country + ' - ' + label)

        plt.xlabel('X Data')  # X axis data label
        plt.ylabel('Y Data')  # Y axis data label

        plt.legend(loc='best')
        plt.xticks(rotation=15, ha="right")

    except RuntimeError:



        # first the raw data as a scatter plot
        if logscale:
            plt.yscale('log')
        plt.plot(data, totale_casi,  'D',label="Original Data - "+country+' - '+label)





        plt.xlabel('X Data') # X axis data label
        plt.ylabel('Y Data') # Y axis data label

        plt.legend(loc='best')
        plt.xticks(rotation=15, ha="right")













# graphics output section

countrylist = list(set(confirmed_df_reshaped['Country/Region']))
countrylist = ['UK','US','Germany','Italy','Mainland China','Singapore','Australia','France']
# for country in countrylist:
#     logger.info('plotting {} data'.format(country))
#     plt.figure()
#     df_bydate_grouper_country = print_Data_by_country(country, deaths_df_reshaped, 'death-cases',
#                                                       show_fit=False,logscale=False)
#     df_bydate_grouper_country = print_Data_by_country(country, confirmed_df_reshaped, 'confirmed-cases',
#                                                       show_fit=False,logscale=False)
#     df_bydate_grouper_country = print_Data_by_country(country, recovered_df_reshaped, 'recovered-cases',
#                                                       show_fit=False,logscale=False)
#     logscale = False
#     if logscale:
#         plt.savefig('./Figures/' + country + '_fitted_log.png', dpi=100)
#     else:
#         plt.savefig('./Figures/' + country + '_fitted.png', dpi=100)


graphWidth = 800
graphHeight = 600
countrylist = ['UK','US','Germany','Italy','Mainland China','Singapore','Australia','France']
# countrylist = ['Italy']
for country in countrylist:
    logger.info('plotting {} data'.format(country))
    # try:
    print(country)
    plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    ModelAndScatterPlot(country,deaths_df_reshaped,'deaths-cases',graphWidth, graphHeight,logscale=True)
    ModelAndScatterPlot(country,confirmed_df_reshaped,'confirmed-cases',graphWidth, graphHeight,logscale=True)
    ModelAndScatterPlot(country,recovered_df_reshaped,'recovered-cases',graphWidth, graphHeight,logscale=True)
    logscale=True
    if logscale:
        plt.savefig('./Figures/'+country + '_fitted_log.png', dpi=100)
    else:
        plt.savefig('./Figures/' + country + '_fitted.png', dpi=100)


    # except:
    #     pass

# plt.close('all')  # clean up after using pyplot
