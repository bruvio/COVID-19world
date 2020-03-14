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

active_df_reshaped = confirmed_df_reshaped.copy()
active_cases = confirmed_df_reshaped['cases'] - recovered_df_reshaped['cases']
active_df_reshaped['cases'] = active_cases


countrylist_df = list(set(confirmed_df_reshaped['Country/Region']))



countrylist = ['United Kingdom','US','Germany','Italy','Mainland China','Singapore','Australia','France','Switzerland','Iran','Korea, South']

# for country in countrylist:
#     if 'Korea' in country.split():
#         print(country)

def exp_func(x,  b ):
    return np.exp(b*x)

def Hill_sigmoidal_func(x, a, b, c): # Hill sigmoidal equation from zunzun.com
    return  a * np.power(x, b) / (np.power(c, b) + np.power(x, b))

def func1(x, a, b,c,d):
        return d + ((a-d)/(1+(x/c)**b))

def func2(x, a,b,c):
    return a/(1+np.exp(-b*(x-c)))


def sigmoidal_func(x,a,b,c):
    return a / (1 + np.exp(-c * (x - b)))

def extract_database(country,dataframe):
    country_database = dataframe[dataframe['Country/Region'] == country]
    country_database = country_database.reset_index(drop=True)
    df_bydate_grouper = country_database.groupby('Date', sort=False)["cases"].sum().reset_index(name='Total Cases')

    totale_casi = np.array(df_bydate_grouper['Total Cases'])

    # data = df_bydate_grouper.index.values

    x = np.arange(0, len(totale_casi))
    return df_bydate_grouper,x,totale_casi

def print_Data_by_country(country,dataframe,label,show_fit=False,logscale=False):
    try:
        # dataframe_bycountry = dataframe.groupby(dataframe['Country/Region'])
        country_database = dataframe[dataframe['Country/Region'] == country]
        country_database = country_database.reset_index(drop=True)
        df_bydate_grouper = country_database.groupby('Date', sort=False)["cases"].sum().reset_index(name='Total Cases')

        totale_casi = np.array(df_bydate_grouper['Total Cases'])

        # data = df_bydate_grouper.index.values

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

def fit_data(x,y,func):
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

        print('Parameters:', fittedParameters)
        print('RMSE:', RMSE)
        print('R-squared:', Rsquared)



        # create data for the fitted equation plot
        xModel = np.linspace(min(x), max(x))
        yModel = func(xModel, *fittedParameters)


        # plt.legend(loc='best')
        # plt.xticks(rotation=15, ha="right")

        return xModel,yModel,fittedParameters,Rsquared
    except:
        print('failed to fit data')
        return [],[],[],[]



def plot_data(Xdata,Ydata,country,label,color,logscale=False):


    # first the raw data as a scatter plot
    if logscale:
        plt.yscale('log')
        # plt.ylim(0, 1e4)
    # plt.plot(Xdata, Ydata, 'D', label="Original Data - " + country + ' - ' + label,color=color)
    plt.scatter(Xdata, Ydata, marker='D', label="Original Data - " + country + ' - ' + label,c=color)


def plot_model(xModel,yModel,country,label,color,marker,logscale=False):

        if logscale:
            plt.yscale('log')

        # plt.plot(xModel, yModel, label="Fitted Data - " + country + ' - ' + label, color=color,marker=marker)
        plt.scatter(xModel, yModel, label="Fitted Data - " + country + ' - ' + label, c=color,marker=marker)

        # plt.legend(loc='best')
        # plt.xticks(rotation=15, ha="right")









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

def recompute_fit(dataframe,label,country,newfit):
    x = np.arange(0, len(np.array(dataframe['Total Cases'])))
    xModel = np.linspace(min(x), max(x))
    yModel = func2(xModel, *newfit)

    # now the model as a line plot
    plt.plot(xModel, yModel, label="Fitted Data - " + country + ' - ' + label, color='m')

# graphWidth = 800
# graphHeight = 600

countrylist = ['Italy']
# countrylist = ['United Kingdom']
for country in countrylist:
    if country in countrylist_df:
        logger.info('plotting {} data'.format(country))
        # try:
        print(country)
        logscale= False
        databasename = 'confirmed cases'
        dataframe,x,y = extract_database(country,confirmed_df_reshaped)
        date = dataframe.index.values
        start_prediction_date=date
        stop_prediction_date=80
        date_prediction = np.arange(0,stop_prediction_date)
        xModel, yModel, fittedParameters_10, Rsquared = fit_data(x, y, sigmoidal_func)
        plt.figure(num=country+'_justfit')
        plot_data(date,y,country,'active','r',logscale=logscale)
        plot_model(xModel, yModel, country, ' fit - ' + databasename, 'g', marker='x', logscale=logscale)
        plt.show()

        plt.figure(num=country)
        xModel,yModel,fittedParameters_10,Rsquared = fit_data(x[-10:],y[-10:],exp_func)
        # xModel,yModel,fittedParameters_10,Rsquared = fit_data(x,y,exp_func)
        plt.figure(num=country)
        plot_data(date,y,country,'active','r',logscale=logscale)
        # if Rsquared > 0.8:
        # plot_model(xModel,yModel,country,' 11days-fit - '+databasename,'g',marker='x',logscale=logscale)
        # else:


        fittedParameters, pcov = curve_fit(sigmoidal_func, x, y, maxfev=5000,p0=[1,1,-fittedParameters_10[0]])

        print('Parameters:', fittedParameters)

        modelPredictions = sigmoidal_func(y, *fittedParameters)

        absError = modelPredictions[0:len(y)] - y

        SE = np.square(absError)  # squared errors
        MSE = np.mean(SE)  # mean squared errors
        RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (np.var(absError) / np.var(y))

        print('Parameters:', fittedParameters)
        print('RMSE:', RMSE)
        print('R-squared:', Rsquared)

        xModel = np.linspace(start_prediction_date, stop_prediction_date)
        yModel = sigmoidal_func(xModel, *fittedParameters)
        # if Rsquared > 0.5:
        plot_model(xModel, yModel, country, 'predictions - '+databasename, 'm',marker='.', logscale=logscale)
        # else:
        #     print('Model did not converge')

        plt.xlabel('days since it started') # X axis data label
        plt.ylabel('confirmed cases') # Y axis data label
        plt.legend(loc='best',fontsize='10')
        # plt.xticks(np.arange(22, 80, step=1))
        if logscale:
            plt.savefig('./Figures/' + country + '_fitted_log.png', dpi=100)
            plt.ylim(1,1e5)
        else:
            plt.savefig('./Figures/' + country + '_fitted.png', dpi=100)


#
plt.show()
