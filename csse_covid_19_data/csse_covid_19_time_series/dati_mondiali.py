import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import logging
import warnings
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
label_size = 8
mpl.rcParams['xtick.labelsize'] = label_size
from scipy.optimize import curve_fit
import pycountry as pc

logger = logging.getLogger(__name__)
# mondiale_df = pd.read_csv("who_covid_19_sit_rep_time_series.csv",parse_dates=['data'],
#                                 index_col=['data'])

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


# print(confirmed_df.head(5))
print(confirmed_df.columns)
# for col in confirmed_df.columns:
#     if confirmed_df[col].dtype == 'object':
#         try:
#             confirmed_df[col] = pd.to_datetime(confirmed_df[col])
#         except ValueError:
#             pass
# print(popt,pcov)

# print(confirmed_df.head(5))
# confirmed_df.dtypes

confirmed_df_reshaped = confirmed_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],var_name="Date", value_name='cases')
# confirmed_df_reshaped.set_index('Date', inplace=True)
deaths_df_reshaped = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],var_name="Date", value_name='cases')
# deaths_df_reshaped.set_index('Date', inplace=True)
recovered_df_reshaped = recovered_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],var_name="Date", value_name='cases')
# recovered_df_reshaped.set_index('Date', inplace=True)

active_df_reshaped = confirmed_df_reshaped
active_df_reshaped['cases'] = confirmed_df_reshaped['cases'] - recovered_df_reshaped['cases']

# print(confirmed_df_reshaped.head(15))
# confirmed_df_reshaped.dtypes


# confirmed_df_reshaped[confirmed_df_reshaped['Country/Region']=="Australia"]



# confirmed_df_reshaped_bycountry = confirmed_df_reshaped.groupby(confirmed_df_reshaped['Country/Region'])
# confirmed_df_reshaped_bycountry = confirmed_df_reshaped.groupby(confirmed_df_reshaped['Date'])

def exp_func(x, a, b ):
    return a*np.exp(b*x)

def func(x, a, b, c): # Hill sigmoidal equation from zunzun.com
    return  a * np.power(x, b) / (np.power(c, b) + np.power(x, b))


def print_Data_by_country(country,dataframe,label,show_fit=False):
    try:
        # dataframe_bycountry = dataframe.groupby(dataframe['Country/Region'])
        # country_database = dataframe[dataframe['Country/Region']==country]
        country_database = dataframe[dataframe['Country/Region'].str.contains(country)]
        country_database = country_database.reset_index(drop=True)

        df_bydate_grouper = country_database.groupby(country_database['Date']).sum()

        # df_bydate = df_bydate_grouper['sumbydate'].sum().to_frame(name='sum').reset_index()

        # df_bydate_grouper.set_index('Date', inplace=True)
        totale_casi = np.array(df_bydate_grouper['cases'])

        data = df_bydate_grouper.index.values
        # convert date to unix time for fit use
        # dates = pd.to_numeric(data)
        x = np.arange(0, len(totale_casi))
        popt1, pcov1 = curve_fit(lambda t, a, b: a * np.exp(b * t), x, totale_casi)
        plt.figure()
        plt.plot(data, totale_casi, 'ko', label="Original Data - "+country+' - '+label)
        if show_fit:
            plt.plot(data, exp_func(x, *popt1), 'r-', label="Fitted Curve -"+country+' - '+label)
        plt.xticks(rotation=15, ha="right")
        plt.legend(loc='best')
        return df_bydate_grouper
    except:
        logger.info('country {} not in database'.format(country))
        return 0

def ModelAndScatterPlot(country,dataframe,label,graphWidth, graphHeight):
    # these are the same as the scipy defaults
    initialParameters = np.array([1.0, 1.0, 1.0])

    # do not print unnecessary warnings during curve_fit()
    warnings.filterwarnings("ignore")

    country_database = dataframe[dataframe['Country/Region'].str.contains(country)]
    country_database = country_database.reset_index(drop=True)

    df_bydate_grouper = country_database.groupby(country_database['Date']).sum()

    # df_bydate = df_bydate_grouper['sumbydate'].sum().to_frame(name='sum').reset_index()

    # df_bydate_grouper.set_index('Date', inplace=True)
    totale_casi = np.array(df_bydate_grouper['cases'])

    data = df_bydate_grouper.index.values
    # convert date to unix time for fit use
    # dates = pd.to_numeric(data)
    x = np.arange(0, len(totale_casi))

    # curve fit the test data
    fittedParameters, pcov = curve_fit(func, x, totale_casi, initialParameters)

    modelPredictions = func(x, *fittedParameters)

    absError = modelPredictions - totale_casi

    SE = np.square(absError)  # squared errors
    MSE = np.mean(SE)  # mean squared errors
    RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(totale_casi))

    print('Parameters:', fittedParameters)
    print('RMSE:', RMSE)
    print('R-squared:', Rsquared)
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    axes = f.add_subplot(111)

    # first the raw data as a scatter plot
    axes.plot(data, totale_casi,  'D')

    # create data for the fitted equation plot
    xModel = np.linspace(min(x), max(x))
    yModel = func(xModel, *fittedParameters)

    # now the model as a line plot
    axes.plot(xModel, yModel)

    axes.set_xlabel('X Data') # X axis data label
    axes.set_ylabel('Y Data') # Y axis data label






# print('size of data is {}'.format(df_bydate_grouper_country.shape))

#
# confirmed_df_reshaped_bycountry_date = confirmed_df_reshaped_bycountry.groupby(confirmed_df_reshaped_bycountry['Date'])
#
# confirmed_df_reshaped.loc[df['Date'] == 1, 'b'].sum()
#
#
#







# graphics output section


# df_bydate_grouper_country = print_Data_by_country('Australia',confirmed_df_reshaped,'confirmed-cases')
df_bydate_grouper_country = print_Data_by_country('Italy',active_df_reshaped,'active-cases',show_fit=True)
# df_bydate_grouper_country = print_Data_by_country('US',active_df_reshaped,'active-cases',show_fit=False)
# df_bydate_grouper_country = print_Data_by_country('China',active_df_reshaped,'active-cases',show_fit=False)
# df_bydate_grouper_country = print_Data_by_country('France',active_df_reshaped,'active-cases',show_fit=False)
# df_bydate_grouper_country = print_Data_by_country('Australia',active_df_reshaped,'active-cases',show_fit=False)

graphWidth = 800
graphHeight = 600
ModelAndScatterPlot('Italy',active_df_reshaped,'active-cases',graphWidth, graphHeight)

plt.show()
plt.close('all')  # clean up after using pyplot