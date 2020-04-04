#!/usr/bin/env python
# coding: utf-8




from bs4 import BeautifulSoup
from requests import get
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math 
import os
import ast
import time
import glob
from datetime import date
pd.options.display.width = 0

# Function for remove comma within numbers
def removeCommas(string): 
    string = string.replace(',','')
    return string 
def removebraket1(string):
    string = string.replace('[','')
    return string
def removebraket2(string):
    string = string.replace(']','')
    return string
# # Scrap data from worldmeter




# Test if we can scrap info from worldometers
# The communication with website is ok if the response is 200
headers = ({'User-Agent':'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'})
worldometers = "https://www.worldometers.info/coronavirus/#countries"
response = get(worldometers, headers=headers)
response




# Scrap all content from the website
html_soup = BeautifulSoup(response.text, 'html.parser')
# After inspect the website content, data are stored inside tag 'tbody' and table header is 'thead'
table_contents = html_soup.find_all('tbody')
table_header = html_soup.find_all('thead')

# Header for the table
header = []
for head_title in table_header[0].find_all('th'):    
    header.append(str(head_title.contents))

# Save value into columns
CountryName = []
TotalCases = []
NewCases = []
TotalDeaths = []
NewDeaths = []
TotalRecovered = []
ActiveCases = []
SeriousCritical = []

for row in table_contents[0].find_all('tr'):
    cells = row.find_all('td')
    if len(cells[0].find_all('a')) >= 1:
        CountryName.append(cells[0].find_all('a')[0].contents[0])
    elif len(cells[0].find_all('span')) >= 1:
        CountryName.append(cells[0].find_all('span')[0].contents[0])
    else:
        CountryName.append(cells[0].contents[0])

    print(CountryName[-1])
    if len(cells[1].contents) >= 1:
        TotalCases.append(cells[1].contents[0])
    else:
        TotalCases.append(0)

    if len(cells[2].contents) >= 1:
        NewCases.append(cells[2].contents[0])
    else:
        NewCases.append(0)

    if len(cells[3].contents) >= 1:
        TotalDeaths.append(cells[3].contents[0])
    else:
        TotalDeaths.append(0)

    if len(cells[4].contents) >= 1:
        NewDeaths.append(cells[4].contents[0])
    else:
        NewDeaths.append(0)

    if len(cells[5].contents) >= 1:
        TotalRecovered.append(cells[5].contents[0])
    else:
        TotalRecovered.append(0)

    if len(cells[6].contents) >= 1:
        ActiveCases.append(cells[6].contents[0])
    else:
        ActiveCases.append(0)

    if len(cells[7].contents) >= 1:
        SeriousCritical.append(cells[7].contents[0])
    else:
        SeriousCritical.append(0)
    # if len(cells[1].contents) > 1:
    #     TotalCases.append(cells[1].contents[0])
    # # else:
    # #     TotalCases.append(0)
    # if len(cells[2].contents) > 1:
    #     NewCases.append(cells[2].contents[0])
    # # else:
    # #     NewCases.append(0)
    # if len(cells[3].contents) > 1:
    #     TotalDeaths.append(cells[3].contents[0])
    # # else:
    # #     TotalDeaths.append(0)
    # if len(cells[4].contents) > 1:
    #     NewDeaths.append(cells[4].contents[0])
    # # else:
    # #     NewDeaths.append(0)
    # if len(cells[5].contents) > 1:
    #     TotalRecovered.append(cells[5].contents[0])
    # # else:
    # #     TotalRecovered.append(0)
    # if len(cells[6].contents) > 1:
    #     ActiveCases.append(cells[6].contents[0])
    # # else:
    # #     ActiveCases.append(0)
    # if len(cells[7].contents) > 1:
    #     SeriousCritical.append(cells[7].contents[0])
    # # else:
    # #     SeriousCritical.append(0)
        
CaseTable = pd.DataFrame({header[0]: CountryName,
                          header[1]: TotalCases,
                          header[2]: NewCases,
                          header[3]: TotalDeaths,
                          header[4]: NewDeaths,                          
                          header[5]: TotalRecovered,
                          header[6]: ActiveCases,
                          header[7]: SeriousCritical,
                          })  

CaseTable.head(40)





CaseTable.tail(40)







caseTableSimple = CaseTable[[CaseTable.columns[0], CaseTable.columns[1], CaseTable.columns[3], CaseTable.columns[5]]]
caseTableSimple.columns = ['Country/Region', 'Confirmed', 'Deaths', 'Recovered']
# Set data type as string first for manuipulation
caseTableSimple = caseTableSimple.astype({'Country/Region':str,'Confirmed':str,'Deaths':str, 'Recovered':str})
# Remove the last row of total number (changed on 20200310, worldmeter moved this row as next tbody)
#caseTableSimple = caseTableSimple.iloc[:-1,:]
# Remove lead and tail space for each element
caseTableSimple = caseTableSimple.apply(lambda x: x.str.strip())
# Remove comma for each element
caseTableSimple = caseTableSimple.applymap(removeCommas)
# Replace empty str with zero. This include row of 'Diamond Princess' (its name is empty)
caseTableSimple = caseTableSimple.replace('', '0')
# After string manipulation, convert data type as correct type
caseTableSimple = caseTableSimple.astype({'Country/Region':'str',
                                          'Confirmed':'int',
                                          'Deaths':'int',
                                          'Recovered':'int',
                                         })

currentTime = datetime.now()
lastUpdateTime = currentTime.strftime('%m/%d/%Y %H:%M')
# Remove the first number (This only works for month number less than 10)

caseTableSimple['Last Update'] = lastUpdateTime



# Change Country name the same as my old data
if 'S. Korea' in list(caseTableSimple['Country/Region']):
    caseTableSimple = caseTableSimple.replace('S. Korea', 'South Korea')

if 'UK' in list(caseTableSimple['Country/Region']):
    caseTableSimple = caseTableSimple.replace('UK', 'United Kingdom')

if 'USA' in list(caseTableSimple['Country/Region']):
    caseTableSimple = caseTableSimple.replace('USA', 'US')

# In my old data, I used 'United Arab Emirates' not 'UAE'
if 'UAE' in list(caseTableSimple['Country/Region']):
    caseTableSimple['Country/Region'].replace({'UAE': 'United Arab Emirates'}, inplace=True)

if 'Réunion' in list(caseTableSimple['Country/Region']):
    caseTableSimple['Country/Region'].replace({'Réunion': 'Reunion'}, inplace=True)

if 'Curaçao' in list(caseTableSimple['Country/Region']):
    caseTableSimple['Country/Region'].replace({'Curaçao': 'Curacao'}, inplace=True)


timeStampe = currentTime.strftime('%m_%d_%Y_%H_%M')
caseTableSimple = caseTableSimple.sort_values(by=['Confirmed'], ascending=False)

worldometer_table = caseTableSimple.copy()

# worldometer_table.to_csv('./worldmeter_data/{}_webData.csv'.format(timeStampe), index=False)
print(caseTableSimple.head(10))
# raise SystemExit

# Data for these countries come from other source
removeRegion = ['China', 'Canada', 'Australia', 'US']
for i in removeRegion:
    caseTableSimple.drop(caseTableSimple[caseTableSimple['Country/Region'] == i].index, axis=0, inplace=True)



# Add column 'Province/State' with empty value
caseTableSimple['Province/State'] = ''

# In my old data, 'Diamond Princess' is represented by 'Yokohama' in the column of 'Province/State'
if 'Diamond Princess' in list(caseTableSimple['Country/Region']):
    caseTableSimple.at[caseTableSimple.loc[caseTableSimple[
                                               'Country/Region'] == 'Diamond Princess',].index, 'Province/State'] = 'Yokohama'
    caseTableSimple['Country/Region'].replace({'Diamond Princess': 'Japan'}, inplace=True)

# In my old data, 'Belgium' has 'Brussels' in the column of 'Province/State'
if 'Belgium' in list(caseTableSimple['Country/Region']):
    caseTableSimple.at[
        caseTableSimple.loc[caseTableSimple['Country/Region'] == 'Belgium',].index, 'Province/State'] = 'Brussels'

# In my old data, I used 'Macau' not 'Macao'
if 'Macao' in list(caseTableSimple['Country/Region']):
    caseTableSimple.at[
        caseTableSimple.loc[caseTableSimple['Country/Region'] == 'Macao',].index, 'Province/State'] = 'Macau'
    caseTableSimple['Country/Region'].replace({'Macao': 'Macau'}, inplace=True)

# In my old data, 'Hong Kong' has 'Hong Kong' in the column of 'Province/State'
if 'Hong Kong' in list(caseTableSimple['Country/Region']):
    caseTableSimple.at[
        caseTableSimple.loc[caseTableSimple['Country/Region'] == 'Hong Kong',].index, 'Province/State'] = 'Hong Kong'

# In my old data, 'Taiwan' has 'Taiwan' in the column of 'Province/State'
if 'Taiwan' in list(caseTableSimple['Country/Region']):
    caseTableSimple.at[
        caseTableSimple.loc[caseTableSimple['Country/Region'] == 'Taiwan',].index, 'Province/State'] = 'Taiwan'


# In my old data I used US time as Last Update time
currentTime = datetime.now()
lastUpdateTime = currentTime.strftime('%m/%d/%Y %H:%M')
# Remove the first number (This only works for month number less than 10)
lastUpdateTime[1:]
caseTableSimple['Last Update'] = lastUpdateTime

# Reorder list as all old data
columnList = caseTableSimple.columns.tolist()
columnList = [columnList[i] for i in [4, 0, 5, 1, 2, 3]]
caseTableSimple = caseTableSimple[columnList]




caseTableSimple.tail(20)


# # Scrap data for US_CAN



try:
# As the website changed to dynamic, using selenium to interact with the website vitually
    from selenium import webdriver

    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    # Open vitual Chrome browser


    driver = webdriver.Chrome()
    # Direct the driver to open a webpage by calling the ‘get’ method, with a parameter of the page we want to visit.
    driver.get("https://coronavirus.1point3acres.com/en")
    # click tab button to let page lode new data (US data is the default)
    python_button = driver.find_element(By.XPATH, "//span[text()='Canada']")
    python_button.click()

    # html_soup22 = BeautifulSoup(driver.page_source)
    html_soup2 = BeautifulSoup(driver.page_source,  'html.parser')


    driver.get("https://coronavirus.1point3acres.com/en")
    # click tab button to let page lode new data (US data is the default)
    python_button = driver.find_element(By.XPATH, "//span[text()='United States']")
    python_button.click()

    # html_soup22 = BeautifulSoup(driver.page_source)
    html_soup22 = BeautifulSoup(driver.page_source,  'html.parser')

    # Wait for the dynamically loaded elements to show up
    # WebDriverWait(driver, 10).until(
    #     EC.visibility_of_element_located((By.CLASS_NAME, CANindex)))
    # # And grab the new page HTML source
    html_page = driver.page_source
    driver.quit()
    indexList = []
    for span in html_soup2.find_all('span'):
        # Only retain 'span' that has contents
        if len(span.contents):
            # Since we only need to find index for table, use one of the table head as target word to locate index
            if span.contents[0] == 'Location':
                # Store the index inside a list
                indexList.append(span['class'][0])

    indexList1 = []
    for span in html_soup22.find_all('span'):
        # Only retain 'span' that has contents
        if len(span.contents):
            # Since we only need to find index for table, use one of the table head as target word to locate index
            if span.contents[0] == 'Location':
                # Store the index inside a list
                indexList1.append(span['class'][0])

except:
    print('failed to connect to US Canada database')

print('scraping Canadian data')
try:

    # USindex, CANindex = indexList
    CANindex = indexList[0]
    Locations = []
    Confirmed = []
    Recovered = []
    Deaths = []
    list1 = range(0, len(html_soup2.find_all('span', class_=CANindex)) - 4, 5)
    list2 = range(1, len(html_soup2.find_all('span', class_=CANindex)) - 3, 5)
    list3 = range(2, len(html_soup2.find_all('span', class_=CANindex)) - 2, 5)
    list4 = range(4, len(html_soup2.find_all('span', class_=CANindex)) - 0, 5)

    for index in list1:
        if len(html_soup2.find_all('span', class_=CANindex)[index].contents):
            Locations.append(html_soup2.find_all('span', class_=CANindex)[index].contents[0])
        else:
            Locations.append(0)
    for index in list2:
        if len(html_soup2.find_all('span', class_=CANindex)[index].contents):
            try:
                Confirmed.append(html_soup2.find_all('span', class_=CANindex)[index].contents[1])
            except:
                Confirmed.append(html_soup2.find_all('span', class_=CANindex)[index].contents[0])
        else:
            Confirmed.append(0)
    for index in list3:
        # . They do not provide Recovered cases number
        if len(html_soup2.find_all('span', class_=CANindex)[index].contents):
           Deaths.append(html_soup2.find_all('span', class_=CANindex)[index].contents[0])
        else:
            Deaths.append(0)
    for index in list4:
        if len(html_soup2.find_all('span', class_=CANindex)[index].contents):
            try:
                Recovered.append(html_soup2.find_all('span', class_=CANindex)[index].contents[1])
            except:
                Recovered.append(html_soup2.find_all('span', class_=CANindex)[index].contents[0])
        else:
            Recovered.append(0)

    CAN_data = pd.DataFrame({'Province/State': Locations,
                             'Confirmed': Confirmed,
                             'Deaths': Deaths,
                             'Recovered':Recovered,
                             })

    # Remove rows that are not data
    CAN_data.drop(CAN_data[CAN_data['Confirmed'] == 'Confirmed'].index, axis=0, inplace=True)
    CAN_data.drop(CAN_data[CAN_data['Recovered'] == 'Recovered'].index, axis=0, inplace=True)
    CAN_data.drop(CAN_data[CAN_data['Deaths'] == 'Deaths'].index, axis=0, inplace=True)

    # Remove rows that are not data
    CAN_data.drop(CAN_data[CAN_data['Province/State'] == 'Canada'].index, axis=0, inplace=True)

    # Locations1 = []
    # for name in CAN_data['Province/State']:
    #     name.parent.find('div', class_=CANindex).get_text().strip()
    #     Locations1.append(name)
    # Locations1 = []
    # for index,row in CAN_data.iterrows():
    #     Locations1.append(row['Province/State'].iloc[index].parent.find('div', class_=CANindex).get_text().strip()

    Locations1 = []
    for index in range(0,len(CAN_data['Province/State'])):
        Locations1.append(CAN_data['Province/State'].iloc[index].parent.find('div', class_=CANindex).get_text().strip())

    CAN_data['Province/State'] = Locations1
        # Remove comma for each element
    CAN_data['Confirmed'] = CAN_data['Confirmed'].apply(removeCommas)
    CAN_data['Recovered'] = CAN_data['Recovered'].apply(removeCommas)
    CAN_data['Deaths'] = CAN_data['Deaths'].apply(removeCommas)
    CAN_total_confirmed = CAN_data['Confirmed'].astype(int).sum()
    CAN_total_deaths =CAN_data['Deaths'].astype(int).sum()
    CAN_total_recovered =CAN_data['Recovered'].astype(int).sum()
    print(CAN_data.head(10))
except:
    print('failed to scrap Canada data')

print('scraping US data')
try:
    html_soup2 = html_soup22
    USindex = indexList1[0]
    Locations = []
    Confirmed = []
    Recovered = []
    Deaths = []
    list1 = range(1, len(html_soup2.find_all('span', class_=USindex)) - 4, 5)
    list2 = range(2, len(html_soup2.find_all('span', class_=USindex)) - 3, 5)
    list3 = range(3, len(html_soup2.find_all('span', class_=USindex)) - 2, 5)
    list4 = range(5, len(html_soup2.find_all('span', class_=USindex)) - 0, 5)

    for index in list1:
        if len(html_soup2.find_all('span', class_=USindex)[index].contents):
            Locations.append(html_soup2.find_all('span', class_=USindex)[index].contents[0])
        else:
            Locations.append(0)
    for index in list2:
        if len(html_soup2.find_all('span', class_=USindex)[index].contents):
            try:
                Confirmed.append(html_soup2.find_all('span', class_=USindex)[index].contents[1])
            except:
                Confirmed.append(html_soup2.find_all('span', class_=USindex)[index].contents[0])
        else:
            Confirmed.append(0)
    for index in list3:
        # They do not provide Recovered cases number anymore.
        if len(html_soup2.find_all('span', class_=USindex)[index].contents):
           Deaths.append(html_soup2.find_all('span', class_=USindex)[index].contents[0])
        else:
            Deaths.append(0)
    for index in list4:
        if len(html_soup2.find_all('span', class_=USindex)[index].contents):
            try:
                Recovered.append(html_soup2.find_all('span', class_=USindex)[index].contents[1])
            except:
                Recovered.append(html_soup2.find_all('span', class_=USindex)[index].contents[0])
        else:
            Recovered.append(0)

    US_data = pd.DataFrame({'Province/State': Locations,
                            'Confirmed': Confirmed,
                            'Deaths': Deaths,
                            'Recovered':Recovered,
                            })

    # Remove rows that are not data
    US_data.drop(US_data[US_data['Confirmed'] == 'Confirmed'].index, axis=0, inplace=True)
    US_data.drop(US_data[US_data['Deaths'] == 'Deaths'].index, axis=0, inplace=True)
    US_data.drop(US_data[US_data['Recovered'] == 'Recovered'].index, axis=0, inplace=True)

    # Remove rows that are not data
    US_data.drop(US_data[US_data['Province/State'] == 'United States'].index, axis=0, inplace=True)

    # Replace Washington, D.C. as Washington DC
    if 'Washington, D.C.' in list(US_data['Province/State']):
        US_data['Province/State'].replace({'Washington, D.C.': 'Washington DC'}, inplace=True)

    # Replace Washington as WA
    if 'Washington' in list(US_data['Province/State']):
        US_data['Province/State'].replace({'Washington': 'WA'}, inplace=True)

    # Replace Grand Princess as From Grand Princess
    # if 'Grand Princess' in list(US_data['Province/State']):
    #    US_data['Province/State'].replace({'Grand Princess':'From Grand Princess'}, inplace=True)

    # Replace Diamond Princess as From Diamond Princess cruise
    # if 'Diamond Princess' in list(US_data['Province/State']):
    #    US_data['Province/State'].replace({'Diamond Princess':'From Diamond Princess cruise'}, inplace=True)

    # Assign 0 in column Province/State as unassigned
    if 0 in list(US_data['Province/State']):
        US_data.at[US_data.loc[US_data['Province/State'] == 0,].index, 'Province/State'] = 'Unassigned'

    Locations1 = []
    for index in range(0,len(US_data['Province/State'])):
        Locations1.append(US_data['Province/State'].iloc[index].parent.find('div', class_=USindex).get_text().strip())

    US_data['Province/State'] = Locations1

    # Remove comma for each element
    US_data['Confirmed'] = US_data['Confirmed'].apply(removeCommas)
    US_data['Recovered'] = US_data['Recovered'].apply(removeCommas)
    US_data['Deaths'] = US_data['Deaths'].apply(removeCommas)



    USA_total_confirmed = US_data['Confirmed'].astype(int).sum()
    USA_total_deaths =US_data['Deaths'].astype(int).sum()
    print(US_data.head(10))
except:
    print('failed to scrap US data')

print('merging US and Canadian data')
try:
    US_Can_data = pd.concat([US_data, CAN_data], ignore_index=True)
    US_Can_data = US_Can_data.apply(lambda x: x.str.strip())
    US_Can_data

    # In[252]:


    nameList = pd.read_csv('./statesNameTranslation.csv')




    US_Can_data_EN = pd.merge(US_Can_data, nameList, how = 'left', left_on = 'Province/State', right_on = 'English')
    US_Can_data_EN = US_Can_data_EN.drop(['Chinese', 'Province/State', 'Abbr.'], axis=1)
    US_Can_data_EN['Last Update'] = lastUpdateTime
    US_Can_data_EN.rename(columns={'English':'Province/State'}, inplace=True)
    US_Can_data_EN = US_Can_data_EN.drop(US_Can_data_EN[US_Can_data_EN['Province/State'] == 'Wuhan Evacuee'].index, axis=0)
    columnOrder = ['Province/State', 'Country/Region', 'Last Update','Confirmed', 'Deaths', 'Recovered']
    # columnOrder1 = ['Province/State', 'Country/Region', 'Last Update','Confirmed', 'Deaths', 'Recovered_x']

    # US_Can_data_EN = US_Can_data_EN[columnOrder1]
    # US_Can_data_EN
    US_Can_data_EN = pd.DataFrame({'Province/State': US_Can_data_EN['Province/State'],
                            'Confirmed': US_Can_data_EN['Confirmed'],
                            'Deaths': US_Can_data_EN['Deaths'],
                            'Recovered':US_Can_data_EN['Recovered_x'],
                            })





    caseTableSimple = pd.concat([US_Can_data_EN, caseTableSimple], ignore_index=True)
# finalTable

except:
    print('failed to merge US and Canada  data')


timeStampe = currentTime.strftime('%m_%d_%Y_%H_%M')
# caseTableSimple.to_csv('./web_data/{}_webData.csv'.format(timeStampe), index=False)

# # Scrap data for China

print('scraping Chinese data')
try:

    CHN = "https://ncov.dxy.cn/ncovh5/view/pneumonia?scene=2&clicktime=1579582238&enterid=1579582238&from=singlemessage&isappinstalled=0"


    res = get('https://ncov.dxy.cn/ncovh5/view/pneumonia?scene=2&clicktime=1579582238&enterid=1579582238&from=singlemessage&isappinstalled=0')
    res.encoding = "GBK"
    r = res.text
    p = re.compile(r'window\.getAreaStat = \[(.*?)\]}catch')
    data = p.findall(r)[0]
    print(ast.literal_eval(data))

    regext = ast.literal_eval(data)

    dataframe = pd.DataFrame(list(regext))

    dataframe = dataframe[['provinceName', 'provinceShortName', 'currentConfirmedCount',
           'confirmedCount', 'suspectedCount', 'curedCount', 'deadCount']]
    dataframe['Last Update'] = lastUpdateTime

    dataframe = pd.DataFrame({'Province/State': dataframe['provinceName'],
                              'Country/Region': 'China',
                              'Last Update': lastUpdateTime,
                            'Confirmed': dataframe['confirmedCount'],
                            'Deaths': dataframe['deadCount'],
                            'Recovered': dataframe['curedCount'],
                            })
    dataframe = dataframe[columnOrder]



    from china_cities import *
    from googletrans import Translator
    translator = Translator()

    provinceNames = []
    for province in dataframe['Province/State']:
            trans = translator.translate(province, dest='en').text
            provinceNames.append(trans)
    dataframe['Province/State']  = provinceNames
    CHN_data = dataframe
    CHN_total_confirmed = CHN_data['Confirmed'].sum()
    CHN_total_deaths =CHN_data['Deaths'].sum()
    CHN_total_recovered =CHN_data['Recovered'].sum()
    print(CHN_data.head(10))


    caseTableSimple = pd.concat([CHN_data, caseTableSimple], ignore_index=True)
except:
    print('failed to scrap Chinese data')


try:
    Australia = "https://www.health.gov.au/news/health-alerts/novel-coronavirus-2019-ncov-health-alert/coronavirus-covid-19-current-situation-and-case-numbers"
    response2 = get(Australia, headers=headers)
    response2

    # Scrap all content from the website
    html_soup2 = BeautifulSoup(response2.text, 'html.parser')

    confirmed_cases = []

    for name in  html_soup2.find_all('td', class_='numeric'):
        salary = name.parent.find_all('td')[-1]  # last cell in the row
        # value = int(name.get_text())
        # print(name.get_text())
        # print(salary.get_text())
        # print(value)
        confirmed_cases.append(int(salary.get_text().strip().replace(',','')))
    confirmed_cases = confirmed_cases[:-1]




    a = html_soup2.find("div", {"class":"health-table__responsive"}).findAll('p')
    locations = []
    for index, value in enumerate(a):
        # print(index,value)
        if "numeric" in str(value):
            continue
        else:

            dummy = str(a[index - 1]).replace('<p>', '')
            dummy = dummy.replace('</p>', '')
            dummy = dummy.replace('<span>', '')
            dummy = dummy.replace('</span>', '')

            locations.append(dummy)


    locations = [locations[2]] +locations[4:]

    AUS_df = pd.DataFrame(list(zip(locations,confirmed_cases)),columns = ['Province/State','Confirmed'])
    AUS_df['Last Update'] = lastUpdateTime
    AUS_df = pd.DataFrame({'Province/State': AUS_df['Province/State'],
                              'Country/Region': 'Australia',
                              'Last Update': lastUpdateTime,
                            'Confirmed': AUS_df['Confirmed'],
                            # 'Deaths': AUS_df['deadCount'],
                            # 'Recovered': AUS_df['curedCount'],
                            })
    AUS_df = AUS_df[columnOrder[:-2]]

    AUS_total_confirmed =AUS_df['Confirmed'].sum()

    print(AUS_df.head(10))

    caseTableSimple = pd.concat([AUS_df, caseTableSimple], ignore_index=True)
except:
    print('failed to scrape Australian data')




# write to file table
caseTableSimple.to_csv('./web_data/{}_webData.csv'.format(timeStampe), index=False)


us_index = worldometer_table.index[worldometer_table['Country/Region']=='US'].tolist()[0]
china_index = worldometer_table.index[worldometer_table['Country/Region']=='China'].tolist()[0]
aus_index = worldometer_table.index[worldometer_table['Country/Region']=='Australia'].tolist()[0]
canada_index = worldometer_table.index[worldometer_table['Country/Region']=='Canada'].tolist()[0]

try:
    print(worldometer_table['Confirmed'].loc[us_index] ,USA_total_confirmed)
    print(worldometer_table['Confirmed'].loc[china_index] ,CHN_total_confirmed)
    print(worldometer_table['Recovered'].loc[china_index] ,CHN_total_recovered)
    print(worldometer_table['Confirmed'].loc[aus_index] , AUS_total_confirmed)
    print(worldometer_table['Confirmed'].loc[canada_index] ,CAN_total_confirmed)
    print(worldometer_table['Deaths'].loc[us_index] , USA_total_deaths)
    print(worldometer_table['Deaths'].loc[china_index] ,CHN_total_deaths)
    print(worldometer_table['Deaths'].loc[canada_index] ,CAN_total_deaths)
except:
    pass



#  updating worldmeter table with most recend data from different sources
try:
    worldometer_table.loc[worldometer_table['Country/Region'] == 'US', 'Confirmed'] = USA_total_confirmed
    worldometer_table.loc[worldometer_table['Country/Region'] == 'US', 'Deaths'] = USA_total_deaths

    worldometer_table.loc[worldometer_table['Country/Region'] == 'Canada', 'Confirmed'] = CAN_total_confirmed
    worldometer_table.loc[worldometer_table['Country/Region'] == 'Canada', 'Deaths'] = CAN_total_deaths

    worldometer_table.loc[worldometer_table['Country/Region'] == 'China', 'Confirmed'] = CHN_total_confirmed
    worldometer_table.loc[worldometer_table['Country/Region'] == 'China', 'Recovered'] = CHN_total_recovered
    worldometer_table.loc[worldometer_table['Country/Region'] == 'China', 'Deaths'] = CHN_total_deaths

    worldometer_table.loc[worldometer_table['Country/Region'] == 'Australia', 'Confirmed'] = AUS_total_confirmed
except:
    pass


# print(worldometer_table[worldometer_table['Country/Region'] == 'Australia', 'Confirmed'],AUS_total_confirmed)
# print(worldometer_table[worldometer_table['Country/Region'] == 'US', 'Confirmed'],USA_total_confirmed)
# print(worldometer_table[worldometer_table['Country/Region'] == 'Canada', 'Confirmed'],CAN_total_confirmed)
# print(worldometer_table[worldometer_table['Country/Region'] == 'China', 'Confirmed'],CHN_total_confirmed)


# load previous worldometer table
# list_of_files = glob.glob('worldmeter_data/*')  # * means all if need specific format then *.csv
# latest_file = max(list_of_files, key=os.path.getctime)
list_of_files = []
now = time.time()

for f in os.listdir('./worldmeter_data/'):
    # mtime = path.stat().st_mtime
    # timestamp_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d-%H:%M')
    if os.stat(os.path.join('./worldmeter_data/',f)).st_mtime < now - 0.5 * 86400:
        list_of_files.append(f)

list_of_files.sort(reverse=True)
latest_file = list_of_files[0]
print('reading {} data'.format(latest_file))
previous_worldometer_table = pd.read_csv('./worldmeter_data/'+   latest_file)

# previous_worldometer_table['increase_confirmed'] = np.where(worldometer_table['Confirmed'] == previous_worldometer_table['Confirmed'], 0, worldometer_table['Confirmed'] - previous_worldometer_table['Confirmed']) #create new column in df1 for price diff
# previous_worldometer_table['increase_deaths'] = np.where(worldometer_table['Deaths'] == previous_worldometer_table['Deaths'], 0, worldometer_table['Deaths'] - previous_worldometer_table['Deaths']) #create new column in df1 for price diff
# previous_worldometer_table['increase_recovered'] = np.where(worldometer_table['Recovered'] == previous_worldometer_table['Recovered'], 0, worldometer_table['Recovered'] - previous_worldometer_table['Recovered']) #create new column in df1 for price diff
# print(previous_worldometer_table.head(20))
# writing table to csv
worldometer_table.to_csv('./worldmeter_data/{}_webData.csv'.format(timeStampe), index=False)


try:
    comparison_df = worldometer_table.merge(previous_worldometer_table,
                                  indicator=True
                                  ,left_on='Country/Region',right_on='Country/Region',suffixes=('_left', '_right'))



    comparison_df = pd.DataFrame(
        {'Country/Region': comparison_df['Country/Region'],
         'Confirmed_diff': comparison_df['Confirmed_left'] - comparison_df['Confirmed_right'],
         'Deaths_diff': comparison_df['Deaths_left'] - comparison_df['Deaths_right'],
         'Recovered_diff': comparison_df['Recovered_left'] - comparison_df['Recovered_right'],
         'date_diff': pd.to_datetime(comparison_df['Last Update_left']) - pd.to_datetime(comparison_df['Last Update_right']),
         })

    today = date.today()
    comparison_df = comparison_df.sort_values(by=['Confirmed_diff'], ascending=False)



    comparison_df.to_csv('./daily_diff/{}_diff.csv'.format(today))
    print(comparison_df.head(20))
    # diff_df = dataframe_difference(worldometer_table,previous_worldometer_table,'Confirmed',which='left_only')
    # diff_df = dataframe_difference(worldometer_table,previous_worldometer_table)
except:
    print('failed to compare with previous day')

