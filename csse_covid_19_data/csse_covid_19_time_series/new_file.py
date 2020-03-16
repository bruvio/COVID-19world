import plotly_express as px
import numpy as np
import pandas as pd

#Preprocessing
data = pd.read_csv('time_series_19-covid-Confirmed.csv')
timeline = ['1/22/20', '1/23/20',
       '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20',
       '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20',
       '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20',
       '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20',
       '2/19/20', '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20',
       '2/25/20', '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20',
       '3/2/20', '3/3/20'] 
#list of values to append equal to length of x axis  for plotly plot
#for eg x[t1,t2] = [[p1,p1],[20,30],[1000,5000]]
time = [];value = [];country=[];province= []
col_value = list(data.columns)
for i in range(len(data)):
    row_value = list(data.iloc[i])
    D = dict(zip(col_value,row_value))
    time.extend(timeline)
    value.extend(D[t] for t in  timeline)
    country.extend(D['Country/Region'] for i in  range(len(timeline)))
newdf = pd.DataFrame({'Timeline':time,'Covid-19 impact':value,'Country':country})

td  = pd.read_csv('time_series_19-covid-Confirmed.csv')
td.head()
newdf['Covid-19 impact'].replace({0:np.nan})
newdf.head()
newdf.to_csv('timelinedata.csv')
fig = px.scatter(newdf, x="Timeline", y="Covid-19 impact", color = 'Country',title = 'Spread of Covid-19 across countries',width=1000)
fig.show()
