import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
import pdb
url = 'https://gist.githubusercontent.com/johnburnmurdoch/4199dbe55095c3e13de8d5b2e5e5307a/raw/fa018b25c24b7b5f47fd0568937ff6c04e384786/city_populations'
df = pd.read_csv(url, usecols=['name', 'group', 'year', 'value'])

import random


def colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r, g, b))
    return ret


confirmed_df = pd.read_csv("time_series_19-covid-Confirmed.csv")
deaths_df = pd.read_csv("time_series_19-covid-Deaths.csv")
recovered_df = pd.read_csv("time_series_19-covid-Recovered.csv")


input_country_list = list(confirmed_df["Country/Region"])
input_country_list = [element.upper() for element in input_country_list]
#


# print(confirmed_df.columns)


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


print(confirmed_df.head(3))
print(confirmed_df_reshaped.head(3))
print(df.head(3))
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
# pdb.set_trace()
colors = dict(zip(
    countrylist,
    colors(len(countrylist)
)))
# pdb.set_trace()
group_lk = confirmed_df_reshaped['Country/Region'].tolist()

# group_lk1 = df.set_index('name')['group'].to_dict()
# print(colors[group_lk['Iran']])
df = confirmed_df_reshaped


fig, ax = plt.subplots(figsize=(15, 8))

def draw_barchart(current_date):
    dff = df[df['Date'].eq(current_date)]
    ax.clear()
    ax.barh(dff['Country/Region'], dff['cases'], color=[colors[group_lk[x]] for x in dff['Country/Region']])
    dx = dff['Country/Region'].max() / 200
    for i, (value, name) in enumerate(zip(dff['cases'], dff['Country/Region'])):
        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')
        ax.text(value-dx, i-.25, group_lk[name], size=10, color='#444444', ha='right', va='baseline')
        ax.text(value+dx, i,     float(value),  size=14, ha='left',  va='center')
    ax.text(1, 0.4, current_date, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    # ax.text(0, 1.06, 'Population (thousands)', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    # ax.text(0, 1.15, 'The most populous cities in the world from 1500 to 2018',
    #         transform=ax.transAxes, size=24, weight=600, ha='left', va='top')
    # ax.text(1, 0, 'by @pratapvardhan; credit @jburnmurdoch', transform=ax.transAxes, color='#777777', ha='right',
    #         bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)
    
draw_barchart(1/31/20)
plt.show()
#
# fig, ax = plt.subplots(figsize=(15, 8))
# animator = animation.FuncAnimation(fig, draw_barchart, frames=range(1900, 2019))
# HTML(animator.to_jshtml())

#
# with plt.xkcd():
#     fig, ax = plt.subplots(figsize=(15, 8))
#     draw_barchart(2018)
#
# current_year = 2018
# dff = df[df['year'].eq(current_year)].sort_values(by='value', ascending=False).head(10)
# dff
#
#
# fig, ax = plt.subplots(figsize=(15, 8))
# ax.barh(dff['name'], dff['value'])
#
# fig, ax = plt.subplots(figsize=(15, 8))
# dff = dff[::-1]
# ax.barh(dff['name'], dff['value'], color=[colors[group_lk[x]] for x in dff['name']])
# for i, (value, name) in enumerate(zip(dff['value'], dff['name'])):
#     ax.text(value, i,     name,            ha='right')
#     ax.text(value, i-.25, group_lk[name],  ha='right')
#     ax.text(value, i,     value, ha='left')
# ax.text(1, 0.4, current_year, transform=ax.transAxes, size=46, ha='right')
#
# fig, ax = plt.subplots(figsize=(15, 8))
# draw_barchart(2018)
