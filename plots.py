#!/Users/bruvio/python37/bin python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import os
import base64
import time
import glob
import plotly.express as px
import plotly.graph_objects as go
import plotly
import inspect
import random
# named_colorscales = px.colors.named_colorscales()
named_colorscales = []
import matplotlib
for name, hex in matplotlib.colors.cnames.items():
    # print(name, hex)
    named_colorscales.append(name)
random.shuffle(named_colorscales)


# colorscale_names = []
# colors_modules = ['carto', 'colorbrewer', 'cmocean', 'cyclical',
#                     'diverging', 'plotlyjs', 'qualitative', 'sequential']
# for color_module in colors_modules:
#     colorscale_names.extend([name for name, body
#                             in inspect.getmembers(getattr(px.colors, color_module))
#                             if isinstance(body, list)])

from textwrap import fill

# print(fill(''.join(sorted({f'{x: <{15}}' for x in colorscale_names})), 75))
# field = 'Confirmed'
fields = []
fields.append('Confirmed')
fields.append('Deaths')
fields.append('Recovered')

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


num_of_elements = 10
# for field in fields:
df_frame = (
                    previous_worldometer_table.sort_values(by='Confirmed', ascending=True)
                    .tail(num_of_elements)
                )
df_frame = df_frame[df_frame["Country/Region"]!= "World"]


    # vars()['top_countries_'+field] = list(set(df_frame["Country/Region"][0:9]))










# Create empty figure canvas
for field in fields:
    # Create empty figure canvas
    fig_confirmed = go.Figure()

    # Add trace to the figure;
    for index,country in enumerate(df_frame["Country/Region"]):
        dataframe = pd.read_csv('./country_data/{}.csv'.format(country))
        print(country,index,named_colorscales[index])
        # for field in fields:
        fig_confirmed.add_trace(go.Scatter(x=dataframe['date'], y=dataframe[field],
                                               mode='lines+markers',
                                               line_shape='spline',
                                               name='{}'.format(country),
                                               # line=dict(color='#921113', width=4),
                                               line=dict(color=named_colorscales[index], width=4),
                                               marker=dict(size=4, color=named_colorscales[index],
                                                           line=dict(width=1, color=named_colorscales[index])),
                                               text=dataframe['date'],
                                               hovertext=[
                                                   country + " {} cases <br>{:,f}%".format(field,i)
                                                   for i in dataframe[field]

                                               ],
                                               hovertemplate="<b>%{text}</b><br></br>"
                                                             + "%{hovertext}"
                                                             + "<extra></extra>"
                                               ))



    fig_confirmed.update_layout(
        #    title=dict(
        #    text="<b>Confirmed Cases Timeline<b>",
        #    y=0.96, x=0.5, xanchor='center', yanchor='top',
        #    font=dict(size=20, color="#292929", family="Playfair Display")
        #   ),
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=10,
            t=5,
            pad=0
        ),

        yaxis_title='{} cases'.format(field),
        yaxis=dict(
            showline=False, linecolor='#272e3e',
            zeroline=False,
            # showgrid=False,
            gridcolor='rgba(203, 210, 211,.3)',
            gridwidth=.1,
            # tickmode='array',
            # Set tick range based on the maximum number
            # tickvals=tickList,
            # Set tick label accordingly
            # ticktext=["{:.0f}k".format(i/1000) for i in tickList]
        ),
        #   yaxis_title="Total Confirmed Case Number",
        xaxis=dict(
            showline=False, linecolor='#272e3e',
            showgrid=False,
            gridcolor='rgba(203, 210, 211,.3)',
            gridwidth=.1,
            zeroline=False
        ),
        xaxis_tickformat='%b %d',
        hovermode='x',
        legend_orientation="h",
        legend=dict(x=.02, y=.95, bgcolor="rgba(0,0,0,0)",),
        plot_bgcolor='#f4f4f2',
        paper_bgcolor='#cbd2d3',
        font=dict(color='#292929', size=10)
    )
    # plt.show()

    # fig_confirmed.show()

    plotly.offline.plot(
    fig_confirmed,
    filename="Figures/{}_cases_TOP10".format(field),
    auto_open=False,
    )



df_frame = (
                    previous_worldometer_table.sort_values(by='Deaths', ascending=True)
                    .tail(num_of_elements)
                )
df_frame = df_frame[df_frame["Country/Region"]!= "World"]



max = 0

fig_rate = go.Figure()
for index, country in enumerate(df_frame["Country/Region"]):
    dataframe = pd.read_csv('./country_data/{}.csv'.format(country))

    if max < (dataframe["Deaths"] / dataframe["Confirmed"] * 100).max():
        max = (dataframe["Deaths"] / dataframe["Confirmed"] * 100).max()
        print(country)
        tickList = list(
            np.arange(
                0,
                (dataframe["Deaths"] / dataframe["Confirmed"] * 100).max()
                + 0.2,
                3,
            )
        )
        justone = False
    dataframe['date'] = pd.to_datetime(dataframe['date'], infer_datetime_format=True)
    fig_rate.add_trace(
        go.Scatter(
            x=dataframe["date"],
            y=dataframe["Deaths"] / dataframe["Confirmed"] * 100,
            mode="lines+markers",
            line_shape="spline",
            name=country,
            line=dict(color=named_colorscales[index], width=4),
            marker=dict(
                size=4, color="#f4f4f2", line=dict(width=1, color=named_colorscales[index])
            ),
            text=[
                datetime.strftime(d, "%b %d %Y AEDT")
                for d in dataframe["date"]
            ],
            hovertext=[
                country + " death rate (%) <br>{:.2f}%".format(i)
                for i in dataframe["Deaths"]
                / dataframe["Confirmed"]
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
    yaxis_title=" Top 10 countries confirmed cases -  death rate (%)",
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
    filename="Figures/death_rates_TOP10",
    auto_open=False,
)