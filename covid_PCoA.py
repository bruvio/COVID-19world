import pandas as pd
from pandas.plotting import register_matplotlib_converters
import subprocess

register_matplotlib_converters()
import logging
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

from numpy import zeros


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


# def cumulative_data_analysis():
today = date.today()
# dd/mm/YY
today = today.strftime("%d-%m-%Y")
# datatemplate = "time_series_19-covid-{}.csv"
datatemplate = "./csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{}_global.csv"
fields = ["confirmed", "deaths", "recovered"]
dataframe_all_countries = pre_process_database(datatemplate, fields)
countrylist_df = list(set(dataframe_all_countries["country"]))
# field = "Deaths"
# field = "Confirmed"
# field = "Recovered"
for field in fields:
    df = dataframe_all_countries[(dataframe_all_countries["quantity"] == field)]
    df = df.reset_index()

    df = df[["country", "date", "counts"]]

    df = df.pivot(index="date", columns="country", values="counts")
    df = df.reset_index()
    # df.to_csv("DailyData-{}.csv".format(field), index=True)
    # df.to_csv("DailyData.csv"), index=True)
    df.to_csv("DailyData-{}.csv".format(field), index=True)
    df.to_csv("DailyData-{}.csv".format(field), index=True)
    df.to_csv("DailyData-{}.csv".format(field), index=True)

    DailyData = pd.read_csv("./DailyData-{}.csv".format(field), index_col=0)
    DailyData.set_index("date", inplace=True)
    DailyData = DailyData.sort_index()

    # DailyData = DailyData.diff()
    # DailyData = DailyData.fillna(0)
    # Remove the latest day as it is not compeleted
    # DailyData = DailyData.drop(DailyData.iloc[-1].name)
    DailyData["China"]
    # Only include coutries have cases more than 50
    # DailyDataFifty = DailyData[[i for i in DailyData.columns if int(DailyData[i].sum()) > 890]]
    DailyDataFifty = DailyData[
        [i for i in DailyData.columns if int(DailyData[i].sum()) > 50]
    ]
    print(DailyDataFifty.shape)

    s = DailyDataFifty.sum()
    DailyDataFifty = DailyDataFifty[s.sort_values(ascending=False).index[:49]]
    print(DailyDataFifty.shape)

    # DailyDataFifty = DailyDataFifty.iloc[0:49]

    # cols = df[[i for i in df.columns if int(df[i].sum()) > 4]].stack().groupby(level=1).sum().head(2).index

    # Data transformation to reduce the effect of data scale on pattern identification
    # Square root transformation
    DailyDataFiftyTrans = DailyDataFifty ** 0.5
    # Normalisation column-wise
    from sklearn import preprocessing

    x = DailyDataFiftyTrans.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    DailyDataFiftyNorm = pd.DataFrame(x_scaled)
    DailyDataFiftyNorm.columns = DailyDataFifty.columns
    fig = plt.figure(figsize=(16, 12), dpi=200, constrained_layout=True)
    #
    axs = fig.subplots(nrows=7, ncols=7)
    #
    #
    #
    #
    #
    for i in range(len(DailyDataFiftyNorm.columns)):
        # print(i)
        axs.flat[i].plot(
            DailyDataFiftyNorm.index, DailyDataFiftyNorm.iloc[:, i], color="black"
        )
        axs.flat[i].get_xaxis().set_ticks([])
        axs.flat[i].get_yaxis().set_ticks([])
        axs.flat[i].annotate(
            DailyDataFiftyNorm.iloc[:, i].name,
            (0.05, 0.8),
            xycoords="axes fraction",
            va="center",
            ha="left",
        )

    fig.savefig("./Figures/" + "pattern_uncluster2-{}.png".format(today))
    #
    # plt.show(block=True)

    # plot similarities of countries/regions incremental patterns on a 2-D dimension and classify them into groups. In a nutshell, nearby circles (i.e countries/regions) in 2-D ordination should have similar growth patterns, circles which are far apart from each other have few patterns in common
    #
    #
    def bray_curtis_distance(table, sample1_id, sample2_id):
        """function to calculate bray-curtis distance"""
        numerator = 0
        denominator = 0
        sample1_counts = table[sample1_id]
        sample2_counts = table[sample2_id]
        for sample1_count, sample2_count in zip(sample1_counts, sample2_counts):
            numerator += abs(sample1_count - sample2_count)
            denominator += sample1_count + sample2_count
        return numerator / denominator

    from skbio.stats.distance import DistanceMatrix

    # from scipy.spatial import distance_matrix
    #
    def table_to_distances(table, pairwise_distance_fn):
        """pairwise distance as a table"""
        sample_ids = table.columns
        num_samples = len(sample_ids)
        data = zeros((num_samples, num_samples))
        for i, sample1_id in enumerate(sample_ids):
            for j, sample2_id in enumerate(sample_ids[:i]):
                data[i, j] = data[j, i] = pairwise_distance_fn(
                    table, sample1_id, sample2_id
                )
        # return data, sample_ids
        return DistanceMatrix(data, sample_ids)

    # data, sample_ids = table_to_distances(DailyDataFiftyNorm, bray_curtis_distance)

    # bc_dm = distance_matrix(data, sample_ids)
    #
    # # Produce a bray-curtis distance matrix
    bc_dm = table_to_distances(DailyDataFiftyNorm, bray_curtis_distance)
    print(bc_dm)
    #
    #
    # Using PCA on distance matrix and keep the first two components
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    projected = pca.fit_transform(bc_dm.data)

    fig0 = plt.figure(figsize=(12, 12), dpi=100)

    plt.scatter(projected[:, 0], projected[:, 1], s=100, alpha=0.5)
    plt.xlabel("component 1")
    plt.ylabel("component 2")

    for i, txt in enumerate(bc_dm.ids):
        plt.annotate(txt, (projected[:, 0][i], projected[:, 1][i] + 0.02))

    # plt.show()
    fig0.savefig("./Figures/" + "pattern_-{}.png".format(today))
    print(datetime)

    # Cluster countries using K-mean
    from sklearn.cluster import KMeans

    kmeans = KMeans(init="k-means++", n_clusters=5, random_state=0).fit(projected)
    kmeans.labels_

    # Variance explained by the first two components
    print(pca.explained_variance_)
    # Assign color to each cluster
    color = []
    for i in kmeans.labels_:
        if i == 3:
            # color.append('#a6d96a')
            color.append("blue")
        elif i == 0:
            # color.append('#d7191c')
            color.append("red")
        elif i == 1:
            # color.append('#2b83ba')
            color.append("yellow")
        elif i == 2:
            # color.append('#1a9641')
            color.append("green")
        else:
            color.append("cyan")
            # color.append('#fdae61')

    # Create a metadata table
    clusterData = pd.DataFrame(
        {
            "Cluster": kmeans.labels_,
            "Region": bc_dm.ids,
            "Color": color,
            "Cases": [DailyDataFifty[i].sum() for i in bc_dm.ids],
            "PC1": projected[:, 0],
            "PC2": projected[:, 1],
        },
    )
    # clusterData
    # Re-assign cluster id as desired order
    clusterData = clusterData.replace({"Cluster": {0: 9, 1: 7, 2: 5, 3: 6, 4: 8}})
    clusterData = clusterData.sort_values(by=["Cluster", "Cases"], ascending=False)
    clusterData = clusterData.reset_index(drop=True)
    # clusterData

    fig1 = plt.figure(figsize=(16, 10), dpi=200, constrained_layout=True)

    axs1 = fig1.subplots(nrows=7, ncols=7)

    for order, i in clusterData.iterrows():
        axs1.flat[order].plot(
            DailyDataFiftyNorm.index, DailyDataFiftyNorm[i["Region"]], color=i["Color"]
        )
        axs1.flat[order].get_xaxis().set_ticks([])
        axs1.flat[order].get_yaxis().set_ticks([])
        axs1.flat[order].annotate(
            i["Region"], (0.05, 0.8), xycoords="axes fraction", va="center", ha="left"
        )
    txt = "daily incremental case numbers against the date - clustered"
    fig1.text(0.5, 0.05, txt, ha="center")

    fig2 = plt.figure(figsize=(16, 10), dpi=200, constrained_layout=True)

    axs2 = fig2.subplots(nrows=7, ncols=7)

    for order, i in clusterData.iterrows():
        axs2.flat[order].plot(
            DailyDataFiftyNorm.index, DailyDataFiftyNorm[i["Region"]], color="black"
        )
        axs2.flat[order].get_xaxis().set_ticks([])
        axs2.flat[order].get_yaxis().set_ticks([])
        axs2.flat[order].annotate(
            i["Region"], (0.05, 0.8), xycoords="axes fraction", va="center", ha="left"
        )

    txt = "daily incremental case numbers against the date - unclustered"
    fig2.text(0.5, 0.05, txt, ha="center")
    # fig3 = plt.figure(figsize=(12,12), dpi=300)
    # ax1 = fig3.add_subplot()
    fig3, ax1 = plt.subplots()
    ax1.scatter(
        clusterData["PC1"],
        clusterData["PC2"],
        c=clusterData["Color"],
        s=clusterData["Cases"] / 500,
        alpha=0.7,
    )
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    txt = "growth pattern of different countries/regions"
    fig3.text(0.5, 0.05, txt, ha="center")
    ax1.set_xlabel("")
    # ax1.set_xlabel('PC1({:.2%} variance explaianed)'.format(pca.explained_variance_[0]))
    # ax1.set_ylabel('PC2({:.2%} variance explaianed)'.format(pca.explained_variance_[1]))

    for i, txt in enumerate(clusterData["Region"]):
        ax1.annotate(txt, (clusterData["PC1"][i], clusterData["PC2"][i] + 0.02))

    fig1.savefig("./Figures/" + "pattern_cluster-{}-{}.png".format(field, today))
    fig2.savefig("./Figures/" + "pattern_uncluster-{}-{}.png".format(field, today))
    fig3.savefig(
        "./Figures/" + "pattern_PCoA-{}-{}.png".format(field, today),
        bbox_inches="tight",
    )
    # plt.show()
    # print(clusterData)
    #
    #
    #
    for clusterNumber in [9, 7, 5, 6, 8]:

        # clusterNumber = 9
        # {0: 9, 1: 7, 2: 5, 3: 6, 4: 8}
        RegionList = clusterData[clusterData["Cluster"] == clusterNumber]["Region"]

        dfs_cumulative = DailyData[DailyData.columns.intersection(RegionList)]
        # dfs_cumulative = {sheet_name: pd.read_csv('../dash-2019-coronavirus/cumulative_data/{}.csv'.format(sheet_name))
        #           for sheet_name in RegionList}
        #
        # for region in RegionList:
        #     dfs_cumulative[region] = dfs_cumulative[region].sort_values(by='date')
        #
        fig4 = plt.figure(figsize=(12, 4), dpi=200, constrained_layout=True)
        ax4 = fig4.subplots()
        for region in RegionList:
            try:
                ax4.plot(
                    dfs_cumulative[region][dfs_cumulative[region] > 50].index,
                    dfs_cumulative[region][dfs_cumulative[region] > 50].values,
                )
                ax4.annotate(
                    region,
                    xy=(
                        list(dfs_cumulative[region][dfs_cumulative[region] > 50].index)[
                            0
                        ],
                        list(dfs_cumulative[region][dfs_cumulative[region] > 50])[0],
                    ),
                )
            except:
                pass

        ax4.set_xlabel("Days of the year")
        ax4.set_ylabel("Confirmed cases (log scale)")
        ax4.set_yscale("log")
        plt.xticks(rotation=70)

        fig4.savefig(
            "./Figures/"
            + "time_data_cluster-{}-{}-{}.png".format(field, clusterNumber, today)
        )


# DailyData20 = DailyDataFifty[s.sort_values(ascending=False).index[:20]]
# # RegionList = clusterData[clusterData["Cluster"] == 9]["Region"]
# RegionList = DailyData20.columns
# dfs_cumulative = DailyData[DailyData.columns.intersection(RegionList)]
# fig5 = plt.figure(figsize=(12, 4), dpi=200, constrained_layout=True)
# ax5 = fig5.subplots()
# for region in RegionList:
#     ax4.plot(
#         dfs_cumulative[region][dfs_cumulative[region] > 50].index,
#         dfs_cumulative[region][dfs_cumulative[region] > 50].values,
#     )
#     ax5.annotate(
#         region,
#         xy=(
#             list(dfs_cumulative[region][dfs_cumulative[region] > 50].index)[0],
#             list(dfs_cumulative[region][dfs_cumulative[region] > 50])[0],
#         ),
#     )
# ax5.set_xlabel('Days of the year')
# ax5.set_ylabel('Confirmed cases (log scale)')
# ax5.set_yscale("log")
# plt.xticks(rotation=70)
#
# fig5.savefig("time_top_20_regions.png")
