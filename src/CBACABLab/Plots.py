from itertools import cycle
import numpy as np
import re
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

CalculateDistance = lambda x, y: np.abs(x-y)
EpanechnikovWeight = lambda d: (3/4)*(1-np.power(d,2))
Silvermans = lambda x: (np.power((4*np.power(x.std(),5))/(3*x.count()),1/5))
CalculateWeight = lambda d: (70/81)*np.power((1-np.power(np.abs(d),3)),3)

def escape_dollar(label):
    return re.sub(r'\$', r'\\$', label)

def FindValue(x, df, xcolumn, ycolumn, bandwidth):
    bwcount = int(np.ceil(bandwidth*df[xcolumn].count()))
    df['distance'] = df[xcolumn].apply(func=CalculateDistance, args=(x,))
    subsample = df.sort_values(by=['distance', ])[:bwcount]
    maxsubvalue = subsample['distance'].max()
    subsample['wdistance'] = subsample['distance']/maxsubvalue
    weight = np.array(subsample['wdistance'].apply(func=CalculateWeight).tolist())
    y = subsample[ycolumn].values
    XM = np.column_stack([
        np.ones(len(subsample)),
        subsample[xcolumn].to_numpy(),
        np.power(subsample[xcolumn].to_numpy(), 2)
    ])
    r = sm.WLS(y, XM, weights=weight).fit().params
    xvals = np.array([1, x, np.power(x,2)]).T
    projection = np.dot(r,xvals)
    return projection

def RunLOESS(bandwidth, data, xcolumn, ycolumn):
    xinterval = np.linspace(data[xcolumn].min(), data[xcolumn].max(), 1000).tolist()
    yprojections = [FindValue(x, data, xcolumn, ycolumn, bandwidth) for x in xinterval]
    return (xinterval, yprojections)

def ManageLOESSProcess(bandwidth, data, xcolumn, ycolumn):
    xinterval, yprojections = RunLOESS(bandwidth, data, xcolumn, ycolumn)
    return xinterval, yprojections

def CalculateKDE(x, data, bandwidth):
    subsample = data.loc[((x-bandwidth) <= data) & (data <= (x+bandwidth))]
    n = data.count()
    if subsample.count()>0:
        normalizedsubsample = subsample.apply(func=CalculateDistance, args=(x,))/(bandwidth)
        density = (1/(n*bandwidth))*normalizedsubsample.apply(func=EpanechnikovWeight).sum()
        return density
    else:
        return 0

def RunKDE(bandwidth, data):
    mean = data.mean()
    minRange = mean - 6*data.std()
    maxRange = mean + 6*data.std()
    xinterval = np.linspace(minRange, maxRange, 1000).tolist()
    yprojections = [CalculateKDE(x, data, bandwidth) for x in xinterval]
    return (xinterval, yprojections)


def ManageKDEProcess(bandwidth, data):
    xinterval, yprojections = RunKDE(bandwidth, data)
    return xinterval, yprojections

def CreateLOESSPlots(data, xcolumn, ycolumn, bandwidths, linetypes=False, save=False, xlabel="X", ylabel="Y", title="LOESS", xStep=2, figsize=(9, 5)):
    defaultLinestyle = ['-', '--', ':', '-.']
    if linetypes is False:
        linetypes = []
        for i, item in enumerate(cycle(defaultLinestyle)):
            if i == len(data):
                break
            linetypes.append(item)
    r = [ ManageLOESSProcess(bandwidth, data, xcolumn, ycolumn) for bandwidth in bandwidths ]
    plt.figure(figsize=figsize)
    y_min, y_max = data[ycolumn].min(), data[ycolumn].max()
    x_min, x_max = data[xcolumn].min(), data[xcolumn].max()
    ax = plt.axes()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(x_min, x_max + 1, xStep))
    ax.set_title(title)
    ax.scatter(data[xcolumn].tolist(), data[ycolumn].tolist(), color='black')
    for index, result in enumerate(r):
        xinterval, yprojections = result
        ax.plot(xinterval, yprojections, label='Bandwidth = ' + str(round(bandwidths[index], 4)), color='black', linestyle=linetypes[index])
    ax.legend()
    if save:
        plt.savefig(save + '.pdf')
    plt.show()

def CreateMultipleLOESSPlots(data, xcolumn, ycolumns, bandwidths = False, linetypes=False, save=False, xlabel="X", ylabel="Y", title="Multiple LOESS Plots", xStep=2, figsize=(9, 5)):   
    if bandwidths is False:
        bandwidths = [0.3, 0.4, 0.5]
    if not isinstance(bandwidths, list):
        bandwidths = [bandwidths,]

    defaultColors = ['black']*4 + ['gray']*4
    colors = []
    for i, item in enumerate(cycle(defaultColors)):
        if i == len(ycolumns)*len(bandwidths):
            break
        colors.append(item)

    defaultLinestyle = ['-', '--', ':', '-.']
    if linetypes is False:
        linetypes = []
        for i, item in enumerate(cycle(defaultLinestyle)):
            if i == len(ycolumns)*len(bandwidths):
                break
            linetypes.append(item)
    r = [ ManageLOESSProcess(bandwidth, data, xcolumn, ycolumn) for bandwidth in bandwidths for ycolumn in ycolumns ]
    labels = [f"{ycolumn} (Bandwidth = {round(bandwidth, 4)})" for bandwidth in bandwidths for ycolumn in ycolumns]
    labels = [escape_dollar(label) for label in labels]
    plt.figure(figsize=figsize)
    y_min, y_max = data[ycolumns].min().min(), data[ycolumns].max().max()
    x_min, x_max = data[xcolumn].min(), data[xcolumn].max()
    ax = plt.axes()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(x_min, x_max + 1, xStep))
    ax.set_title(title)
    for index, result in enumerate(r):
        xinterval, yprojections = result
        ax.plot(xinterval, yprojections, label=labels[index], color=colors[index], linestyle=linetypes[index])
    ax.legend()
    if save:
        plt.savefig(save + '.pdf', bbox_inches='tight')
        plt.savefig(save + '.svg', bbox_inches='tight')
    plt.show()


def CreateMultipleLinePlots(data, xcolumn, ycolumns, linetypes=False, save=False, xlabel="X", ylabel="Y", title="Multiple Line Plots", xStep=2, figsize=(9, 6)):
    defaultColors = ['black']*4 + ['gray']*4
    colors = []
    for i, item in enumerate(cycle(defaultColors)):
        if i == len(ycolumns):
            break
        colors.append(item)
    defaultLinestyle = ['-', '--', ':', '-.']
    if linetypes is False:
        linetypes = []
        for i, item in enumerate(cycle(defaultLinestyle)):
            if i == len(ycolumns):
                break
            linetypes.append(item)
    plt.figure(figsize=figsize)
    y_min, y_max = data[ycolumns].min().min(), data[ycolumns].max().max()
    x_min, x_max = data[xcolumn].min(), data[xcolumn].max()
    ax = plt.axes()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(x_min, x_max + 1, xStep))
    ax.set_title(title)
    for index, ycolumn in enumerate(ycolumns):
        ax.plot(data[xcolumn], data[ycolumn], label=escape_dollar(ycolumn), color=colors[index], linestyle=linetypes[index])
    ax.legend()
    if save:
        plt.savefig(save + '.pdf', bbox_inches='tight')
        plt.savefig(save + '.svg', bbox_inches='tight')
    plt.show()

def ProjectKDEPlot(data, label='Score', labels=False, linetypes=False, save=False, lower=False, upper=False, bottom=0, top=False, mean=False, bwMultiplier = 1, sameBandwidth=False):
    showlegend = True
    if labels is False:
        labels = ['',]*len(data)
        showlegend = False
    defaultLinestyle = ['-', '--', ':', '-.']
    if linetypes is False:
        linetypes = []
        for i, item in enumerate(cycle(defaultLinestyle)):
            if i == len(data):
                break
            linetypes.append(item)
    dataset = []
    bandwidths = []
    for d in data:
        dataset.append(d)
        if sameBandwidth:
            bandwidths.append(Silvermans(data[0]) * bwMultiplier)
        else:
            bandwidths.append(Silvermans(d) * bwMultiplier)
    r = []
    for index, d in enumerate(dataset):
        r.append(ManageKDEProcess(bandwidths[index], d))
    plt.figure()
    ax = plt.axes()
    ax.set_ylabel('Density')
    ax.set_xlabel(label)
    xintervals = []
    yintervals = []
    for index, result in enumerate(r):
        xinterval, yprojections = result
        xintervals.append(xinterval)
        yintervals.append(yprojections)
    xintervalsNP = np.concatenate(xintervals)
    yintervalsNP = np.concatenate(yintervals)
    computedLower = np.percentile(xintervalsNP, 25)
    computedUpper = np.percentile(xintervalsNP, 75)
    computedTop = np.percentile(yintervalsNP, 100)*1.1
    if lower is False:
        lower = computedLower
    if upper is False:
        upper = computedUpper
    if top is False:
        top = computedTop
    for index, result in enumerate(r):
        xinterval, yprojections = result
        ax.plot(xinterval, yprojections, label=labels[index], color='black', linestyle=linetypes[index])
        if mean:
            ax.vlines(dataset[index].mean(), bottom, top, color='gray', linestyles=linetypes[index])
    ax.set_xlim(lower, upper)
    ax.set_ylim(bottom, top)
    #plt.vlines(subdata.mean(), 0, 1, color='black', linestyles='-')
    if showlegend:
        ax.legend()
    if save:
        plt.savefig(save + '.pdf', bbox_inches='tight')
        plt.savefig(save + '.svg', bbox_inches='tight')
    print('Silverman Bandwidths: ', bandwidths)
    plt.show()


def makeHeatMap(displayData, heatmapData, xlabel, ylabel, title, save=False, figsize=(11, 7), label=None, vmin=None, vmax=None, reversed=False, yticklabels=None):
    displayData.index = displayData.index.str.replace('_marginal', '', regex=False)
    common_index = displayData.index.intersection(heatmapData.index)
    displayData = displayData.loc[common_index, :]
    heatmapData = heatmapData.loc[common_index, :]
    heatmapData.reindex_like(displayData)
    heatmapData = heatmapData[[col for col in displayData.columns if col in heatmapData.columns]]
    cmap = 'Greys_r' if reversed else 'Greys'
    plt.figure(figsize=figsize)
    ax = sns.heatmap(heatmapData, annot=displayData, fmt='.4f', cmap=cmap, cbar_kws={'label': label}, vmin=vmin, vmax=vmax, yticklabels=yticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if save:
        plt.savefig(save + '.pdf', bbox_inches='tight')
        plt.savefig(save + '.svg', bbox_inches='tight')
    plt.show()