
## Prefix (Unique Identifier for Experiments)
PREFIX = "TemporalEffectsMisalignment"

## Output
PLOT_DIR = "./plots/temporal_effects/misalignment/"
DATA_DIR = "./data/results/temporal_effects/"

## Metrics
METRICS = ["f1","precision","recall","accuracy","auc"]

## Flags
PLOT_MISALIGNMENT = True
PLOT_TRANSFER_MATRIX = True
PLOT_TRANSFER_LINE = False
PLOT_DATA_SET_TRANSFER = False

## Support Based Filter (Most Relevant for Small datasets)
FILTER_SUPPORT = False

#######################
### Imports
#######################

## Standard Library
import os
import json
from glob import glob

## External Libaries
import pandas as pd
import numpy as np 
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from pandas.plotting import register_matplotlib_converters
_ = register_matplotlib_converters()

#######################
### Helpers
#######################

## Default Data Set Order
DS_ORDER = ["clpsych_deduped","multitask","rsdd","smhd","wolohan"]

## Data Set Names
DS_NAMES = {
    "clpsych_deduped":"CLPsych",
    "clpsych":"ClPysch",
    "multitask":"Multi-Task",
    "rsdd":"RSDD",
    "smhd":"SMHD",
    "wolohan":"Topic-Restricted"
}

## Colors
COLORS = {
    "clpsych_deduped":{
        "cmap":plt.cm.Blues,
        "c":"navy",
        "marker":"o"
    },
    "clpsych":{
        "cmap":plt.cm.Blues,
        "c":"navy",
        "marker":"o"
    },
    "multitask":{
        "cmap":plt.cm.Oranges,
        "c":"darkorange",
        "marker":"^"
    },
    "rsdd":{
        "cmap":plt.cm.Greens,
        "c":"darkgreen",
        "marker":"s"
    },
    "smhd":{
        "cmap":plt.cm.Reds,
        "c":"darkred",
        "marker":"D"
    },
    "wolohan":{
        "cmap":plt.cm.Purples,
        "c":"darkviolet",
        "marker":"v"
    }
}

def load_output_data(directory,
                     load_predictions=False,
                     load_scores=True):
    """

    """
    directory = directory.rstrip("/")
    ## Load Configuration
    config_file = f"{directory}/config.json"
    with open(config_file, "r") as the_file:
        config = json.load(the_file)
    ## Load Predictions
    predictions_file = f"{directory}/cross_validation/predictions.csv"
    if load_predictions and os.path.exists(predictions_file):
        predictions = pd.read_csv(predictions_file)
    else:
        predictions = None
    ## Load Scores
    scores_file = f"{directory}/cross_validation/scores.csv"
    if load_scores and os.path.exists(scores_file):
       scores = pd.read_csv(scores_file)
    else:
        scores = None
    ## Source/Target Domains
    source_domain = config["train_data"]
    target_domain = config["test_data"]
    return source_domain, target_domain, predictions, scores

def plot_misalignment(scores,
                      group="dev",
                      metric="f1",
                      seen_subset="unseen",
                      plot_type = "bar"):
    """
    H1: Classification is easier when there is a temporal mismatch 
    between classes (e.g. depression, control group)
        - F(x, x, x’, x’) < F(x, y, x’, y’),
        - F(y, y, y’, y’) < F(x, y, x’, y’)
        - F(x, x, x’, x’) <  F(y, x, y’, x’)
        - F(y, y, y’, y’) < F(y, x, y’, x’)
            - Why? Because when there is an additional temporal element that separates 
            the control and target groups that should in theory make classification easier
    
    Args:
        scores (pandas DataFrame)
        plot_type (str): "hist", "bar", or "box"
    
    Returns:
        fig, ax (matplotlib objects)
    """
    ## Get Subset
    score_subset = scores.loc[(scores["source"]==scores["target"])&
                              (scores["group"]==group)&
                              (scores["seen_subset"]==seen_subset)].copy()
    ## Support Filter
    if FILTER_SUPPORT:
        median_support = score_subset.groupby(["target"])["support"].median().to_dict()
        score_subset = score_subset.loc[score_subset.apply(lambda row: row["support"] >= median_support[row["target"]], axis=1)]
        ## Get Hypothesis Subsets
    scores_same = score_subset.loc[(score_subset["target_train"]==score_subset["control_train"])&
                                   (score_subset["target_train"]==score_subset["target_test"])&
                                   (score_subset["target_test"]==score_subset["control_test"])]
    scores_mismatch = score_subset.loc[(score_subset["target_train"]!=score_subset["control_train"])&
                                       (score_subset["target_test"]!=score_subset["control_test"])&
                                       (score_subset[["target_train","control_train"]].apply(tuple,axis=1)==\
                                        score_subset[["target_test","control_test"]].apply(tuple,axis=1))]
    ## Temporal Groups
    datasets = score_subset["source"].unique()
    ## Histogram
    if plot_type == "hist":
        fig, ax = plt.subplots(1, len(datasets), figsize=(10,5.8))
        for d, ds in enumerate(datasets):
            for s, scoreset in enumerate([scores_same, scores_mismatch]):
                plot_data = scoreset.loc[scoreset["source"]==ds]
                ax[d].hist(plot_data[metric],
                           bins=10,
                           alpha=0.5,
                           color=f"C{s}",
                           label="Aligned" if s == 0 else "Mismatch",
                           density=True)
                ax[d].axvline(plot_data[metric].mean(),
                              alpha = .8,
                              color=f"C{s}",
                              linewidth=3,
                              linestyle="--")
            ax[d].set_xlabel(metric.title(), fontsize=10, fontweight="bold")
            ax[d].set_ylabel("Density", fontsize=10, fontweight="bold")
            ax[d].set_title(DS_NAMES[ds], loc="left", fontsize=10, fontweight="bold")
            l = ax[d].legend(loc="upper right", frameon=True, fontsize=8, title="Class Temporal Periods")
            plt.setp(l.get_title(),fontsize=8)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(10,5.8))
        for d, ds in enumerate(datasets):
            for s, scoreset in enumerate([scores_same, scores_mismatch]):
                plot_data = scoreset.loc[scoreset["source"]==ds]
                if plot_type == "bar":
                    mean, std = plot_data[metric].mean(), plot_data[metric].std()
                    std_error = std / np.sqrt(len(plot_data)-1)
                    ax.bar(d + 0.05 + s * .45,
                           mean,
                           yerr = std_error,
                           color=f"C{s}",
                           alpha=.5,
                           label="Aligned" if s == 0 and d == 0 else "Mismatch" if s == 1 and d ==0 else "",
                           align="edge",
                           width=.45)
                elif plot_type == "box":
                    b = ax.boxplot(plot_data[metric],
                               positions=[d + 0.05 + s * .45 + .45/2],
                               widths=0.45,
                               boxprops={"color":f"C{s}","linewidth":2},
                               meanprops={"color":"black"},
                               medianprops={"color":f"C{s}","linewidth":2},
                               capprops={"linewidth":2},
                               whiskerprops={"linewidth":2}
                               )
                    k = b["boxes"][0]
                    k.set_label("Aligned" if s == 0 and d == 0 else "Mismatch" if s == 1 and d ==0 else "")
        ax.set_xticks(np.arange(d+1)+.5)
        ax.set_xticklabels([ds.replace("_"," ").upper() for ds in datasets])
        ax.set_xlabel("Data Set", fontsize=10, fontweight="bold")
        ax.set_ylabel(metric.title(), fontweight="bold")
        l = ax.legend(loc="upper left", frameon=True, framealpha=1, title="Class Temporal Periods", fontsize=8)
        plt.setp(l.get_title(), fontsize=8)
        fig.tight_layout()
    return fig, ax

def plot_within_domain_matrix(scores,
                              metric="f1",
                              group="dev",
                              seen_subset="unseen"):
    """

    """
    ## Isolate Subset
    score_subset = scores.loc[(scores["group"]==group)&
                              (scores["seen_subset"]==seen_subset)&
                              (scores["source"]==scores["target"])&
                              (scores["target_train"]==scores["control_train"])&
                              (scores["target_test"]==scores["control_test"])].reset_index(drop=True).copy()
    ## Add Temporal Difference
    score_subset["date_train"] = pd.to_datetime(score_subset["target_train"])
    score_subset["date_test"] = pd.to_datetime(score_subset["target_test"])
    score_subset["date_delta"] = score_subset["date_train"].map(lambda i: i.year) - score_subset["date_test"].map(lambda i: i.year)
    ## Support Filter
    if FILTER_SUPPORT:
        median_support = score_subset.groupby(["target"])["support"].median().to_dict()
        score_subset = score_subset.loc[score_subset.apply(lambda row: row["support"] >= median_support[row["target"]], axis=1)]
    ## Add Data Set
    score_subset["dataset"] = score_subset["source"]
    ## Data Sets
    datasets = set(score_subset["dataset"].values)
    datasets = [i for i in DS_ORDER if i in datasets]
    ## Plot Transfer Metrics
    fig, ax = plt.subplots(2, 3, figsize=(8,6))
    ax = ax.ravel()
    for d, ds in enumerate(datasets):
        ## Get Data Set Subset
        d_data = score_subset.loc[score_subset["dataset"]==ds]
        ## Pivot Table Plot
        d_pivot = pd.pivot_table(d_data,
                                 index="date_train",
                                 columns="date_test",
                                 values=metric,
                                 aggfunc=np.mean)
        ax[d].imshow(d_pivot.values,
                     aspect="auto",
                     cmap=COLORS[ds]["cmap"],
                     interpolation="nearest",
                     alpha=.75)
        for i, col in enumerate(d_pivot.values):
            for j, val in enumerate(col):
                ax[d].text(j, i, "{:.2f}".format(val).lstrip("0"), ha="center", va="center", fontsize=12)
        ## Average Test Performance Plot (Within-Domain)
        avg_test_perf = d_data.groupby(["date_test"]).agg({metric:[np.mean,np.std,len]})[metric]
        avg_test_perf["standard_error"] = avg_test_perf["std"] / np.sqrt(avg_test_perf["len"]-1)
        ax[-1].errorbar(avg_test_perf.index,
                        avg_test_perf["mean"],
                        yerr=avg_test_perf["standard_error"],
                        color=COLORS[ds]["c"],
                        linewidth=2)
        ax[-1].scatter(avg_test_perf.index,
                       avg_test_perf["mean"],
                       color=COLORS[ds]["c"],
                       marker=COLORS[ds]["marker"],
                       label=DS_NAMES[ds])
        ## formatting
        ax[d].set_xticks(list(range(len(d_pivot.columns))))
        ax[d].set_xticklabels([i.year for i in d_pivot.columns],fontsize=14,rotation=45,ha="right")
        ax[d].set_yticks(list(range(len(d_pivot.index))))
        ax[d].set_yticklabels([i.year for i in d_pivot.index],fontsize=14)
        if d > 2:
            ax[d].set_xlabel("Test Year", fontsize=16, fontweight="bold")
        if d in [0, 3]:
            ax[d].set_ylabel("Train Year", fontsize=16, fontweight="bold")
        ax[d].set_ylim(len(d_pivot.index)-.5, -.5)
        ax[d].set_title(DS_NAMES[ds],loc="center",fontsize=16,fontweight="bold", fontstyle="italic")
    ## Format Axes
    for i in range(d+1,5):
        ax[i].axis("off")
    ## Format Timeseries
    xticks = sorted(score_subset["date_test"].unique())
    ax[-1].set_xticks(xticks)
    xticklabels = [pd.to_datetime(i).year for i in xticks]
    ax[-1].set_xticklabels(list(map(lambda x: x[1] if x[0] % 2 ==0 else "", list(enumerate(xticklabels)))), rotation=45, ha="right", fontsize=8)
    ax[-1].set_xlabel("Test Year", fontsize=16, fontweight="bold")
    ax[-1].set_ylabel(f"Average {metric.title()}", fontsize=16, fontweight="bold")
    ax[-1].tick_params(labelsize=14)
    ax[-1].spines["top"].set_visible(False)
    ax[-1].spines["right"].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    return fig, ax


def create_metric_comparisons(scores_subset,
                              metric="f1",
                              filter_within_domain_zero=False):
    """

    """
    ## Subset of Columns
    col_subset = ["source","target","date_train","date_test","date_delta","fold", metric]
    metric_subset = scores_subset[col_subset]
    ## Baselines (Within-Domain, Within Time Window)
    baselines = metric_subset.loc[(metric_subset["date_train"]==metric_subset["date_test"])&
                                  (metric_subset["source"]==metric_subset["target"])].reset_index(drop=True).copy()
    baselines = baselines.set_index(["target","date_test","fold"])[metric].to_dict()
    ## Out-of-Window
    comparisons = metric_subset.copy()
    comparisons[f"{metric}_baseline"] = comparisons.apply(lambda row: baselines[(row["target"],row["date_test"],row["fold"])], axis=1)
    comparisons[f"{metric}_difference"] = comparisons[metric] - comparisons[f"{metric}_baseline"]
    comparisons[f"{metric}_percent_difference"] = (comparisons[metric] - comparisons[f"{metric}_baseline"]) / comparisons[f"{metric}_baseline"] * 100
    if filter_within_domain_zero:
        for col in [f"{metric}_difference",f"{metric}_percent_difference"]:
            comparisons.loc[(comparisons["source"]==comparisons["target"])&(comparisons["date_delta"]==0), col] = np.nan
    return comparisons

def plot_transfer(scores,
                  metric="f1",
                  group="dev",
                  seen_subset="unseen",
                  percent=False):
    """

    """
    ## Get Subset of Data (Group/Seen Subset/Date Splits)
    scores_subset = scores.loc[(scores["seen_subset"]==seen_subset)&
                               (scores["group"]==group)&
                               (scores["target_test"]==scores_df["control_test"])&
                               (scores["target_train"]==scores["control_train"])].copy()
    scores_subset["date_train"] = pd.to_datetime(scores_subset["target_train"])
    scores_subset["date_test"] = pd.to_datetime(scores_subset["target_test"])
    scores_subset["date_delta"] = scores_subset["date_train"].map(lambda i: i.year) - scores_subset["date_test"].map(lambda i: i.year)
    ## Support Filter
    if FILTER_SUPPORT:
        median_support = scores_subset.groupby(["target"])["support"].median().to_dict()
        scores_subset = scores_subset.loc[scores_subset.apply(lambda row: row["support"] >= median_support[row["target"]], axis=1)]
    ## Target Data Sets
    target_datasets = set(scores_subset["target"])
    target_datasets = [d for d in DS_ORDER if d in target_datasets]
    ## Source Data Sets
    source_datasets = {}
    for t in target_datasets:
        t_source = scores_subset.loc[scores_subset["target"]==t]["source"].unique()
        source_datasets[t] = [d for d in DS_ORDER if d in t_source]
    ## Represent Performance ~ Within Domain/Within Time Baselines
    score_comparisons = create_metric_comparisons(scores_subset, metric)
    ## Plot Value
    if percent:
        plot_col = f"{metric}_percent_difference"
    else:
        plot_col = f"{metric}_difference"
    ## Create Plot
    f = plt.figure(constrained_layout=False,figsize=(8,6))
    gs = gridspec.GridSpec(2, 3, figure=f)
    row = 0
    col = 0
    for d_test, ds_test in enumerate(target_datasets):
        test_gs = gridspec.GridSpecFromSubplotSpec(len(source_datasets[ds_test]),
                                                   1,
                                                   subplot_spec=gs[row,col])
        if col == 2:
            row += 1
            col = 0
        else:
            col += 1
        ax_added = 0
        xmax = -1; xmin = np.inf
        ymax = -np.inf; ymin = np.inf
        axes = []
        for d_train, ds_train in enumerate(target_datasets):
            ## Check Existence
            if ds_train not in source_datasets[ds_test]:
                continue
            ## Get Axis
            d_ax = f.add_subplot(test_gs[ax_added,:],
                                 label=(d_test,d_train))
            axes.append(d_ax)
            ## Get Data
            ds_data = score_comparisons.loc[(score_comparisons["source"]==ds_train)&
                                            (score_comparisons["target"]==ds_test)]
            ## Standard Delta Averages
            col_agg_standard = ds_data.groupby(["date_delta"]).agg({plot_col:[np.mean, np.std, len]})[plot_col]
            col_agg_standard["standard_error"] = col_agg_standard["std"] / np.sqrt(col_agg_standard["len"]-1)
            col_agg_standard.sort_index(inplace=True)
            ## Plot Error Bar
            eb = d_ax.errorbar(col_agg_standard.index.values,
                               col_agg_standard["mean"].values,
                               yerr=col_agg_standard["standard_error"].values,
                               color=COLORS[ds_train]["c"],
                               label=ds_train,
                               alpha=.9,
                               linewidth=2,
                               zorder=10,
                               marker=COLORS[ds_train]["marker"],
                               ms=4)
            if d_ax.get_ylim()[1] > 0 and d_ax.get_ylim()[0] < 0:
                d_ax.axhline(0,
                            color="black",
                            linestyle=":",
                            alpha=0.5,
                            linewidth=1)
            d_ax.spines["top"].set_visible(False)
            d_ax.spines["right"].set_visible(False)
            ## Update Limits
            if col_agg_standard.index.min() < xmin:
                xmin = col_agg_standard.index.min()
            if col_agg_standard.index.max() > xmax:
                xmax = col_agg_standard.index.max()
            axmin = (col_agg_standard["mean"] - col_agg_standard["standard_error"]).min()
            axmax = (col_agg_standard["mean"] + col_agg_standard["standard_error"]).max()
            if axmin < ymin:
                ymin = axmin
            if axmax > ymax:
                ymax = axmax
            ## Format Axis
            ax_added += 1
            if ax_added != len(source_datasets[ds_test]):
                d_ax.tick_params(labelbottom=False)
            else:
                d_ax.set_xlabel("Latency (yrs.)", fontsize=12, fontweight="bold")
                d_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                xticks_filt = [x for x in d_ax.get_xticks() if x % 2 == 0]
                if xmin < xticks_filt[0]:
                    xticks_filt = [xticks_filt[0]-2] + xticks_filt
                if xmax > xticks_filt[-1]:
                    xticks_filt = xticks_filt + [xticks_filt[-1]+2]
                xmax = xticks_filt[-1]+0.5
                xmin = xticks_filt[0]-0.5
                d_ax.set_xticks(xticks_filt)
                d_ax.set_xticklabels([int(j) for j in xticks_filt],fontsize=8)
            if ax_added == 1:
                d_ax.set_title("{}".format(DS_NAMES[ds_test]), fontsize=16, fontweight="bold", loc="center", fontstyle="italic")
            d_ax.yaxis.set_major_locator(MaxNLocator(nbins=1))
            d_ax.tick_params(labelsize=8)
            d_ax.spines["left"].set_visible(False)
            if percent:
                d_ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
        ## Format Group of Axes
        for a in axes:
            a.set_xlim(xmin-1, xmax+1)
            # a.set_ylim(ymin, ymax)
            a.axvline(0, color="black", linestyle=":", alpha=.5, zorder=-1, linewidth=1)
    ## Legend
    legend_gs = gridspec.GridSpecFromSubplotSpec(1,
                                                 1,
                                                 subplot_spec=gs[1,2])
    legend_ax = f.add_subplot(legend_gs[:],
                              label="legend")
    for i, t_ds in enumerate(target_datasets):
        legend_ax.plot([],
                       [],
                       color=COLORS[t_ds]["c"],
                       linewidth=2,
                       marker=COLORS[t_ds]["marker"],
                       label=DS_NAMES[t_ds])
    legend = legend_ax.legend(title="Source Data", loc="center", frameon=False, fontsize=14)
    legend_ax.axis("off")
    plt.setp(legend.get_title(),fontsize=12,fontweight="bold")
    ylabel = f" {metric.title()} Percent Difference From\nWithin Domain/Within Time" if percent else \
             f" {metric.title()} Difference From Within\nDomain/Within Time"
    f.text(0.05, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=16, fontweight="bold")
    f.tight_layout()
    f.subplots_adjust(left=0.18, wspace=0.4)
    return f, None

def plot_dataset_transfer(scores,
                          dataset,
                          metric="f1",
                          group="dev",
                          seen_subset="unseen",
                          percent=False):
    """
    Plot transfer *to* a specific target data set
    """
    ## Get Subset of Data (Group/Seen Subset/Date Splits)
    scores_subset = scores.loc[(scores["seen_subset"]==seen_subset)&
                               (scores["group"]==group)&
                               (scores["target_test"]==scores_df["control_test"])&
                               (scores["target_train"]==scores["control_train"])&
                               (scores["target"]==dataset)].copy()
    scores_subset["date_train"] = pd.to_datetime(scores_subset["target_train"])
    scores_subset["date_test"] = pd.to_datetime(scores_subset["target_test"])
    scores_subset["date_delta"] = scores_subset["date_train"].map(lambda i: i.year) - scores_subset["date_test"].map(lambda i: i.year)
    ## Support Filter
    if FILTER_SUPPORT:
        median_support = scores_subset.groupby(["target"])["support"].median().to_dict()
        scores_subset = scores_subset.loc[scores_subset.apply(lambda row: row["support"] >= median_support[row["target"]], axis=1)]
    ## Source Data Sets
    source_datasets = scores_subset["source"].unique()
    source_datasets = [d for d in DS_ORDER if d in source_datasets]
    ## Represent Performance ~ Within Domain/Within Time Baselines
    score_comparisons = create_metric_comparisons(scores_subset, metric)
    ## Plot Value
    if percent:
        plot_col = f"{metric}_percent_difference"
    else:
        plot_col = f"{metric}_difference"
    ## Create Plot
    fig, ax = plt.subplots(len(source_datasets),1,figsize=(8,6))
    ax = ax.ravel()
    xmax = -1; xmin = np.inf
    for d_train, ds_train in enumerate(source_datasets):
        ## Get Axis
        d_ax = ax[d_train]
        ## Get Data
        ds_data = score_comparisons.loc[(score_comparisons["source"]==ds_train)]
        ## Standard Delta Averages
        col_agg_standard = ds_data.groupby(["date_delta"]).agg({plot_col:[np.mean, np.std, len]})[plot_col]
        col_agg_standard["standard_error"] = col_agg_standard["std"] / np.sqrt(col_agg_standard["len"]-1)
        col_agg_standard.sort_index(inplace=True)
        ## Plot Error Bar
        eb = d_ax.fill_between(col_agg_standard.index.values,
                               col_agg_standard["mean"].values - col_agg_standard["standard_error"].values,
                               col_agg_standard["mean"].values + col_agg_standard["standard_error"].values,
                               color="navy",
                               label=ds_train,
                               alpha=.5,
                               zorder=10
                               )
        d_ax.plot(col_agg_standard.index.values,
                  col_agg_standard["mean"].values,
                  color="navy",
                  alpha=1,
                  linewidth=2,
                  zorder=10)
        if d_ax.get_ylim()[1] > 0 and d_ax.get_ylim()[0] < 0:
            d_ax.axhline(0,
                        color="black",
                        linestyle="--",
                        alpha=0.75,
                        linewidth=1)
        ## Update Limits
        if col_agg_standard.index.min() < xmin:
            xmin = col_agg_standard.index.min()
        if col_agg_standard.index.max() > xmax:
            xmax = col_agg_standard.index.max()
        if d_train != len(source_datasets)-1:
            d_ax.tick_params(labelbottom=False)
        else:
            d_ax.set_xlabel("Train/Test Difference (yrs.)", fontsize=14, fontweight="bold")
            d_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            d_ax.yaxis.set_major_locator(MaxNLocator(nbins=1))
            xticks_filt = [x for x in d_ax.get_xticks() if x % 2 == 0]
            if xmin < xticks_filt[0]:
                xticks_filt = [xticks_filt[0]-2] + xticks_filt
            if xmax > xticks_filt[-1]:
                xticks_filt = xticks_filt + [xticks_filt[-1]+2]
            xmax = xticks_filt[-1]+0.5
            xmin = xticks_filt[0]-0.5
            d_ax.set_xticks(xticks_filt)
        ## Format Group of Axes
        d_ax.tick_params(labelsize=12)
        d_ax.set_title("Source: {}".format(DS_NAMES[ds_train]),
                       loc="left",
                       fontsize=12,
                       fontweight="bold")
    for a in ax:
        a.set_xlim(xmin-1, xmax+1)
        a.axvline(0, color="black", linestyle="--", alpha=.5, zorder=-1, linewidth=1)
    ylabel = f" {metric.title()} Percent Difference From\nWithin Domain/Within Time Baseline" if percent else \
             f" {metric.title()} Difference From Within\nDomain/Within Time Baseline"
    fig.text(0.05, 0.5, ylabel, ha="center", va="center", rotation=90, fontsize=14, fontweight="bold")
    fig.suptitle("Target Dataset: {}".format(DS_NAMES[dataset]),
                 va="center",
                 ha="center",
                 fontweight="bold",
                 fontsize=16,
                 y=.96)
    fig.tight_layout()
    fig.subplots_adjust(left=0.175, top=.89)
    return fig, ax

#######################
### Load Results
#######################

## Create Plot Directory
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

## Identify Result Directories based on Prefix
result_dirs = glob(f"{DATA_DIR}*{PREFIX}*/")

## Load Predictions and Scores
predictions_df = []
scores_df = []
for rd in result_dirs:
    source, target, predictions, scores = load_output_data(rd, True, True)
    for df in [predictions, scores]:
        df["source"] = source
        df["target"] = target
    predictions_df.append(predictions)
    scores_df.append(scores)
predictions_df = pd.concat(predictions_df).reset_index(drop=True)
scores_df = pd.concat(scores_df).reset_index(drop=True)

#######################
### H1: Misalignment
#######################

if PLOT_MISALIGNMENT:
    for metric in METRICS:
        fig, ax = plot_misalignment(scores_df,
                                    group="dev",
                                    seen_subset="unseen",
                                    metric=metric,
                                    plot_type="bar")
        fig.savefig(f"{PLOT_DIR}{metric}_temporal_misalignment.png", dpi=300)
        plt.close(fig)

#######################
### H2: Within-Domain Transfer
#######################

if PLOT_TRANSFER_MATRIX:
    for metric in METRICS:
        ## Plot Matrix Figure
        fig, ax = plot_within_domain_matrix(scores_df,
                                            metric=metric,
                                            group="dev",
                                            seen_subset="unseen")
        fig.savefig(f"{PLOT_DIR}{metric}_within_domain_matrix.png", dpi=300)
        plt.close(fig)

#######################
### H3: Cross-domain Transfer
#######################

if PLOT_TRANSFER_LINE:
    for metric in METRICS:
        for percent in [False, True]:
            ## Plot Relative Performance
            fig, ax = plot_transfer(scores_df,
                                    metric=metric,
                                    group="dev",
                                    seen_subset="unseen",
                                    percent=percent)
            if not percent:
                fig.savefig(f"{PLOT_DIR}{metric}_domain_transfer_line.png", dpi=300)
            else:
                fig.savefig(f"{PLOT_DIR}{metric}_percent_change_domain_transfer_line.png", dpi=300)
            plt.close(fig)

if PLOT_DATA_SET_TRANSFER:
    for metric in METRICS:
        for percent in [False, True]:
            for target in scores_df["target"].unique():
                ## Plot Performance
                fig, ax = plot_dataset_transfer(scores_df,
                                                target,
                                                metric=metric,
                                                group="dev",
                                                seen_subset="unseen",
                                                percent=percent)
                if not percent:
                    fig.savefig(f"{PLOT_DIR}{metric}_{target}_domain_transfer_line.png", dpi=300)
                else:
                    fig.savefig(f"{PLOT_DIR}{metric}_{target}_percent_change_domain_transfer_line.png", dpi=300)
                plt.close(fig)