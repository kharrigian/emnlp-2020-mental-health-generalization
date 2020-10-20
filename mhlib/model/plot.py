
######################
### Imports
######################

## External Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

######################
### Functions
######################

def plot_model_coefficients(model,
                            top_n=30):
    """
    Plot feature relationships with the target outcome.

    Args:
        model (MentalHealthClassifier): Trained mental health classification model
    
    Returns:
        fig, ax (matplotlib figure objects)
    """
    ## Extract Import Feature Values
    is_coefficient = False
    try:
        coefs_ = getattr(model.model, "coef_")
        if coefs_.shape[0] > 1:
            return None, None
        coefs_ = coefs_[0]
        is_coefficient = True
    except:
        try:
            coefs_ = getattr(model.model, "feature_importances_")
        except:
            return None, None
    ## Extract Feature names and Sort
    ordered_feature_names = model.get_feature_names()
    coef_df = pd.Series(index = ordered_feature_names, data = coefs_)
    coef_df = coef_df.sort_values(ascending=False).reset_index()
    ## Focus on Top Weighted Features
    coefs_for_plot = coef_df.head(int(top_n / 2)).append(coef_df.tail(int(top_n / 2)))
    coefs_for_plot = coefs_for_plot.drop_duplicates("index")
    ## Generate Plot
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ind = list(range(len(coefs_for_plot)))
    ax.barh(ind,
            coefs_for_plot[0],
            color = "C0",
            alpha = .7)
    ax.set_yticks(ind)
    ax.set_yticklabels(coefs_for_plot["index"].tolist())
    ax.set_xlabel("Feature Coefficient" if is_coefficient else "Feature Importance")
    if ax.get_xlim()[0] < 0 and ax.get_xlim()[1] > 0:
        ax.axvline(0,
                   color = "black",
                   alpha = .3,
                   linestyle = ":")
    fig.tight_layout()
    return fig, ax

def plot_roc_auc(results_df,
                 groups=["train","dev"]):
    """
    Plot ROC curves with AUC scores for the training and 
    dev sets of cross-validation.

    Args:
        results_df (pandas DataFrame): DataFrame with 
                [y_true, y_pred, fold, group] columns.
        groups (list of str): Expected group variables. Should be of length 2. Default
                              is ["train","dev"]
    
    Returns:
        fig, ax (matplotlib figure objects): ROC/AUC plot
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.8), sharey=True, sharex=True)
    auc_scores = dict((g,[]) for g in groups)
    for g, group in enumerate(groups):
        avg_fpr, avg_tpr, avg_thresh = metrics.roc_curve(
                   results_df.loc[results_df["group"]==group]["y_true"].values,
                   results_df.loc[results_df["group"]==group]["y_pred"].values,
                   pos_label=1)
        for fold in sorted(results_df.fold.unique()):
            plot_data = results_df.loc[(results_df.group==group)&
                                       (results_df.fold==fold)]
            plot_fpr, plot_tpr, plot_thresh = metrics.roc_curve(
                plot_data["y_true"].values,
                plot_data["y_pred"].values,
                pos_label=1)
            plot_auc = metrics.auc(plot_fpr, plot_tpr)
            auc_scores[group].append(plot_auc)
            ax[g].plot(plot_fpr,
                       plot_tpr,
                       color = "C0",
                       alpha = .5)
        ax[g].plot([0, 1],
                   [0, 1],
                   color="red",
                   linestyle="--",
                   label="Random Chance")
        ax[g].plot(avg_fpr,
                   avg_tpr,
                   color = "C0",
                   alpha = 1,
                   linewidth = 2,
                   label = "Average AUC = {:.3f} ($\pm${:.3f})".format(
                            np.mean(auc_scores[group]),
                            np.std(auc_scores[group])
                   ))
        ax[g].set_xlim(0,1)
        ax[g].set_ylim(0,1)
        ax[g].set_xlabel("False Positive Rate", fontsize = 12)
        ax[g].set_title(f"{group.title()}", fontsize = 13)
        ax[g].legend(loc = "lower right", fontsize = 12)
    ax[0].set_ylabel("True Positive Rate", fontsize = 12)
    fig.tight_layout()
    return fig, ax


def _compute_precision_recall_f1(results_df,
                                 groups=["train","dev"]):
    """
    Compute precision, recall, and f1 score for the training and dev
    splits across each fold of the cross-validation procedure.
    
    Args:
        results_df (pandas DataFrame): DataFrame with [source, y_true, 
                        y_pred, fold, group] columns.
        groups (list of str): Expected group variables. Should be of length 2. Default
                              is ["train","dev"]

    Returns:
        scores (pandas DataFrame): Precision, Recall, F1 for each fold, prediction group
    """
    ## Initialize Scores Dictionary
    scores = dict((group, {'precision': [], 'recall': [], 'f1': []}) for group in groups)
    ## Compute Precision/Recall/F1 for each group and fold
    for group in groups:
        for fold in sorted(results_df.fold.unique()):
            fold_data = results_df.loc[(results_df.group==group)&
                                       (results_df.fold==fold)]
            prf_res = metrics.precision_recall_fscore_support(
                fold_data["y_true"],
                (fold_data["y_pred"]>=0.5).astype(int),
                pos_label=1,
                warn_for=set(),
                average="binary"
            )
            for m, met in enumerate(["precision","recall","f1"]):
                scores[group][met].append(prf_res[m])
    ## Format into DF
    train_scores = pd.DataFrame(scores[groups[0]])
    train_scores["group"] = groups[0]
    train_scores["fold"] = list(range(1, len(train_scores)+1))
    dev_scores = pd.DataFrame(scores[groups[1]])
    dev_scores["group"] = groups[1]
    dev_scores["fold"] = list(range(1, len(dev_scores)+1))
    scores = train_scores.append(dev_scores)
    return scores

def plot_precision_recall_f1(results_df,
                             groups=["train","dev"]):
    """
    Plot a bar chart of average precision, recall, and F1. Error
    bars represent standard error around the mean.

    Args:
        results_df (pandas DataFrame): DataFrame with [source, y_true, 
                        y_pred, fold, group] columns.
        groups (list of str): Expected group variables. Should be of length 2. Default
                              is ["train","dev"]

    Returns:
        fig, ax (matplotlib figure objects): Generated figure.
    """
    ## Compute Summary Stats
    prf_scores = _compute_precision_recall_f1(results_df, groups)
    ## Get Mean and Standard Error
    mu = prf_scores.groupby(["group"]).agg(np.mean)
    se = prf_scores.groupby(["group"]).agg(lambda vals: np.std(vals) / np.sqrt(len(vals)))
    ## Plot
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for g, group in enumerate(groups):
        g_mu = mu.loc[group, ["precision","recall","f1"]].values
        g_se = se.loc[group, ["precision","recall","f1"]].values
        g_ind = 0.025 * (1-g) + np.arange(3) + 0.5*g
        ax.bar(g_ind,
               g_mu,
               yerr = g_se,
               color = f"C{g}",
               alpha = .7,
               width = .475,
               label = group.title(),
               align = "edge")
        annot_str = ["{:.2f}\n($\pm${:.2f})".format(x, y) for x, y in zip(g_mu, g_se)]
        annot_pts = [[x, y+0.01] for x, y in zip(0.025 * (1-g) + np.arange(3) + 0.5*g + .475/2, g_mu+g_se)]
        for (x, y), s in zip(annot_pts, annot_str):
            ax.text(x, y, s, fontsize = 12, multialignment="center", ha = "center")
    ax.legend(loc = "lower right", fontsize = 12)
    ax.set_xticks(np.arange(3)+.5)
    ax.set_xticklabels(["Precision","Recall","F1"], fontsize = 12)
    ax.set_ylabel("Score", fontsize = 12)
    ax.set_ylim(0,1.1)
    fig.tight_layout()
    return fig, ax

def plot_predicted_probability_distribution(results_df,
                                            groups=["train","dev"]):
    """
    Plot the distribution of probabilities predicted by the model, compared
    against the true class label.

    Args:
        results_df (pandas DataFrame): DataFrame with [source, y_true, 
                        y_pred, fold, group] columns.
        groups (list of str): Expected group variables. Should be of length 2. Default
                              is ["train","dev"]

    Returns:
        fig, ax (matplotlib figure objects): Generated figure.
    """
    folds = sorted(results_df["fold"].unique())
    fig, ax = plt.subplots(2, len(folds), figsize=(10,5.8), sharey = True, sharex=True)
    for g, group in enumerate(groups):
        for i, f in enumerate(folds):
            if len(folds) > 1:
                plot_ax = ax[g][i]
            else:
                plot_ax = ax[g]
            for c, condition in enumerate(["Control","Disorder"]):
                plot_data = results_df.loc[(results_df["fold"]==f)&
                                           (results_df["group"]==group)&
                                           (results_df["y_true"]==c)]
                plot_ax.hist(plot_data["y_pred"].values,
                              color = f"C{c}",
                              alpha = .3,
                              bins = np.linspace(0, 1, 21),
                              density = True,
                              label = str(condition).title() + " (True)")
            plot_ax.set_xlim(0, 1)
            plot_ax.axvline(0.5,
                             color = "black",
                             alpha = .5,
                             linestyle = ":",
                             label = "Decision Boundary")
            if g == 1:
                plot_ax.set_xlabel("Pr(Disorder)")
            else:
                plot_ax.set_title(f"Fold {f}")
            if i == 0:
                plot_ax.set_ylabel(f"Density\n({group.title()})")
    handles, labels = plot_ax.get_legend_handles_labels()
    midpoint = int(np.ceil(len(folds) / 2)) - 1
    if len(folds) > 1:
        leg_ax = ax[0][i]
    else:
        leg_ax = ax[-1]
    leg = plot_ax.legend(handles,
                         labels,
                         loc = (1.05, .725),
                         ncol = 1,
                         frameon=False,
                         columnspacing=1)
    fig.tight_layout()
    return fig, ax
