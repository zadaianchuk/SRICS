import matplotlib.pyplot as plt
import numpy as np



def get_gt_weights(env):
    dims = {"4-obj-complex": 5, "6-obj": 7, "4-obj": 5, "3-obj": 4}
    weights_gt_path = np.zeros((dims[env], dims[env]))
    if env == "4-obj-complex":
        weights_gt_path[2:, 0] = 1
        weights_gt_path[1, :] = 0
        weights_gt_path[2, 3] = 1
        weights_gt_path[3, 2] = 1
    elif env == "4-obj" or env == "3-obj" or  env == "6-obj" :
        weights_gt_path[1:, 0] = 1
    return weights_gt_path


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def plot_weights(weights, weights_gt, env, env_names, saturate=0.1):
    weights[weights>saturate] = saturate
    weights_gt = weights_gt.astype(np.float)
    if "4" in env: 
        names = ['Arm', '1', '2', '3', '4']
    elif "3" in env:
        names = ['Arm', '1', '2', '3']
    elif "6" in env:
        names = ['Arm', '1', '2', '3', '4', '5', '6']
    else:
        raise NotImplementedError
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    titles = ["Recurrent GNN", "GT"]
    for ax, title in zip(axs, titles):
        ax.set_xlabel(title, fontsize=18)

    _, _ = heatmap(weights, names, names, ax=axs[0])
    _, _ = heatmap(weights_gt, names, names, ax=axs[1])
    for title, ax  in zip(titles, axs):
        ax.set_xlabel(title, fontsize=18) 
        for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(18)   
    plt.suptitle(env_names[env], y=1.05)
    fig.tight_layout()