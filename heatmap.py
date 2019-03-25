import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
zhfont1 = matplotlib.font_manager.FontProperties(fname='/home/hypo/.local/share/fonts/simsun.ttc')
zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/times.ttf')
FontSize=15
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs,origin='lower')

    # Create colorbar
    #cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontproperties=zhfont,fontsize=15)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels,fontsize=FontSize)
    ax.set_yticklabels(row_labels,fontsize=FontSize)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center",fontsize=FontSize)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# harvest = np.array(

# [[0.7577741407528642, 0.05994694960212202, 0.0, 0.0, 0.0012135922330097086], [0.23895253682487724, 0.8779840848806366, 0.24684684684684685, 0.16026711185308848, 0.04004854368932039], [0.0016366612111292963, 0.03819628647214854, 0.5315315315315315, 0.1335559265442404, 0.12742718446601942], [0.0, 0.013262599469496022, 0.04504504504504504, 0.6878130217028381, 0.013349514563106795], [0.0016366612111292963, 0.010610079575596816, 0.17657657657657658, 0.018363939899833055, 0.8179611650485437]]
 
#                     )

# mat = [[8027,6,5,0,0,15],
#                 [83,408,41,0,1,66],
#                 [21,15,3470,57,4,49],
#                 [5,3,85,520,50,0],
#                 [3,0,7,43,592,0],
#                 [23,26,76,0,0,1474]
#                 ]
# harvest = np.array(mat).astype(int)
# True_lable = ["N3", "N2", "N1", "REM","W"]
# Pred_lable = ["N3", "N2", "N1", "REM","W"]

def draw(harvest,
    True_lable = ["N3", "N2", "N1", "REM","W"],
    Pred_lable = ["N3", "N2", "N1", "REM","W"],
    name = 'train'):
    
    harvest = harvest.astype(float)
    wide = harvest.shape[0]
    for i in range(wide):
        harvest[i,:]=harvest[i,:]/np.sum(harvest[i])

    # plt.close()
    # plt.figure('confusion_mat')
    fig, ax = plt.subplots()
    ax.set_ylabel('True',fontsize=FontSize)
    ax.set_xlabel('Pred',fontsize=FontSize)
    im = heatmap(harvest, True_lable, Pred_lable, ax=ax,
                       cmap="Wistia")
    try:
        texts = annotate_heatmap(im, valfmt="{x:.2f}")
    except Exception as e:
        print('Draw heatmap error:',e)
    
    fig.tight_layout()
    plt.savefig(name+'_heatmap.png')
    # plt.show()
    # del fig
    # plt.pause(1)
    plt.close('all')



