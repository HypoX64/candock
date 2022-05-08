import os
import numpy as np
import matplotlib
import torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from sklearn.decomposition import PCA

colors= ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
markers = ['o','^','.',',','v','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_']
def getcolor(num,mode='color'):
    colors = []
    if mode == colors:
        step = 255*3/50
        for i in range(1,num+1):
            if step*i < 255:
                colors.append([step*i/255,0,0])
            elif 255<=step*i <255*2:
                colors.append([1,(step*i-255)/255,0])
            else:
                colors.append([1,1,(step*i-255*2)/255])
    else:
        step = 255/50
        for i in range(1,num+1):
            colors.append([0,step*i/255,0])
    return colors
#---------------------------------heatmap---------------------------------

"""
heatmap: https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
choose color:https://matplotlib.org/tutorials/colors/colormaps.html?highlight=wistia
      recommend:  YlGn  Wistia Blues YlOrBr
"""
def create_heatmap(data, row_labels, col_labels, ax=None,
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
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="left",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
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
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def draw_heatmap(mat,opt,name = 'train',step=0):
    if 'merge' in name:
        label_name = opt.mergelabel_name
    else:
        label_name = opt.label_name
    
    mat = mat.astype(float)
    for i in range(mat.shape[0]):
        mat[i,:]=mat[i,:]/np.sum(mat[i])*100
    if len(mat)>8:
        fig, ax = plt.subplots(figsize=(len(mat)+2.5, len(mat)))
    else:
        fig, ax = plt.subplots()
    ax.set_ylabel('True',fontsize=12)
    ax.set_xlabel('Pred',fontsize=12)

    im, cbar = create_heatmap(mat, label_name, label_name, ax=ax,
                       cmap="Blues", cbarlabel="percentage")
    texts = annotate_heatmap(im,valfmt="{x:.1f}%")

    fig.tight_layout()
    opt.TBGlobalWriter.add_figure('confusion matrix', figure=fig, global_step=step )

    # plt.show()
    # plt.savefig(os.path.join(opt.save_dir,name+'_heatmap.png'))
    plt.close('all')


#---------------------------------loss---------------------------------

def draw_loss(plot_result,epoch,opt):
    train = np.array(plot_result['train'])
    val = np.array(plot_result['eval'])
    plt.figure('running loss')
    plt.clf()
    train_x = np.linspace(0,epoch,len(train))
    test_x = np.linspace(0,int(epoch),len(val))
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    if epoch <10:
        plt.xlim((0,10))
    else:
        plt.xlim((0,epoch))
    plt.plot(train_x,train,label='train',linewidth = 1.5)
    plt.plot(test_x,val,label='eval', linewidth = 1.5)
    plt.legend(loc=1)
    plt.title('Running loss',fontsize='large')
    plt.savefig(os.path.join(opt.save_dir,'running_loss.png'))


#---------------------------------scatter---------------------------------
def label_statistics(labels):
    labels = (np.array(labels)).astype(np.int64)
    label_num = np.max(labels)+1
    label_cnt = np.zeros(label_num,dtype=np.int64)
    for i in range(len(labels)):
        label_cnt[labels[i]] += 1
    label_cnt_per = label_cnt/len(labels)
    return label_cnt,label_cnt_per,label_num

def draw_dml(opt,embeddings,labels,step,n_max_plot=10):
    fig = plt.figure()
    # print(type(embeddings))
    if isinstance(embeddings,torch.Tensor) and isinstance(labels,torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    embeddings = embeddings[np.argsort(labels[:,0])]
    labels = labels[np.argsort(labels[:,0])]
    length,n_embedding = embeddings.shape
    label_cnt,_,label_num = label_statistics(labels[:,0])
    # print(label_num)
    colors = getcolor(label_num,mode='blue')
    if n_embedding>2:
        pca=PCA(n_components=2) 
        embeddings=pca.fit_transform(embeddings)

    cnt = 0
    for i in range(min(label_num,n_max_plot)):
        plt.scatter(
            (embeddings[cnt:cnt+label_cnt[i],0])[:100], 
            (embeddings[cnt:cnt+label_cnt[i],1])[:100],
            label=str(i),
            # color = tuple(colors[i])
        )
        cnt += label_cnt[i]

    plt.title('Autoencoder Embedding Result')
    plt.legend(loc=2)
    opt.TBGlobalWriter.add_figure('dml_embedding',fig,step)
    plt.close('all')


def draw_gan_result(real_signal,gan_signal,opt):
    if real_signal.shape[0]>4:
        fig = plt.figure(figsize=(18,4))
        for i in range(4):
            plt.subplot(2,4,i+1)
            plt.plot(real_signal[i][0])
            plt.title('real')
        for i in range(4):
            plt.subplot(2,4,4+i+1)
            plt.plot(gan_signal[i][0])
            plt.title('gan')
    else:       
        plt.subplot(211)
        plt.plot(real_signal[0][0])
        plt.title('real')
        plt.subplot(212)
        plt.plot(gan_signal[0][0])
        plt.title('gan')
    plt.savefig(os.path.join(opt.save_dir,'gan_result.png'))
    plt.close('all')

def draw_eg_spectrums(spectrums,opt):
    if len(spectrums) > 1:
        fig, ax = plt.subplots(figsize=(6.4*2,4.8*2))
        for i in range(len(spectrums)):
            plt.subplot(len(spectrums)//2+1,2,i+1)
            plt.imshow(spectrums[i])
    else:
        fig = plt.figure()
        plt.imshow(spectrums[0])
    opt.TBGlobalWriter.add_figure('spectrum_eg',figure=fig)
    # plt.savefig(os.path.join(opt.save_dir,'spectrum_eg.jpg'))
    plt.close('all')

def draw_eg_signals(signals,opt):
    fig = plt.figure()
    if len(signals) > 1:
        fig, ax = plt.subplots(figsize=(6.4*2,4.8*2))
        for i in range(len(signals)):
            plt.subplot(len(signals)//2+1,2,i+1)
            plt.plot(signals[i])
    else:
        fig = plt.figure()
        plt.plot(signals[0])

    opt.TBGlobalWriter.add_figure('signal_eg',figure=fig)

    #plt.savefig(os.path.join(opt.save_dir,'signal_eg.jpg'))
    plt.close('all')


#---------------------------------plot final evaluation on tensorboard---------------------------------
def final(opt,results):
    f1 = []; err = []; loss = []
    for fold in results:
        f1.append(results[fold]['F1'])
        err.append(results[fold]['err'])
        loss.append(results[fold]['loss'])

    f1 = np.mean(np.array(f1),axis=0)
    err = np.mean(np.array(err),axis=0)
    loss = np.mean(np.array(loss),axis=0)

    for epoch in range(opt.n_epochs):
        opt.TBGlobalWriter.add_scalars('final'+'/F1', {'eval':f1[epoch]}, epoch)
        opt.TBGlobalWriter.add_scalars('final'+'/Top1.err', {'eval':err[epoch]}, epoch)
        opt.TBGlobalWriter.add_scalars('final'+'/loss', {'loss':loss[epoch]}, epoch)