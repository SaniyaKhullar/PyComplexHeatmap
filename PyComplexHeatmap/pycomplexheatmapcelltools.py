import numpy as np
import pandas as pd
import collections
import matplotlib
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
import matplotlib.patches as patches
from tqdm.auto import tqdm
mm2inch=1/25.4
import fastcluster
import os, sys
from scipy.cluster import hierarchy
import copy
import io
from matplotlib import rcParams
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Use an available font
plt.rcParams['pdf.fonttype']=42

# Please note :) Saniya Khullar adapted this code from PyComplexHeatmap
#  for the CELL-TOOLS package
# =============================================================================
def set_default_style():
    from matplotlib import rcParams
    D={
        'font.family':['sans serif'], #'serif',
        # 'mathtext.fontset':'dejavuserif',
        'font.sans-serif':['DejaVu Sans'],
        'pdf.fonttype':42,

        # Remove legend frame
        'legend.frameon': True,
        'legend.fontsize': 10,

        # Savefig
        'figure.dpi': 100,
        'savefig.bbox': 'tight',
        'savefig.dpi':300,
        'savefig.pad_inches': 0.05
    }
    
    rcParams.update(D)
# =============================================================================
def _check_mask(data, mask):
    """

    Ensure that data and mask are compatible and add missing values and infinite values.
    Values will be plotted for cells where ``mask`` is ``False``.
    ``data`` is expected to be a DataFrame; ``mask`` can be an array or
    a DataFrame.
    """
    if mask is None:
        mask = np.zeros(data.shape, bool)
        #print("if mask is None")
    if isinstance(mask, np.ndarray):
        if mask.shape != data.shape:
            raise ValueError("Mask must have the same shape as data.")
        mask = pd.DataFrame(mask, index=data.index, columns=data.columns, dtype=bool)
        #print("mask = pd.DataFrame(mask, index=data.index, columns=data.columns, dtype=bool)")
    elif isinstance(mask, pd.DataFrame):
        #print(" isinstance(mask, pd.DataFrame)")
        if not mask.index.equals(data.index) and mask.columns.equals(data.columns):
            err = "Mask must have the same index and columns as data."
            raise ValueError(err)

    # Add any cells with missing values or infinite values to the mask
    mask = mask | pd.isnull(data) | np.logical_not(np.isfinite(data))
    #print("_check_mask mask", mask)
    return mask
# =============================================================================
def _calculate_luminance(color):
    """
    Calculate the relative luminance of a color according to W3C standards

    Parameters
    ----------
    color : matplotlib color or sequence of matplotlib colors
        Hex code, rgb-tuple, or html color name.
    Returns
    -------
    luminance : float(s) between 0 and 1

    """
    rgb = matplotlib.colors.colorConverter.to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= .03928, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    lum = rgb.dot([.2126, .7152, .0722])
    try:
        return lum.item()
    except ValueError:
        return lum
# =============================================================================
def get_colormap(cmap):
	try:
		return plt.colormaps.get(cmap)  # matplotlib >= 3.5.1?
	except:
		return plt.get_cmap(cmap)  # matplotlib <=3.4.3?

    
    
def define_cmap(plot_data, vmin=None, vmax=None, cmap=None, center=None, robust=True,
                          na_col='white'):
    """Use some heuristics to set good defaults for colorbar and range."""
    # plot_data is a np.ma.array instance
    # plot_data=np.ma.masked_where(np.asarray(plot_data), plot_data)
    # calc_data = plot_data.astype(float).filled(np.nan)
    if vmin is None:
        if robust:
            vmin = np.nanpercentile(plot_data, 2)
        else:
            vmin = np.nanmin(plot_data)
    if vmax is None:
        if robust:
            vmax = np.nanpercentile(plot_data, 98)
        else:
            vmax = np.nanmax(plot_data)

    # Choose default colormaps if not provided
    if cmap is None:
        if center is None:
            cmap = 'jet'
        else:
            cmap = 'exp1'
    if isinstance(cmap, str):
        cmap1 = matplotlib.cm.get_cmap(cmap).copy()
    elif isinstance(cmap, list):
        cmap1 = matplotlib.colors.ListedColormap(cmap)
    else:
        cmap1 = cmap

    cmap1.set_bad(color=na_col)  # set the color for NaN values
    # Recenter a divergent colormap
    if center is not None:
        # bad = cmap1(np.ma.masked_invalid([np.nan]))[0]  # set the first color as the na_color
        under = cmap1(-np.inf)
        over = cmap1(np.inf)
        under_set = under != cmap1(0)
        over_set = over != cmap1(cmap1.N - 1)

        vrange = max(vmax - center, center - vmin)
        normalize = matplotlib.colors.Normalize(center - vrange, center + vrange)
        cmin, cmax = normalize([vmin, vmax])
        cc = np.linspace(cmin, cmax, 256)
        cmap1 = matplotlib.colors.ListedColormap(cmap1(cc))
        # cmap1.set_bad(bad)
        if under_set:
            cmap1.set_under(under)  # set the color of -np.inf as the color for low out-of-range values.
        if over_set:
            cmap1.set_over(over)
    else:
        normalize = matplotlib.colors.Normalize(vmin, vmax)
    return cmap1,normalize
# =============================================================================
def despine(fig=None, ax=None, top=True, right=True, left=False,
            bottom=False):
    """
    Remove the top and right spines from plot(s).

    Parameters
    ----------
    fig : matplotlib figure, optional
        Figure to despine all axes of, defaults to the current figure.
    ax : matplotlib axes, optional
        Specific axes object to despine. Ignored if fig is provided.
    top, right, left, bottom : boolean, optional
        If True, remove that spine.

    Returns
    -------
    None

    """
    if fig is None and ax is None:
        axes = plt.gcf().axes
    elif fig is not None:
        axes = fig.axes
    elif ax is not None:
        axes = [ax]

    for ax_i in axes:
        for side in ["top", "right", "left", "bottom"]:
            is_visible = not locals()[side]
            ax_i.spines[side].set_visible(is_visible)
        if left and not right: #remove left, keep right
            maj_on = any(t.tick1line.get_visible() for t in ax_i.yaxis.majorTicks)
            min_on = any(t.tick1line.get_visible() for t in ax_i.yaxis.minorTicks)
            ax_i.yaxis.set_ticks_position("right")
            for t in ax_i.yaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.yaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if bottom and not top:
            maj_on = any(t.tick1line.get_visible() for t in ax_i.xaxis.majorTicks)
            min_on = any(t.tick1line.get_visible() for t in ax_i.xaxis.minorTicks)
            ax_i.xaxis.set_ticks_position("top")
            for t in ax_i.xaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.xaxis.minorTicks:
                t.tick2line.set_visible(min_on)
# =============================================================================
def _draw_figure(fig):
    """
    Force draw of a matplotlib figure, accounting for back-compat.

    """
    # See https://github.com/matplotlib/matplotlib/issues/19197 for context
    fig.canvas.draw()
    if fig.stale:
        try:
            fig.draw(fig.canvas.get_renderer())
        except AttributeError:
            pass
# =============================================================================
def axis_ticklabels_overlap(labels):
    """
    Return a boolean for whether the list of ticklabels have overlaps.

    Parameters
    ----------
    labels : list of matplotlib ticklabels
    Returns
    -------
    overlap : boolean
        True if any of the labels overlap.

    """
    if not labels:
        return False
    try:
        bboxes = [l.get_window_extent() for l in labels]
        overlaps = [b.count_overlaps(bboxes) for b in bboxes]
        return max(overlaps) > 1
    except RuntimeError:
        # Issue on macos backend raises an error in the above code
        return False
# =============================================================================
# =============================================================================
def _skip_ticks(labels, tickevery):
    """Return ticks and labels at evenly spaced intervals."""
    n = len(labels)
    if tickevery == 0:
        ticks, labels = [], []
    elif tickevery == 1:
        ticks, labels = np.arange(n) + .5, labels
    else:
        start, end, step = 0, n, tickevery
        ticks = np.arange(start, end, step) + .5
        labels = labels[start:end:step]
    return ticks, labels
# =============================================================================
def _auto_ticks(ax, labels, axis):
    """Determine ticks and ticklabels that minimize overlap."""
    transform = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(transform)
    size = [bbox.width, bbox.height][axis]
    axis = [ax.xaxis, ax.yaxis][axis]
    tick, = axis.set_ticks([0])
    fontsize = tick.label1.get_size()
    max_ticks = int(size // (fontsize / 72))
    if max_ticks < 1:
        return [], []
    tick_every = len(labels) // max_ticks + 1
    tick_every = 1 if tick_every == 0 else tick_every
    ticks, labels =_skip_ticks(labels, tick_every)
    return ticks, labels
# =============================================================================
def to_utf8(obj):
    """
    Return a string representing a Python object.
    Strings (i.e. type ``str``) are returned unchanged.
    Byte strings (i.e. type ``bytes``) are returned as UTF-8-decoded strings.
    For other objects, the method ``__str__()`` is called, and the result is
    returned as a string.

    Parameters
    ----------
    obj : object
        Any Python object
    Returns
    -------
    s : str
        UTF-8-decoded string representation of ``obj``

    """
    if isinstance(obj, str):
        return obj
    try:
        return obj.decode(encoding="utf-8")
    except AttributeError:  # obj is not bytes-like
        return str(obj)
# =============================================================================
def _index_to_label(index):
    """
    Convert a pandas index or multiindex to an axis label.

    """
    if isinstance(index, pd.MultiIndex):
        return "-".join(map(to_utf8, index.names))
    else:
        return index.name
# =============================================================================
def _index_to_ticklabels(index):
    """
    Convert a pandas index or multiindex into ticklabels.

    """
    if isinstance(index, pd.MultiIndex):
        return ["-".join(map(to_utf8, i)) for i in index.values]
    else:
        return index.values
# =============================================================================
def cluster_labels(labels=None,xticks=None,majority=True):
    """
    Merge the adjacent labels into one.

    Parameters
    ----------
    labels : a list of labels.
    xticks : a list of x or y ticks coordinates.
    majority: if majority=True, keep the labels with the largest clusters.

    Returns
    -------
    labels,ticks: merged labels and ticks coordinates.

    Examples
    -------
    labels=['A','A','B','B','A','C','C','B','B','B','C']
    xticks=list(range(len(labels)))
    new_labels,x=cluster_labels(labels,xticks)

    """
    clusters_x = collections.defaultdict(list)
    clusters_labels = {}
    scanned_labels = ''
    i = 0
    for label, x in zip(labels, xticks):
        if label != scanned_labels:
            scanned_labels = label
            i += 1
            clusters_labels[i] = scanned_labels
        clusters_x[i].append(x)
    if majority:
        cluster_size=collections.defaultdict(int)
        largest_cluster={}
        for i in clusters_labels:
            if len(clusters_x[i]) > cluster_size[clusters_labels[i]]:
                cluster_size[clusters_labels[i]]=len(clusters_x[i])
                largest_cluster[clusters_labels[i]]=i
        labels = [clusters_labels[i] for i in clusters_x if i==largest_cluster[clusters_labels[i]]]
        x = [np.mean(clusters_x[i]) for i in clusters_x if i==largest_cluster[clusters_labels[i]]]
        return labels, x

    labels = [clusters_labels[i] for i in clusters_x]
    x = [np.mean(clusters_x[i]) for i in clusters_x]
    return labels, x
# =============================================================================
def plot_color_dict_legend(D=None, ax=None, title=None, color_text=True,
                           label_side='right',kws=None):
    """
    plot legned for color dict

    Parameters
    ----------
    D: a dict, key is categorical variable, values are colors.
    ax: axes to plot the legend.
    title: title of legend.
    color_text: whether to change the color of text based on the color in D.
    label_side: right of left.
    kws: kws passed to plt.legend.

    Returns
    -------
    ax.legend

    """
    if ax is None:
        ax=plt.gca()
    lgd_kws=kws.copy() if not kws is None else {} #bbox_to_anchor=(x,-0.05)
    lgd_kws.setdefault("frameon",True)
    lgd_kws.setdefault("ncol", 1)
    lgd_kws['loc'] = 'upper left'
    lgd_kws['bbox_transform'] = ax.figure.transFigure
    lgd_kws.setdefault('borderpad',0.1 * mm2inch * 72)  # 0.1mm
    lgd_kws.setdefault('markerscale',1)
    lgd_kws.setdefault('handleheight',1)  # font size, units is points
    lgd_kws.setdefault('handlelength',1)  # font size, units is points
    lgd_kws.setdefault('borderaxespad',0.1) #The pad between the axes and legend border, in font-size units.
    lgd_kws.setdefault('handletextpad',0.3) #The pad between the legend handle and text, in font-size units.
    lgd_kws.setdefault('labelspacing',0.1)  # gap height between two Patches,  0.05*mm2inch*72
    lgd_kws.setdefault('columnspacing', 1)
    lgd_kws.setdefault('bbox_to_anchor',(0,1))
    if label_side=='left':
        lgd_kws.setdefault("markerfirst", False)
        align = 'right'
    else:
        lgd_kws.setdefault("markerfirst", True)
        align='left'

    availabel_height=ax.figure.get_window_extent().height * lgd_kws['bbox_to_anchor'][1]
    l = [mpatches.Patch(color=c, label=l) for l, c in D.items()] #kws:?mpatches.Patch; rasterized=True
    L = ax.legend(handles=l, title=title,**lgd_kws)
    ax.figure.canvas.draw()
    while L.get_window_extent().height > availabel_height:
        # ax.cla()
        print("Incresing ncol")
        lgd_kws['ncol']+=1
        if lgd_kws['ncol']>=3:
            print("More than 3 cols is not supported")
            L.remove()
            return None
        L = ax.legend(handles=l, title=title, **lgd_kws)
        ax.figure.canvas.draw()
    L._legend_box.align = align
    if color_text:
        for text in L.get_texts():
            try:
                lum = _calculate_luminance(D[text.get_text()])
                text_color = 'black' if lum > 0.408 else D[text.get_text()]
                text.set_color(text_color)
            except:
                pass
    ax.add_artist(L)
    ax.grid(False)
    # print(availabel_height,L.get_window_extent().height)
    return L
# =============================================================================
def plot_cmap_legend(cax=None,ax=None,cmap='turbo',label=None,kws=None,label_side='right'):
    """
    Plot legend for cmap.

    Parameters
    ----------
    cax : Axes into which the colorbar will be drawn.
    ax :  axes to anchor.
    cmap : turbo, hsv, Set1, Dark2, Paired, Accent,tab20,exp1,exp2,meth1,meth2
    label : title for legend.
    kws : kws passed to plt.colorbar.
    label_side : right or left.

    Returns
    -------
    cbar: axes of legend

    """
    label='' if label is None else label
    cbar_kws={} if kws is None else kws.copy()
    cbar_kws.setdefault('label',label)
    # cbar_kws.setdefault("aspect",3)
    cbar_kws.setdefault("orientation","vertical")
    # cbar_kws.setdefault("use_gridspec", True)
    # cbar_kws.setdefault("location", "bottom")
    cbar_kws.setdefault("fraction", 1)
    cbar_kws.setdefault("shrink", 1)
    cbar_kws.setdefault("pad", 0)
    vmax=cbar_kws.pop('vmax',1)
    vmin=cbar_kws.pop('vmin',0)
    # print(vmin,vmax,'vmax,vmin')
    cax.set_ylim([vmin,vmax])
    cbar_kws.setdefault("ticks",[vmin,(vmax+vmin)/2,vmax])
    m = plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
        cmap=cmap)
    cax.yaxis.set_label_position(label_side)
    cax.yaxis.set_ticks_position(label_side)
    cbar=ax.figure.colorbar(m,cax=cax,**cbar_kws) #use_gridspec=True
    return cbar
# =============================================================================
def plot_marker_legend(obj=None, ax=None, title=None, color_text=True,
                       label_side='right',kws=None):
    """
    plot legned for different marker

    Parameters
    ----------
    D: a dict, key is categorical variable, values are marker.
    ax: axes to plot the legend.
    title: title of legend.
    color_text: whether to change the color of text based on the color in D.
    label_side: right of left.
    kws: kws passed to plt.legend.

    Returns
    -------
    ax.legend

    """
    if ax is None:
        ax=plt.gca()
    markers,colors,ms = obj
    # markers = {'A': 'o', 'B': 's', 'C': 'D'}
    # color_dict = {'A': 'red', 'B': 'blue', 'C': 'green'}
    if colors is None:
        colors='black'
    elif type(colors)==dict:
        color_dict=colors
    if type(colors)==str:
        color_dict = {}
        for k in markers:
            color_dict[k]=colors

    lgd_kws=kws.copy() if not kws is None else {} #bbox_to_anchor=(x,-0.05)
    lgd_kws.setdefault("frameon", True)
    lgd_kws.setdefault("ncol", 1)
    lgd_kws['loc'] = 'upper left'
    lgd_kws['bbox_transform'] = ax.figure.transFigure
    lgd_kws.setdefault('borderpad', 0.1 * mm2inch * 72)  # 0.1mm
    if ms is None:
        s=lgd_kws.pop('markersize',10)
        ms_dict={} #key is label (markers.keys), values is markersize.
        for k in markers:
            ms_dict[k]=s
    elif type(ms) != dict:
        ms_dict = {}
        for k in markers:
            ms_dict[k] = ms
    else:
        ms_dict=ms

    lgd_kws.setdefault('markerscale', 1)
    lgd_kws.setdefault('handleheight', 1)  # font size, units is points
    lgd_kws.setdefault('handlelength', 1)  # font size, units is points
    lgd_kws.setdefault('borderaxespad', 0.1)  # The pad between the axes and legend border, in font-size units.
    lgd_kws.setdefault('handletextpad', 0.3)  # The pad between the legend handle and text, in font-size units.
    lgd_kws.setdefault('labelspacing', 0.5)  # gap height between two Patches,  0.05*mm2inch*72
    lgd_kws.setdefault('columnspacing', 1)
    lgd_kws.setdefault('bbox_to_anchor', (0, 1))
    if label_side=='left':
        lgd_kws.setdefault("markerfirst", False)
        align = 'right'
    else:
        lgd_kws.setdefault("markerfirst", True)
        align='left'

    availabel_height = ax.figure.get_window_extent().height * lgd_kws['bbox_to_anchor'][1]
    # print(ms_dict,markers)
    L = [mlines.Line2D([], [], color=color_dict.get(l,'black'), marker=m, linestyle='None',
                          markersize=ms_dict.get(l,10), label=l)
         for l, m in markers.items()] #kws:?mpatches.Patch; rasterized=True
    ms=lgd_kws.pop('markersize',10)
    Lgd = ax.legend(handles=L, title=title,**lgd_kws)
    ax.figure.canvas.draw()
    while Lgd.get_window_extent().height > availabel_height:
        print("Incresing ncol")
        lgd_kws['ncol']+=1
        if lgd_kws['ncol']>=3:
            print("More than 3 cols is not supported")
            Lgd.remove()
            return None
        Lgd = ax.legend(handles=L, title=title, **lgd_kws)
        ax.figure.canvas.draw()
    Lgd._legend_box.align = align
    if color_text:
        for text in Lgd.get_texts():
            try:
                lum = _calculate_luminance(color_dict[text.get_text()])
                text_color = 'black' if lum > 0.408 else color_dict[text.get_text()]
                text.set_color(text_color)
            except:
                pass
    ax.add_artist(Lgd)
    ax.grid(False)
    # print(availabel_height,L.get_window_extent().height)
    return Lgd
# =============================================================================
def cal_legend_width(legend_list):
    lgd_w=4.5
    legend_width=0
    for lgd in legend_list:
        obj, title, legend_kws, n, lgd_t = lgd
        if lgd_t=='color_dict':
            max_text_len=max(len(str(title)),max([len(str(k)) for k in obj]))
            fontsize = legend_kws.get('fontsize',plt.rcParams['legend.fontsize'])
            lgd_w=max_text_len * fontsize * 0.65 / 72 / mm2inch #point to inches to mm. in average, width = height * 0.6
        elif lgd_t=="markers":
            max_text_len = len(str(title))
            fontsize = legend_kws.get('fontsize', plt.rcParams['legend.fontsize'])
            lgd_w = max_text_len * fontsize * 0.65 / 72 / mm2inch
        if legend_width < lgd_w:
            legend_width=lgd_w
    return legend_width

def plot_legend_list(legend_list=None,ax=None,space=0,legend_side='right',
                     y0=None,gap=2,delta_x=None,legend_width=None,legend_vpad=5,
                     cmap_width=4.5):
    """
    Plot all lengends for a given legend_list.

    Parameters
    ----------
    legend_list : a list including [handles(dict) / cmap / markers dict, title, legend_kws, height, legend_type]
    ax :axes to plot.
    space : unit is pixels.
    legend_side :right, or left
    y0 : the initate coordinate of y for the legend.
    gap : gap between legends, default is 2mm.
    legend_width: width of the legend, default is 4.5mm.

    Returns
    -------
    legend_axes,boundry:

    """
    if ax is None:
        print("No ax was provided, using plt.gca()")
        ax=plt.gca()
        ax.set_axis_off()
        left=ax.get_position().x0+ax.yaxis.labelpad*2/ax.figure.get_window_extent().width if delta_x is None else ax.get_position().x0+delta_x
    else:
        #labelpad: Spacing in points, pad is the fraction relative to x1.
        pad = (space+ax.yaxis.labelpad*1.2*ax.figure.dpi / 72) / ax.figure.get_window_extent().width if delta_x is None else delta_x #labelpad unit is points
        left=ax.get_position().x1 + pad
    if legend_width is None:
        legend_width=cal_legend_width(legend_list) + 2.5 #base width for color rectangle is set to 2.5 mm
        print(f"Estimated legend width: {legend_width} mm")
    legend_width=legend_width*mm2inch*ax.figure.dpi / ax.figure.get_window_extent().width #mm to px to fraction
    cmap_width = cmap_width * mm2inch * ax.figure.dpi / ax.figure.get_window_extent().width  # mm to px to fraction
    if legend_side=='right':
        ax_legend=ax.figure.add_axes([left,ax.get_position().y0,legend_width,ax.get_position().height]) #left, bottom, width, height
    legend_axes=[ax_legend]
    cbars=[]
    leg_pos = ax_legend.get_position() #left bototm: x0,y0; top right: x1,y1

    # y is the bottom position of the first legend (from top to the bottom)
    y = leg_pos.y1 - legend_vpad*mm2inch * ax.figure.dpi / ax.figure.get_window_extent().height if y0 is None else y0
    lgd_col_max_width=0 #the maximum width of all legends in one column
    v_gap=round(gap*mm2inch*ax.figure.dpi/ax.figure.get_window_extent().height,2) #2mm vertically height gap between two legends
    i=0
    while i <= len(legend_list)-1:
        obj, title, legend_kws, n, lgd_t = legend_list[i]
        ax1 = legend_axes[-1] #ax for the legend on the right
        ax1.set_axis_off()
        color_text=legend_kws.pop("color_text",True)
        if lgd_t=='cmap': #type(obj)==str: # a cmap, plot colorbar
            f = 15 * mm2inch * ax.figure.dpi / ax.figure.get_window_extent().height  # 15 mm
            if y-f < 0: #add a new column of axes to plot legends
                offset=(lgd_col_max_width + ax.yaxis.labelpad * 2) / ax.figure.get_window_extent().width
                ax2=ax.figure.add_axes(rect=[ax1.get_position().x0+offset, ax.get_position().y0, cmap_width, ax.get_position().height]) #left_pos.width
                legend_axes.append(ax2)
                ax1=legend_axes[-1]
                ax1.set_axis_off()
                leg_pos = ax1.get_position()
                y = leg_pos.y1 - legend_vpad * mm2inch * ax.figure.dpi / ax.figure.get_window_extent().height if y0 is None else y0
                lgd_col_max_width = 0

            cax=ax1.figure.add_axes(rect=[leg_pos.x0,y-f,cmap_width,f],
                                   xmargin=0,ymargin=0) #unit is fractions of figure width and height
            # [i.set_linewidth(0.5) for i in cax.spines.values()]
            cax.figure.subplots_adjust(bottom=0) #wspace=0, hspace=0
            #https://matplotlib.org/stable/api/figure_api.html
            #[left, bottom, width, height],sharex=True,anchor=(0,0),frame_on=False.
            cbar=plot_cmap_legend(ax=ax1,cax=cax,cmap=obj,label=title,label_side=legend_side,kws=legend_kws)
            cbar_width=cbar.ax.get_window_extent().width
            cbars.append(cbar)
            if cbar_width > lgd_col_max_width:
                lgd_col_max_width=cbar_width
        elif lgd_t == 'color_dict':
            # print(obj, title, legend_kws)
            legend_kws['bbox_to_anchor']=(leg_pos.x0,y) #lower left position of the box.
            #x, y, width, height #kws['bbox_transform'] = ax.figure.transFigure
            # ax1.scatter(leg_pos.x0,y,s=6,color='red',zorder=20,transform=ax1.figure.transFigure)
            L = plot_color_dict_legend(D=obj, ax=ax1, title=title, label_side=legend_side,
                                       color_text=color_text, kws=legend_kws)
            if L is None:
                print("Legend too long, generating a new column..")
                pad = (lgd_col_max_width + ax.yaxis.labelpad * 2) / ax.figure.get_window_extent().width
                left_pos = ax1.get_position()
                ax2 = ax.figure.add_axes([left_pos.x0 + pad, ax.get_position().y0, left_pos.width, ax.get_position().height])
                legend_axes.append(ax2)
                ax1 = legend_axes[-1]
                ax1.set_axis_off()
                leg_pos = ax1.get_position()
                y = leg_pos.y1 - legend_vpad * mm2inch * ax.figure.dpi / ax.figure.get_window_extent().height if y0 is None else y0
                legend_kws['bbox_to_anchor'] = (leg_pos.x0, y)
                L = plot_color_dict_legend(D=obj, ax=ax1, title=title, label_side=legend_side,
                                           color_text=color_text, kws=legend_kws)
                lgd_col_max_width = 0
            L_width = L.get_window_extent().width
            if L_width > lgd_col_max_width:
                lgd_col_max_width = L_width
            f = L.get_window_extent().height / ax.figure.get_window_extent().height
            cbars.append(L)
        elif lgd_t == 'markers':
            legend_kws['bbox_to_anchor'] = (leg_pos.x0, y)  # lower left position of the box.
            L = plot_marker_legend(obj=obj, ax=ax1, title=title, label_side=legend_side,
                                   color_text=color_text, kws=legend_kws) #obj is a tuple: markers and colors
            if L is None:
                print("Legend too long, generating a new column..")
                pad = (lgd_col_max_width + ax.yaxis.labelpad * 2) / ax.figure.get_window_extent().width
                left_pos = ax1.get_position()
                ax2 = ax.figure.add_axes(
                    [left_pos.x0 + pad, ax.get_position().y0, left_pos.width, ax.get_position().height])
                legend_axes.append(ax2)
                ax1 = legend_axes[-1]
                ax1.set_axis_off()
                leg_pos = ax1.get_position()
                y = leg_pos.y1 - legend_vpad * mm2inch * ax.figure.dpi / ax.figure.get_window_extent().height if y0 is None else y0
                legend_kws['bbox_to_anchor'] = (leg_pos.x0, y)
                L = plot_marker_legend(obj=obj, ax=ax1, title=title, label_side=legend_side,
                                           color_text=color_text, kws=legend_kws)
                lgd_col_max_width = 0
            L_width = L.get_window_extent().width
            if L_width > lgd_col_max_width:
                lgd_col_max_width = L_width
            f = L.get_window_extent().height / ax.figure.get_window_extent().height
            cbars.append(L)

        y = y - f - v_gap
        i+=1

    if legend_side=='right':
        boundry=ax1.get_position().y1+lgd_col_max_width / ax.figure.get_window_extent().width
    else:
        boundry = ax1.get_position().y0 - lgd_col_max_width / ax.figure.get_window_extent().width
    return legend_axes,cbars,boundry
# =============================================================================
set_default_style()



def plot_heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
            annot=None, fmt=".2g",annot_kws=None,
            xticklabels=True, yticklabels=True, mask=None, na_col='white', ax=None,
            linewidths=0, linecolor="white",
                 xlabel=None, ylabel=None,
                 **kwargs):
    """
    Plot heatmap.
        heatmap(self.data2d.loc[rows, cols], ax=ax1,cmap=self.cmap,
                        mask=self.mask.loc[rows, cols], rasterized=self.rasterized,
                        xticklabels='auto', yticklabels='auto', annot=annot1, **self.kwargs)
    Parameters
    ----------
    data: dataframe
        pandas dataframe
    vmax, vmin: float
        the maximal and minimal values for cmap colorbar.
    center, robust:
        the same as seaborn.heatmap
    annot: bool
        whether to add annotation for values
    fmt: str
        annotation format.
    anno_kws: dict
        passed to ax.text
    xticklabels,yticklabels: bool
        whether to show ticklabels

    """

    if isinstance(data, pd.DataFrame):
        plot_data = data.values
    else:
        plot_data = np.asarray(data)
        data = pd.DataFrame(plot_data)
    # Validate the mask and convert to DataFrame
    mask = _check_mask(data, mask)
    #print("mask")
    #print(mask)
    plot_data = np.ma.masked_where(np.asarray(mask), plot_data)
    #print("plot_data")
    #print(plot_data)
    # Get good names for the rows and columns
    if xticklabels is False:
        xticks = []
        xticklabels = []
    else:
        xticks = "auto"
        xticklabels = _index_to_ticklabels(data.columns)

    if yticklabels is False:
        yticks = []
        yticklabels = []
    else:
        yticks = "auto"
        yticklabels = _index_to_ticklabels(data.index)

    # Determine good default values for the colormapping
    calc_data = plot_data.astype(float).filled(np.nan)
    if vmin is None:
        if robust:
            vmin = np.nanpercentile(calc_data, 2)
        else:
            vmin = np.nanmin(calc_data)
    if vmax is None:
        if robust:
            vmax = np.nanpercentile(calc_data, 98)
        else:
            vmax = np.nanmax(calc_data)

    # Choose default colormaps if not provided
    if isinstance(cmap, str):
        try:
            cmap = matplotlib.colormaps[cmap].copy() # cmap = matplotlib.cm.get_cmap(cmap).copy()
        except:
            cmap = matplotlib.colormaps[cmap].copy() #cmap = matplotlib.cm.get_cmap(cmap)

    cmap.set_bad(color=na_col)  # set the color for NaN values
    # Recenter a divergent colormap
    if center is not None:
        # bad = cmap(np.ma.masked_invalid([np.nan]))[0]  # set the first color as the na_color
        under = cmap(-np.inf)
        over = cmap(np.inf)
        under_set = under != cmap(0)
        over_set = over != cmap(cmap.N - 1)

        vrange = max(vmax - center, center - vmin)
        normlize = matplotlib.colors.Normalize(center - vrange, center + vrange)
        cmin, cmax = normlize([vmin, vmax])
        cc = np.linspace(cmin, cmax, 256)
        cmap = matplotlib.colors.ListedColormap(cmap(cc))
        # self.cmap.set_bad(bad)
        if under_set:
            cmap.set_under(under)  # set the color of -np.inf as the color for low out-of-range values.
        if over_set:
            cmap.set_over(over)

    # Sort out the annotations
    if annot is None or annot is False:
        #print("here")
        annot = False
        annot_data = None
    else:
        #print("else annot")
        if isinstance(annot, bool):
            annot_data = plot_data
        else:
            annot_data = np.asarray(annot)
            #print("annot_data", annot_data)
            if annot_data.shape != plot_data.shape:
                err = "`data` and `annot` must have same shape."
                raise ValueError(err)
        annot = True

    if annot_kws is None:
        annot_kws = {}

    # Add the pcolormesh kwargs here
    kwargs["linewidths"] = linewidths
    kwargs["edgecolor"] = linecolor

    # Draw the plot and return the Axes
    despine(ax=ax, left=True, bottom=True)
    if "norm" not in kwargs:
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)

    # Draw the heatmap
    mesh = ax.pcolormesh(plot_data, cmap=cmap, **kwargs)
    # Set the axis limits
    ax.set(xlim=(0, data.shape[1]), ylim=(0, data.shape[0]))
    # Invert the y axis to show the plot in matrix form
    ax.invert_yaxis()  # from top to bottom

    
    # Set axis labels if provided
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    # Add row and column labels
    if isinstance(xticks, str) and xticks == "auto":
        xticks, xticklabels = _auto_ticks(ax, xticklabels, 0)

    if isinstance(yticks, str) and yticks == "auto":
        yticks, yticklabels = _auto_ticks(ax, yticklabels, 1)

    ax.set(xticks=xticks, yticks=yticks)
    xtl = ax.set_xticklabels(xticklabels)
    ytl = ax.set_yticklabels(yticklabels, rotation="vertical")

    _draw_figure(ax.figure)
    if axis_ticklabels_overlap(xtl):
        plt.setp(xtl, rotation="vertical")
    if axis_ticklabels_overlap(ytl):
        plt.setp(ytl, rotation="horizontal")

    # Annotate the cells with the formatted values
    
    # Annotate the cells with the formatted values
    if annot:
        mesh.update_scalarmappable()
        #print("mesh.update_scalarmappable()", mesh.update_scalarmappable())
        height, width = annot_data.shape
        xpos, ypos = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)

        # Get the current mesh array and face colors
        fixed_mesh_array = mesh.get_array().copy()
        fixed_face_col_array = mesh.get_facecolors().copy()

        for x, y, color, val in tqdm(zip(xpos.flat, ypos.flat, fixed_face_col_array, annot_data.flat),
                                     desc = "annotations"):
            #print("mesh.get_array()", mesh.get_array().shape)
            #print("mesh.get_facecolors()", mesh.get_facecolors().shape)
            #print("fixed_mesh_array", fixed_mesh_array.shape)
            #print("fixed_face_col_array", fixed_face_col_array.shape)

            #print(x, y, color, val)

            # Check if the annotation value is not masked
            if not np.ma.is_masked(val) and not np.isnan(val):
                #print("enter here")
                lum = _calculate_luminance(color)
                #print("lum", lum)
                text_color = ".15" if lum > .408 else "w"
                #print("fmt", fmt, "val ", val)
                annotation = ("{:" + fmt + "}").format(val)
                #print("annotation", annotation)

                text_kwargs = dict(color=text_color, ha="center", va="center", fontsize = 11)
                text_kwargs.update(annot_kws)
                ax.text(x, y, annotation, **text_kwargs)
            #else:
                #print("don't make it here")
                #print("val", val)

    ################################            
    # Draw red borders around diagonal cells
    if data.shape[0] == data.shape[1]:
        for i in range(min(data.shape)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='purple', lw=2))
            #rect = patches.Rectangle((i, -1), 1, 1, linewidth=2, edgecolor='purple', facecolor='none')
            #ax.add_patch(rect)            
                
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    return ax


# =============================================================================
class DendrogramPlotterCellTools(object):
    def __init__(self, data, linkage, metric, method, axis, label, rotate, dendrogram_kws=None):
        """Plot a dendrogram of the relationships between the columns of data
        """
        self.axis = axis
        if self.axis == 1:  # default 1, columns, when calculating dendrogram, each row is a point.
            data = data.T
        self.check_array(data)
        self.shape = self.data.shape
        self.metric = metric
        self.method = method
        self.label = label
        self.rotate = rotate
        self.dendrogram_kws = dendrogram_kws if not dendrogram_kws is None else {}
        if linkage is None:
            self.linkage = self.calculated_linkage
        else:
            self.linkage = linkage
        self.dendrogram = self.calculate_dendrogram()
        # Dendrogram ends are always at multiples of 5, who knows why
        ticks = np.arange(self.data.shape[0]) + 0.5  # xticklabels

        if self.label:
            ticklabels = _index_to_ticklabels(self.data.index)
            ticklabels = [ticklabels[i] for i in self.reordered_ind]
            if self.rotate:  # horizonal
                self.xticks = []
                self.yticks = ticks
                self.xticklabels = []

                self.yticklabels = ticklabels
                self.ylabel = _index_to_label(self.data.index)
                self.xlabel = ''
            else:  # vertical
                self.xticks = ticks
                self.yticks = []
                self.xticklabels = ticklabels
                self.yticklabels = []
                self.ylabel = ''
                self.xlabel = _index_to_label(self.data.index)
        else:
            self.xticks, self.yticks = [], []
            self.yticklabels, self.xticklabels = [], []
            self.xlabel, self.ylabel = '', ''

        self.dependent_coord = np.array(self.dendrogram['dcoord'])
        self.independent_coord = np.array(self.dendrogram['icoord']) / 10

    def check_array(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        # To avoid missing values and infinite values and further error, remove missing values
        # nrow = data.shape[0]
        # keep_col = data.apply(np.isfinite).sum() == nrow
        # if keep_col.sum() < 3:
        #     raise ValueError("There are too many missing values or infinite values")
        # data = data.loc[:, keep_col[keep_col].index.tolist()]
        if data.isna().sum().sum() > 0:
            data = data.apply(lambda x: x.fillna(x.median()),axis=1)
        self.data = data
        self.array = data.values

    def _calculate_linkage_scipy(self):  # linkage is calculated by columns
        # print(type(self.array),self.method,self.metric)
        linkage = hierarchy.linkage(self.array, method=self.method, metric=self.metric)
        return linkage  # array is a distance matrix?

    def _calculate_linkage_fastcluster(self):
        import fastcluster
        # Fastcluster has a memory-saving vectorized version, but only
        # with certain linkage methods, and mostly with euclidean metric
        # vector_methods = ('single', 'centroid', 'median', 'ward')
        euclidean_methods = ('centroid', 'median', 'ward')
        euclidean = self.metric == 'euclidean' and self.method in euclidean_methods
        if euclidean or self.method == 'single':
            return fastcluster.linkage_vector(self.array, method=self.method, metric=self.metric)
        else:
            linkage = fastcluster.linkage(self.array, method=self.method, metric=self.metric)
            return linkage

    @property
    def calculated_linkage(self):
        try:
            return self._calculate_linkage_fastcluster()
        except ImportError:
            if np.product(self.shape) >= 1000:
                msg = ("Clustering large matrix with scipy. Installing "
                       "`fastcluster` may give better performance.")
                warnings.warn(msg)
        return self._calculate_linkage_scipy()

    def calculate_dendrogram(self):  # Z (linkage) shape = (n,4), then dendrogram icoord shape = (n,4)
        return hierarchy.dendrogram(self.linkage, no_plot=True, labels=self.data.index.tolist(),
                                    get_leaves=True, **self.dendrogram_kws)  # color_threshold=-np.inf,

    @property
    def reordered_ind(self):
        """Indices of the matrix, reordered by the dendrogram"""
        return self.dendrogram['leaves']  # idx of the matrix

    def plot(self, ax, tree_kws):
        """Plots a dendrogram of the similarities between data on the axes
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object upon which the dendrogram is plotted
        """
        tree_kws = {} if tree_kws is None else tree_kws
        tree_kws.setdefault("linewidth", .5)
        tree_kws.setdefault("colors", None)
        # tree_kws.setdefault("colors", tree_kws.pop("color", (.2, .2, .2)))
        if self.rotate and self.axis == 0:  # 0 is rows, 1 is columns (default)
            coords = zip(self.dependent_coord, self.independent_coord)  # independent is icoord (x), horizontal
        else:
            coords = zip(self.independent_coord, self.dependent_coord)  # vertical
        # lines = LineCollection([list(zip(x,y)) for x,y in coords], **tree_kws)  #
        # ax.add_collection(lines)
        colors = tree_kws.pop('colors')
        if colors is None:
            # colors=self.dendrogram['leaves_color_list']
            colors = ['black'] * len(self.dendrogram['ivl'])
        for (x, y), color in zip(coords, colors):
            ax.plot(x, y, color=color, **tree_kws)
        number_of_leaves = len(self.reordered_ind)
        max_dependent_coord = max(map(max, self.dependent_coord))  # max y
        # if self.axis==0: #TODO
        #     ax.invert_yaxis()  # 20230227 fix the bug for inverse order of row dendrogram

        if self.rotate:  # horizontal
            ax.yaxis.set_ticks_position('right')
            # Constants 10 and 1.05 come from
            # `scipy.cluster.hierarchy._plot_dendrogram`
            ax.set_ylim(0, number_of_leaves)
            # ax.set_xlim(0, max_dependent_coord * 1.05)
            ax.set_xlim(0, max_dependent_coord)
            ax.invert_xaxis()
            ax.invert_yaxis()
        else:  # vertical
            # Constants 10 and 1.05 come from
            # `scipy.cluster.hierarchy._plot_dendrogram`
            ax.set_xlim(0, number_of_leaves)
            ax.set_ylim(0, max_dependent_coord)
        despine(ax=ax, bottom=True, left=True)
        ax.set(xticks=self.xticks, yticks=self.yticks,
               xlabel=self.xlabel, ylabel=self.ylabel)
        xtl = ax.set_xticklabels(self.xticklabels)
        ytl = ax.set_yticklabels(self.yticklabels, rotation='vertical')
        # Force a draw of the plot to avoid matplotlib window error
        # _draw_figure(ax.figure)
        if len(ytl) > 0 and axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation="horizontal")
        if len(xtl) > 0 and axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation="vertical")
        return self
    





# -----------------------------------------------------------------------------
class AnnotationBaseCellTools:
	"""
	Base class for annotation objects.

	Parameters
	----------
	df: dataframe
		a pandas series or dataframe (only one column).
	cmap: str
		colormap, such as Set1, Dark2, bwr, Reds, jet, hsv, rainbow and so on. Please see
		https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html for more information, or run
		matplotlib.pyplot.colormaps() to see all availabel cmap.
		default cmap is 'auto', it would be determined based on the dtype for each columns of df.
	colors: dict, list or str
		a dict or list (for boxplot, barplot) or str.
		If colors is a dict: keys should be exactly the same as df.iloc[:,0].unique(),
		values for the dict should be colors (color names or HEX color).
		If  colors is a list, then the length of this list should be equal to df.iloc[:,0].nunique()
		If colors is a string, means all values in df.iloc[:,0].unique() share the same color.
	height: float
		height (if axis=1) / width (if axis=0) for the annotation size.
	legend: bool
		whether to plot legend for this annotation when legends are plotted or
		plot legend with HeatmapAnnotationCellTools.plot_legends().
	legend_kws: dict
		vmax, vmin and other kws passed to plt.legend, such as title, prop, fontsize, labelcolor,
		markscale, frameon, framealpha, fancybox, shadow, facecolor, edgecolor, mode and so on, for more
		arguments, pleast type ?plt.legend. There is an additional parameter `color_text` (default is True),
		which would set the color of the text to the same color as legend marker. if one set
		`legend_kws={'color_text':False}`, then, black would be the default color for the text.
		If the user want to use a custom color instead of black (such as blue), please set
		legend_kws={'color_text':False,'labelcolor':'blue'}.
	plot_kws: dict
		other plot kws passed to annotation.plot, such as rotation, rotation_mode, ha, va,
		annotation_clip, arrowprops and matplotlib.text.Text for anno_labelCT. For example, in anno_simpleCT,
		there is also kws: vmin and vmax, if one want to change the range, please try:
		anno_simpleCT(df_box.Gene1,vmin=0,vmax=1,legend_kws={'vmin':0,'vmax':1}).

	Returns
	----------
	Class AnnotationBaseCellTools.
	"""
	def __init__(
		self,
		df=None,
		cmap="auto",
		colors=None,
		height=None,
		legend=None,
		legend_kws=None,
		**plot_kws
	):
		self._check_df(df)
		self.label = None
		self.ylim = None
		self.color_dict = None
		self.nrows = self.df.shape[0]
		self.ncols = self.df.shape[1]
		self.height = self._height(height)
		self._type_specific_params()
		self.legend = legend
		self.legend_kws = legend_kws if not legend_kws is None else {}
		self._set_default_plot_kws(plot_kws)

		if colors is None:
			self._check_cmap(cmap)
			self._calculate_colors()  # modify self.plot_data, self.color_dict (each col is a dict)
		else:
			self._check_colors(colors)
			self._calculate_cmap()  # modify self.plot_data, self.color_dict (each col is a dict)
		self.plot_data = self.df.copy()

	def _check_df(self, df):
		if isinstance(df, pd.Series):
			df = df.to_frame()
		if isinstance(df, pd.DataFrame):
			self.df = df
		else:
			raise TypeError("df must be a pandas DataFrame or Series.")

	def _height(self, height):
		return 3 * self.ncols if height is None else height

	def _set_default_plot_kws(self, plot_kws):
		self.plot_kws = {} if plot_kws is None else plot_kws
		self.plot_kws.setdefault("zorder", 10)

	def set_orientation(self, orientation):
		self.orientation = orientation
	def update_plot_kws(self, plot_kws):
		self.plot_kws.update(plot_kws)

	def set_label(self, label):
		self.label = label

	def set_legend(self, legend):
		if self.legend is None:
			self.legend = legend

	def set_axes_kws(self, subplot_ax):
		# ax.set_xticks(ticks=np.arange(1, self.nrows + 1, 1), labels=self.plot_data.index.tolist())
		if self.axis == 1:
			if self.ticklabels_side == "left":
				subplot_ax.yaxis.tick_left()
			elif self.ticklabels_side == "right":
				subplot_ax.yaxis.tick_right()
			subplot_ax.yaxis.set_label_position(self.label_side)
			subplot_ax.yaxis.label.update(self.label_kws)
			# ax.yaxis.labelpad = self.labelpad
			subplot_ax.xaxis.set_visible(False)
			subplot_ax.yaxis.label.set_visible(False)
		else:  # axis=0, row annotation
			if self.ticklabels_side == "top":
				subplot_ax.xaxis.tick_top()
			elif self.ticklabels_side == "bottom":
				subplot_ax.xaxis.tick_bottom()
			subplot_ax.xaxis.set_label_position(self.label_side)
			subplot_ax.xaxis.label.update(self.label_kws)
			subplot_ax.xaxis.set_tick_params(self.ticklabels_kws)
			# ax.yaxis.labelpad = self.labelpad
			subplot_ax.yaxis.set_visible(False)
			subplot_ax.xaxis.label.set_visible(False)

	def _check_cmap(self, cmap):
		if cmap == "auto":
			col = self.df.columns.tolist()[0]
			if self.df.dtypes[col] == object:
				if self.df[col].nunique() <= 10:
					self.cmap = "Set1"
				elif self.df[col].nunique() <= 20:
					self.cmap = "tab20"
				else:
					self.cmap = "random50"
			elif self.df.dtypes[col] == float or self.df.dtypes[col] == int:
				self.cmap = "jet"
			else:
				raise TypeError(
					"Can not assign cmap for column %s, please specify cmap" % col
				)
		elif type(cmap) == str:
			self.cmap = cmap
		else:
			print("WARNING: cmap is not a string!")
			self.cmap = cmap
		if (
			get_colormap(self.cmap).N == 256
		):  # then heatmap will automatically calculate vmin and vmax
			try:
				self.plot_kws.setdefault("vmax", np.nanmax(self.df.values))
				self.plot_kws.setdefault("vmin", np.nanmin(self.df.values))
			except:
				pass

	def _calculate_colors(self):  # add self.color_dict (each col is a dict)
		self.color_dict = {}
		col = self.df.columns.tolist()[0]
		if get_colormap(self.cmap).N < 256 or self.df.dtypes[col] == object:
			cc_list = (
				self.df[col].value_counts().index.tolist()
			)  # sorted by value counts
			self.df[col] = self.df[col].map({v: cc_list.index(v) for v in cc_list})
			for v in cc_list:
				color = get_colormap(self.cmap)(cc_list.index(v))
				self.color_dict[v] = color  # matplotlib.colors.to_hex(color)
		else:  # float
			self.color_dict = {
				v: get_colormap(self.cmap)(v) for v in self.df[col].values
			}
		self.colors = None

	def _check_colors(self, colors):
		assert isinstance(colors,(str,list,dict,tuple))
		if isinstance(colors, str):
			color_dict = {label: colors for label in self.df.iloc[:, 0].unique()}
		elif isinstance(colors, (list,tuple)):
			assert len(colors) == self.df.iloc[:, 0].nunique()
			color_dict = {
				label: color
				for label, color in zip(self.df.iloc[:, 0].unique(), colors)
			}
		else:
			color_dict=colors.copy()
		if len(color_dict) >= self.df.iloc[:, 0].nunique():
			self.colors = color_dict
		else:
			raise TypeError(
				"The length of `colors` is not consistent with the shape of the input data"
			)

	def _calculate_cmap(self):
		self.color_dict = self.colors
		col = self.df.columns.tolist()[0]
		cc_list = list(self.color_dict.keys())  # column values
		self.df[col] = self.df[col].map({v: cc_list.index(v) for v in cc_list})
		self.cmap = matplotlib.colors.ListedColormap([self.color_dict[k] for k in cc_list])
		self.plot_kws.setdefault("vmax", get_colormap(self.cmap).N - 1)
		self.plot_kws.setdefault("vmin", 0)

	def _type_specific_params(self):
		if self.ylim is None:
			Max = np.nanmax(self.df.values)
			Min = np.nanmin(self.df.values)
			gap = Max - Min
			self.ylim = [Min - 0.05 * gap, Max + 0.05 * gap]

	def reorder(self, idx):
		# Before plotting, df needs to be reordered according to the new clustered order.
		self.plot_data = self.df.reindex(idx)  #
		self.plot_data.fillna(np.nan, inplace=True)
		self.nrows = self.plot_data.shape[0]

	def get_label_width(self):
		return self.ax.yaxis.label.get_window_extent(
			renderer=self.ax.figure.canvas.get_renderer()
		).width

	def get_ticklabel_width(self):
		yticklabels = self.ax.yaxis.get_ticklabels()
		if len(yticklabels) == 0:
			return 0
		else:
			return max(
				[
					label.get_window_extent(
						renderer=self.ax.figure.canvas.get_renderer()
					).width
					for label in self.ax.yaxis.get_ticklabels()
				]
			)

	def get_max_label_width(self):
		return max([self.get_label_width(), self.get_ticklabel_width()])


# =============================================================================
class anno_simpleCT(AnnotationBaseCellTools):
	"""
		Annotate simple annotation, categorical or continuous variables.
	"""
	def __init__(
		self,
		df=None,
		cmap="auto",
		colors=None,
		add_text=False,
		majority=True,
		text_kws=None,
		height=None,
		legend=True,
		legend_kws=None,
		**plot_kws
	):
		self.add_text = add_text
		self.majority = majority
		self.text_kws = text_kws if not text_kws is None else {}
		self.plot_kws = plot_kws
		# print(self.plot_kws)
		legend_kws = legend_kws if not legend_kws is None else {}
		if "vmax" in plot_kws:
			legend_kws.setdefault("vmax", plot_kws.get("vmax"))
		if "vmin" in plot_kws:
			legend_kws.setdefault("vmin", plot_kws.get("vmin"))
		super().__init__(
			df=df,
			cmap=cmap,
			colors=colors,
			height=height,
			legend=legend,
			legend_kws=legend_kws,
			**plot_kws
		)

	def _set_default_plot_kws(self, plot_kws):
		self.plot_kws = {} if plot_kws is None else plot_kws
		self.plot_kws.setdefault("zorder", 10)
		self.text_kws.setdefault("zorder", 16)
		self.text_kws.setdefault("ha", "center")
		self.text_kws.setdefault("va", "center")

	def _calculate_colors(self):  # add self.color_dict (each col is a dict)
		self.color_dict = {}
		col = self.df.columns.tolist()[0]
		if get_colormap(self.cmap).N < 256:
			cc_list = (
				self.df[col].value_counts().index.tolist()
			)  # sorted by value counts
			for v in cc_list:
				color = get_colormap(self.cmap)(cc_list.index(v))
				self.color_dict[v] = color  # matplotlib.colors.to_hex(color)
		else:  # float
			cc_list = None
			self.color_dict = {
				v: get_colormap(self.cmap)(v) for v in self.df[col].values
			}
		self.cc_list = cc_list
		self.colors = None

	def _calculate_cmap(self):
		self.color_dict = self.colors
		col = self.df.columns.tolist()[0]
		cc_list = list(self.color_dict.keys())  # column values
		self.cc_list = cc_list
		self.cmap = matplotlib.colors.ListedColormap(
			[self.color_dict[k] for k in cc_list]
		)

	def _type_specific_params(self):
		pass
	def plot(self, ax=None, axis=1):
		if hasattr(self.cmap, "N"):
			vmax = self.cmap.N - 1
		elif type(self.cmap) == str:
			vmax = get_colormap(self.cmap).N - 1
		else:
			vmax = len(self.color_dict) - 1
		self.plot_kws.setdefault("vmax", vmax)  # get_colormap(self.cmap).N
		self.plot_kws.setdefault("vmin", 0)
		if self.cc_list:
			mat = (
				self.plot_data.iloc[:, 0]
				.map({v: self.cc_list.index(v) for v in self.cc_list})
				.values
			)
		else:
			mat = self.plot_data.values
		matrix = mat.reshape(1, -1) if axis == 1 else mat.reshape(-1, 1)
		ax1 = plot_heatmap(
			matrix,
			cmap=self.cmap,
			ax=ax,
			xticklabels=False,
			yticklabels=False,
			**self.plot_kws
		) #y will be inverted inside plot_heatmap
		ax.tick_params(
			axis="both",
			which="both",
			left=False,
			right=False,
			top=False,
			bottom=False,
			labeltop=False,
			labelbottom=False,
			labelleft=False,
			labelright=False,
		)
		if self.add_text:
			if axis == 0:
				self.text_kws.setdefault("rotation", 90)
				self.text_kws.setdefault("rotation_mode", 'anchor')
			labels, ticks = cluster_labels(
				self.plot_data.iloc[:, 0].values,
				np.arange(0.5, self.nrows, 1),
				self.majority,
			)
			n = len(ticks)
			if axis == 1:
				x = ticks
				y = [0.5] * n
			else:
				y = ticks
				x = [0.5] * n
			s = (
				ax.get_window_extent().height
				if axis == 1
				else ax.get_window_extent().width
			)
			self.text_kws.setdefault("fontsize", 72 * s * 0.8 / ax.figure.dpi)
			# fontsize = self.text_kws.pop('fontsize', 72 * s * 0.8 / ax.figure.dpi)
			color = self.text_kws.pop("color", None)
			for x0, y0, t in zip(x, y, labels):
				# print(t,self.color_dict)
				lum = _calculate_luminance(self.color_dict.get(t,'black'))
				if color is None:
					text_color = "black" #if lum > 0.408 else "white"
				else:
					text_color = color
				# print(t,self.color_dict,text_color,color)
				self.text_kws.setdefault("color", text_color)
				ax.text(x0, y0, t, **self.text_kws)
		self.ax = ax
		self.fig = self.ax.figure
		return self.ax


# =============================================================================
class anno_labelCT(AnnotationBaseCellTools):
	"""
	Add label and text annotations. See example on documentatin website:
	https://dingwb.github.io/PyComplexHeatmap/build/html/notebooks/single_cell_methylation.html

	Parameters
	----------
	merge: bool
		whether to merge the same clusters into one and label only once.
	extend: bool
		whether to distribute all the labels extend to the all axis, figure or ax or False.
	frac: float
		fraction of the armA and armB relative to length of connection label, will be passed to
		connectionstyle: f"arc,angleA={angleA},angleB={angleB},armA={arm_height},armB={arm_height},rad={self.rad}",
		frac will be used to calculate arm_height: arm_height = arrow_height * self.frac
	rad: int
		rad of the connection arrow.
	majority: bool
		If there are multiple group for one label, whether to annotate the label in the largest group. [True]
	adjust_color: bool
		When the luminance of the color is too high, use black color replace the original color. [True]
	luminance: float
		luminance values [0-1], used together with adjust_color, when the calculated luminance > luminance,
		the color will be replaced with black. [0.5]
	relpos: tuple
		relpos passed to arrowprops in plt.annotate, tuple (x,y) means the arrow start point position relative to the
		 label. default is (0, 0) if self.orientation == 'top' else (0, 1) for columns labels, (1, 1) if self.orientation == 'left'
		 else (0, 0) for rows labels.
	plot_kws: dict
		passed to plt.annotate, including annotation_clip, arrowprops and matplotlib.text.Text,
		more information about arrowprops could be found in
		matplotlib.patches.FancyArrowPatch. For example, to remove arrow, just set
		arrowprops = dict(visible=False). See: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html for more information.
		arrowprops:
			arrowstyle:
				https://matplotlib.org/stable/gallery/text_labels_and_annotations/fancyarrow_demo.html
			connectionstyle:
				https://matplotlib.org/stable/gallery/userdemo/connectionstyle_demo.html

	Returns
	----------
	Class AnnotationBaseCellTools.

	"""
	def __init__(
		self,
		df=None,
		cmap="auto",
		colors=None,
		merge=False,
		extend=False,
		frac=0.2,
		rad=2,
		majority=True,
		adjust_color=True,
		luminance=0.8,
		height=None,
		legend=False,
		legend_kws=None,
		relpos=None,
		**plot_kws
	):
		super().__init__(
			df=df,
			cmap=cmap,
			colors=colors,
			height=height,
			legend=legend,
			legend_kws=legend_kws,
			**plot_kws
		)
		self.merge = merge
		self.majority = majority
		self.adjust_color = adjust_color
		self.luminance = luminance
		self.extend = extend
		self.frac = frac
		self.rad=rad
		self.relpos = relpos
		self.annotated_texts = []

	def _height(self, height):
		return 4 if height is None else height

	def set_plot_kws(self, axis):
		shrink = 1  # 1 * mm2inch * 72  # 1mm -> points
		if axis == 1:  # columns
			relpos = (
				(0, 0) if self.orientation == "up" else (0, 1)
			)  # position to anchor, x: left -> right, y: down -> top
			rotation = 90 if self.orientation == "up" else -90
			ha = "left"
			va = "center"
		else:
			relpos = (
				(1, 1) if self.orientation == "left" else (0, 0)
			)  # (1, 1) if self.orientation == 'left' else (0, 0)
			rotation = 0
			ha = "right" if self.orientation == "left" else "left"
			va = "center"
		# relpos: The exact starting point position of the arrow is defined by relpos. It's a tuple of relative
		# coordinates of the text box, where (0, 0) is the lower left corner and (1, 1) is the upper right corner.
		# Values <0 and >1 are supported and specify points outside the text box. By default (0.5, 0.5) the starting
		# point is centered in the text box.
		self.plot_kws.setdefault("rotation", rotation)
		self.plot_kws.setdefault("ha", ha)
		self.plot_kws.setdefault("va", va)
		rp = relpos if self.relpos is None else self.relpos
		arrowprops = dict(
			arrowstyle="-",
			color="black",
			shrinkA=shrink,
			shrinkB=shrink,
			relpos=rp,
			patchA=None,
			patchB=None,
			connectionstyle=None,
			linewidth=0.5
		)
		# arrow: ->, from text to point.
		# self.plot_kws.setdefault('transform_rotates_text', False)
		self.plot_kws.setdefault("arrowprops", {})
		for k in arrowprops:
			if k not in self.plot_kws['arrowprops']:
				self.plot_kws['arrowprops'][k]=arrowprops[k]
		self.plot_kws.setdefault("rotation_mode", "anchor")

	def _calculate_colors(self):  # add self.color_dict (each col is a dict)
		self.color_dict = {}
		col = self.df.columns.tolist()[0]
		if get_colormap(self.cmap).N < 256 or self.df.dtypes[col] == object:
			cc_list = (
				self.df[col].value_counts().index.tolist()
			)  # sorted by value counts
			for v in cc_list:
				color = get_colormap(self.cmap)(cc_list.index(v))
				self.color_dict[v] = color  # matplotlib.colors.to_hex(color)
		else:  # float
			self.color_dict = {
				v: get_colormap(self.cmap)(v) for v in self.df[col].values
			}
		self.colors = None

	def _calculate_cmap(self):
		self.color_dict = self.colors
		col = self.df.columns.tolist()[0]
		cc_list = list(self.color_dict.keys())  # column values
		self.cmap = matplotlib.colors.ListedColormap(
			[self.color_dict[k] for k in cc_list]
		)

	def _type_specific_params(self):
		pass

	def plot(self, ax=None, axis=1):  # add self.gs,self.fig,self.ax,self.axes
		self.axis = axis
		if self.orientation is None:
			ax_index = ax.figure.axes.index(ax)
			ax_n = len(ax.figure.axes)
			i = ax_index / ax_n
			if axis == 1 and i <= 0.5:
				orientation = "up"
			elif axis == 1:
				orientation = "down"
			elif axis == 0 and i <= 0.5:
				orientation = "left"
			else:
				orientation = "right"
			self.orientation = orientation
		self.set_plot_kws(axis)
		if (
			self.merge
		):  # merge the adjacent ticklabels with the same text to one, return labels and mean x coordinates.
			labels, ticks = cluster_labels(
				self.plot_data.iloc[:, 0].values,
				np.arange(0.5, self.nrows, 1),
				self.majority,
			)
		else:
			labels = self.plot_data.iloc[:, 0].values
			ticks = np.arange(0.5, self.nrows, 1)
		# labels are the merged labels, ticks are the merged mean x coordinates.

		n = len(ticks)
		arrow_height = self.height * mm2inch * ax.figure.dpi # convert height (mm) to inch and to pixels.
		text_y =  arrow_height
		if axis == 1:
			if self.orientation == "down":
				# ax.invert_yaxis() # top -> bottom
				text_y = -1 * arrow_height
			ax.set_xticks(ticks=np.arange(0.5, self.nrows, 1))
			x = ticks  # coordinate for the arrow start point
			y = [0] * n if self.orientation == "up" else [1] * n  # position for line start on axes
			if self.extend:
				extend_pos = np.linspace(0, 1, n + 1) #0,0.1,0.2,...0.9,1
				x1 = [(extend_pos[i] + extend_pos[i - 1]) / 2 for i in range(1, n + 1)] #coordinates for text: 0.05,0.15..
				y1 = [1] * n if self.orientation == "up" else [0] * n
			else:
				x1 = [0] * n #offset pixels
				y1 = [text_y] * n #offset pixels
		else:
			if self.orientation == "left":
				# ax.invert_xaxis() # right -> left, will not affect ax.get_xaxis_transform()
				text_y = -1 * arrow_height
			ax.set_yticks(ticks=np.arange(0.5, self.nrows, 1))
			y=ticks
			x = [1] * n if self.orientation == "left" else [0] * n #coordinate for start point, side=left, x axis <---
			if self.extend: #ax.transAxes
				# extend_pos = np.linspace(0, 1, n + 1)
				extend_pos = np.linspace(1,0, n + 1) #y, top -> bottom
				y1 = [(extend_pos[i] + extend_pos[i - 1]) / 2 for i in range(1, n + 1)]
				x1 = [1] * n if self.orientation == "right" else [0] * n
			else: #offset pixels
				y1 = [0] * n #vertical distance related to point (anno_simpleCT)
				x1 = [text_y] * n #horizonal distance related to point (anno_simpleCT)
		# angleA is the angle for the data point (clockwise), B is for text.
		# https://matplotlib.org/stable/gallery/userdemo/connectionstyle_demo.html
		xycoords = ax.get_xaxis_transform() if axis == 1 else ax.get_yaxis_transform()
		# get_xaxis_transform: x is data coordinates,y is between [0,1], will not be affected by invert_xaxis()
		if self.extend:
			text_xycoords = ax.transAxes #relative coordinates
		else:
			text_xycoords = "offset pixels"
		if self.plot_kws["arrowprops"]["connectionstyle"] is None:
			arm_height = arrow_height * self.frac
			# rad = self.rad  # arm_height / 10
			if axis == 1 and self.orientation == "up":
				angleA, angleB = (self.plot_kws["rotation"] - 180, 90)
			elif axis == 1 and self.orientation == "down":
				angleA, angleB = (180 + self.plot_kws["rotation"], -90)
			elif axis == 0 and self.orientation == "left":
				angleA, angleB = (self.plot_kws["rotation"], -180)
			else:
				angleA, angleB = (self.plot_kws["rotation"] - 180, 0)
			connectionstyle = f"arc,angleA={angleA},angleB={angleB},armA={arm_height},armB={arm_height},rad={self.rad}"
			self.plot_kws["arrowprops"]["connectionstyle"] = connectionstyle
		# print("connectionstyle: ",self.plot_kws["arrowprops"]["connectionstyle"])
		# import pdb;
		# pdb.set_trace()
		for t, x_0, y_0, x_1, y_1 in zip(labels, x, y, x1, y1):
			if pd.isna(t):
				continue
			color = self.color_dict[t]
			if self.adjust_color:
				lum = _calculate_luminance(color)
				if lum > self.luminance:
					color = "black"
			self.plot_kws["arrowprops"]["color"] = color
			annotated_text = ax.annotate(
				text=t,
				xy=(x_0, y_0), #The point (x, y) to annotate
				xytext=(x_1, y_1), #The position (x, y) to place the text at. The coordinate system is determined by textcoords.
				xycoords=xycoords,
				textcoords=text_xycoords,
				color=color,
				**self.plot_kws
			)  # unit for shrinkA is point (1 point = 1/72 inches)
			self.annotated_texts.append(annotated_text)
		ax.set_axis_off()
		self.ax = ax
		self.fig = self.ax.figure
		return self.ax

	def get_ticklabel_width(self):
		hs = [text.get_window_extent().width for text in self.annotated_texts]
		if len(hs) == 0:
			return 0
		else:
			return max(hs)


# =============================================================================
class anno_boxplotCT(AnnotationBaseCellTools):
	"""
		annotate boxplots, all arguments are included in AnnotationBase,
		plot_kws for anno_boxplotCT include showfliers, edgecolor, grid, medianlinecolor
		width,zorder and other arguments passed to plt.boxplot.

	Parameters
	----------
	"""

	def _height(self, height):
		return 10 if height is None else height

	def _set_default_plot_kws(self, plot_kws):
		self.plot_kws = plot_kws if plot_kws is not None else {}
		self.plot_kws.setdefault("showfliers", False)
		self.plot_kws.setdefault("edgecolor", "black")
		self.plot_kws.setdefault("medianlinecolor", "black")
		self.plot_kws.setdefault("grid", False)
		self.plot_kws.setdefault("zorder", 10)
		self.plot_kws.setdefault("widths", 0.5)

	def _check_cmap(self, cmap):
		if cmap == "auto":
			self.cmap = "jet"
		elif type(cmap) == str:
			self.cmap = cmap
		else:
			print("WARNING: cmap for boxplot is not a string!")
			self.cmap = cmap

	def _calculate_colors(self):  # add self.color_dict (each col is a dict)
		self.colors = None

	def _check_colors(self, colors):
		if type(colors) == str:
			self.colors = colors
		else:
			raise TypeError(
				"Boxplot only support one string as colors now, if more colors are wanted, cmap can be specified."
			)

	def _calculate_cmap(self):
		self.set_legend(False)
		self.cmap = None

	def plot(self, ax=None, axis=1):  # add self.gs,self.fig,self.ax,self.axes
		fig = ax.figure
		if self.colors is None:  # calculate colors based on cmap
			colors = [
				get_colormap(self.cmap)(self.plot_data.loc[sampleID].mean())
				for sampleID in self.plot_data.index.values
			]
		else:
			colors = [self.colors] * self.plot_data.shape[0]  # self.colors is a string
		# print(self.plot_kws)
		plot_kws = self.plot_kws.copy()
		edgecolor = plot_kws.pop("edgecolor")
		mlinecolor = plot_kws.pop("medianlinecolor")
		grid = plot_kws.pop("grid")
		# bp = ax.boxplot(self.plot_data.T.values, patch_artist=True,**self.plot_kws)
		if axis == 1:
			vert = True
			ax.set_xticks(ticks=np.arange(0.5, self.nrows, 1))
		else:
			vert = False
			ax.set_yticks(ticks=np.arange(0.5, self.nrows, 1))
		# bp = self.plot_data.T.boxplot(ax=ax, patch_artist=True,vert=vert,return_type='dict',**self.plot_kws)
		bp = ax.boxplot(
			x=self.plot_data.T.values, #shape=(n_fea,n_samples)
			positions=np.arange(0.5, self.nrows, 1),
			patch_artist=True,
			vert=vert, #If True, draws vertical boxes. If False, draw horizontal boxes
			**plot_kws
		)
		if grid:
			ax.grid(linestyle="--", zorder=-10)
		for box, color in zip(bp["boxes"], colors):
			box.set_facecolor(color)
			box.set_edgecolor(edgecolor)
		for median_line in bp["medians"]:
			median_line.set_color(mlinecolor)
		if axis == 1:
			ax.set_xlim(0, self.nrows)
			ax.set_ylim(*self.ylim)
			ax.tick_params(
				axis="both",
				which="both",
				top=False,
				bottom=False,
				labeltop=False,
				labelbottom=False,
			)
		else:
			ax.set_ylim(0, self.nrows)
			ax.set_xlim(*self.ylim)
			ax.tick_params(
				axis="both",
				which="both",
				left=False,
				right=False,
				labelleft=False,
				labelright=False,
			)
			# if self.orientation=='left':
			# 	ax.invert_xaxis()
		self.fig = fig
		self.ax = ax
		return self.ax


# =============================================================================
class anno_barplotCT(anno_boxplotCT):
	"""
	Annotate barplot, all arguments are included in AnnotationBaseCellTools,
		plot_kws for anno_boxplotCT include edgecolor, grid,align,zorder,
		and other arguments passed to plt.barplot.
	"""

	def _set_default_plot_kws(self, plot_kws):
		self.plot_kws = plot_kws if plot_kws is not None else {}
		self.plot_kws.setdefault("edgecolor", "black")
		self.plot_kws.setdefault("grid", False)
		self.plot_kws.setdefault("zorder", 10)
		# self.plot_kws.setdefault('width', 0.7)
		self.plot_kws.setdefault("align", "center")

	def _check_cmap(self, cmap):
		if cmap == "auto":
			if self.ncols == 1:
				self.cmap = "jet"
			else:
				self.cmap = "Set1"
		# print(cmap,self.cmap)
		else:
			self.cmap = cmap
		if self.ncols >= 2 and get_colormap(self.cmap).N >= 256:
			raise TypeError(
				"cmap for stacked barplot should not be continuous, you should try: Set1, Dark2 and so on."
			)

	def _calculate_colors(self):  # add self.color_dict (each col is a dict)
		col_list = self.df.columns.tolist()
		self.color_dict = {}
		if self.ncols >= 2:  # more than two columns, colored by columns names
			self.colors = [
				get_colormap(self.cmap)(col_list.index(v)) for v in self.df.columns
			] #list
			for v, color in zip(col_list, self.colors):
				self.color_dict[v] = color
		else:  # only one column, colored by cols[0] values (float)
			# vmax, vmin = np.nanmax(self.df[col_list[0]].values), np.nanmin(self.df[col_list[0]].values)
			# delta = vmax - vmin
			# values = self.df[col_list[0]].fillna(np.nan).unique()
			self.cmap, normalize = define_cmap(
				self.df[col_list[0]].fillna(np.nan).values,
				vmin=None,
				vmax=None,
				cmap=self.cmap,
				center=None,
				robust=False,
				na_col="white",
			)
			# self.colors = {v: matplotlib.colors.rgb2hex(get_colormap(self.cmap)((v - vmin) / delta)) for v in values}
			self.colors = lambda v: matplotlib.colors.rgb2hex(
				self.cmap(normalize(v))
			)  # a function
			self.color_dict = None

	def _check_colors(self, colors):
		self.colors = colors
		col_list = self.df.columns.tolist()
		if not isinstance(colors, (list, str, dict, tuple)):
			raise TypeError("colors must be list of string,list, tuple or dict")
		if type(colors) == str:
			color_dict = {label: colors for label in col_list}
		elif isinstance(colors,(list,tuple)):
			assert len(colors) == self.ncols, "length of colors should match length of df.columns"
			color_dict = {
				label: color
				for label, color in zip(col_list, colors)
			}
		else:
			assert isinstance(colors, dict)
			color_dict=colors.copy()
			keys=list(color_dict.keys())
			for key in keys:
				if key not in col_list:
					del color_dict[key]
		self.color_dict = color_dict

	def _calculate_cmap(self):
		self.cmap = None
		# self.set_legend(False)

	def _type_specific_params(self):
		if self.ncols > 1:
			self.stacked = True
		else:
			self.stacked = False
		if self.ylim is None:
			Max = np.nanmax(self.df.sum(axis=1).values) if self.stacked else np.nanmax(self.df.values)
			Min = np.nanmin(self.df.sum(axis=1).values) if self.stacked else np.nanmin(self.df.values)
			gap = Max - Min
			self.ylim = [Min - 0.05 * gap, Max + 0.05 * gap]

	def plot(self, ax=None, axis=1):  # add self.gs,self.fig,self.ax,self.axes
		if ax is None:
			ax = plt.gca()
		fig = ax.figure
		plot_kws = self.plot_kws.copy()
		grid = plot_kws.pop("grid", False)
		if grid:
			ax.grid(linestyle="--", zorder=-10)
		if self.ncols ==1 and not self.cmap is None: # only one columns, use cmap
			colors = [[self.colors(v) for v in self.plot_data.iloc[:, 0].values]]
		else: # self.ncols ==1: #cmap is None,use color_dict
			assert not self.color_dict is None
			colors=[self.color_dict[col] for col in self.plot_data.columns]

		base_coordinates = [0] * self.plot_data.shape[0]
		for col, color in zip(self.plot_data.columns, colors):
			if axis == 1: #columns annotations
				ax.set_xticks(ticks=np.arange(0.5, self.nrows, 1))
				ax.bar(
					x=np.arange(0.5, self.nrows, 1),
					height=self.plot_data[col].values,
					bottom=base_coordinates,
					color=color,
					**plot_kws
				)
				ax.set_xlim(0, self.nrows)
				ax.set_ylim(*self.ylim)
			else:
				ax.set_yticks(ticks=np.arange(0.5, self.nrows, 1))
				ax.barh(
					y=np.arange(0.5, self.nrows, 1),
					width=self.plot_data[col].values,
					left=base_coordinates,
					color=color,
					**plot_kws
				)
				ax.set_ylim(0, self.nrows)
				ax.set_xlim(*self.ylim)
			base_coordinates = self.plot_data[col].values + base_coordinates
		# for patch in ax.patches:
		#     patch.set_edgecolor(edgecolor)
		if axis == 0:
			ax.tick_params(
				axis="both",
				which="both",
				left=False,
				right=False,
				labelleft=False,
				labelright=False,
			)
			# if self.orientation == 'left':
			# 	ax.invert_xaxis()
		else:
			ax.tick_params(
				axis="both",
				which="both",
				top=False,
				bottom=False,
				labeltop=False,
				labelbottom=False,
			)
		self.fig = fig
		self.ax = ax
		return self.ax


# =============================================================================
class anno_scatterplotCT(anno_barplotCT):
	"""
	Annotate scatterplot, all arguments are included in AnnotationBaseCellTools,
		plot_kws for anno_scatterplotCT include linewidths, grid, edgecolors
		and other arguments passed to plt.scatter.
	"""

	def _check_df(self, df):
		if isinstance(df, pd.Series):
			df = df.to_frame(name=df.name)
		if isinstance(df, pd.DataFrame) and df.shape[1] != 1:
			raise ValueError("df must have only 1 column for scatterplot")
		elif isinstance(df, pd.DataFrame):
			self.df = df
		else:
			raise TypeError("df must be a pandas DataFrame or Series.")

	def _set_default_plot_kws(self, plot_kws):
		self.plot_kws = plot_kws if plot_kws is not None else {}
		self.plot_kws.setdefault("grid", False)
		self.plot_kws.setdefault("zorder", 10)
		self.plot_kws.setdefault("linewidths", 0)
		self.plot_kws.setdefault("edgecolors", "black")

	def _check_cmap(self, cmap):
		self.cmap = "jet"
		if cmap == "auto":
			pass
		elif type(cmap) == str:
			self.cmap = cmap
		else:
			print("WARNING: cmap for scatterplot is not a string!")
			self.cmap = cmap

	def _calculate_colors(self):  # add self.color_dict (each col is a dict)
		self.colors = None

	def _check_colors(self, colors):
		if not isinstance(colors, str):
			raise TypeError(
				"colors must be string for scatterplot, if more colors are neded, please try cmap!"
			)
		self.colors = colors

	def _calculate_cmap(self):
		self.cmap = None
		self.set_legend(False)

	def _type_specific_params(self):
		Max = np.nanmax(self.df.values)
		Min = np.nanmin(self.df.values)
		self.gap = Max - Min
		if self.ylim is None:
			self.ylim = [Min - 0.05 * self.gap, Max + 0.05 * self.gap]

	def plot(self, ax=None, axis=1):  # add self.gs,self.fig,self.ax,self.axes
		if ax is None:
			ax = plt.gca()
		fig = ax.figure
		plot_kws = self.plot_kws.copy()
		grid = plot_kws.pop("grid", False)
		if grid:
			ax.grid(linestyle="--", zorder=-10)
		values = self.plot_data.iloc[:, 0].values
		if self.colors is None:
			colors = self.plot_data.iloc[:, 0].dropna().values
		else:  # self.colors is a string
			colors = [self.colors] * self.plot_data.dropna().shape[0]
		if axis == 1:
			spu = (
				ax.get_window_extent().height * 72 / self.gap / fig.dpi
			)  # size per unit
		else:
			spu = (
				ax.get_window_extent().width * 72 / self.gap / fig.dpi
			)  # size per unit
		value_min=np.nanmin(values)
		self.s = [(v - value_min + self.gap * 0.1) * spu for v in values if not pd.isna(v)]  # fontsize
		if axis == 1:
			ax.set_xticks(ticks=np.arange(0.5, self.nrows, 1))
			# x = np.arange(0.5, self.nrows, 1)
			# y = values
			y = []
			x=[]
			for x1,y1 in zip(np.arange(0.5, self.nrows, 1),values):
				if pd.isna(y1):
					continue
				x.append(x1)
				y.append(y1)
		else:
			ax.set_yticks(ticks=np.arange(0.5, self.nrows, 1))
			# y = np.arange(0.5, self.nrows, 1)
			# x = values
			y = []
			x = []
			for x1, y1 in zip(np.arange(0.5, self.nrows, 1), values):
				if pd.isna(y1):
					continue
				y.append(x1)
				x.append(y1)
		c = self.plot_kws.get("c", colors)
		s = self.plot_kws.get("s", self.s)
		scatter_ax = ax.scatter(x=x, y=y, c=c, s=s, cmap=self.cmap, **plot_kws)
		if axis == 0: #row annotations
			ax.set_ylim(0, self.nrows)
			ax.set_xlim(*self.ylim)
			ax.tick_params(
				axis="both",
				which="both",
				left=False,
				right=False,
				labelleft=False,
				labelright=False,
			)
			# if self.orientation == 'left':
			# 	ax.invert_xaxis()
		else: #columns annotations
			ax.set_xlim(0, self.nrows)
			ax.set_ylim(*self.ylim)
			ax.tick_params(
				axis="both",
				which="both",
				top=False,
				bottom=False,
				labeltop=False,
				labelbottom=False,
			)
		self.fig = fig
		self.ax = ax
		return self.ax


class anno_imgCT(AnnotationBaseCellTools):
	"""
	Annotate images.

	Parameters
	----------
	border_width : int
		width of border lines between images (0-256?). Ignored when merge is True.
	border_color : int
		color of border lines. black:0, white:255. Ignored when merge is True.
	merge: bool
		whether to merge the same clusters into one and show image only once.
	merge_width: float
        width of image when merge is True
		whether to merge the same clusters into one and show image only once.
	rotate: int
		Rotate the input images
	mode: str
		all possible mode to convert, between "L", "RGB" and "CMYK", 'RGBA', default is RGBA
	"""
	def __init__(
		self,
		df=None,
		cmap=None,
		colors=None,
		border_width=1,
		border_color=255,
        merge=False,
        merge_width=1,
		rotate=None,
		mode='RGBA',
		**plot_kws
	):
		self.border_width = border_width
		self.border_color = border_color
		self.merge = merge
		self.merge_width = merge_width
		self.rotate=rotate
		self.mode=mode
		self.plot_kws = plot_kws
		super().__init__(
			df=df,
			cmap=cmap,
			colors=colors,
			**plot_kws
		)

	def _height(self, height):
		return 10 if height is None else height

	def _set_default_plot_kws(self, plot_kws):
		self.plot_kws = plot_kws if plot_kws is not None else {}

	def _calculate_colors(self):  # add self.color_dict (each col is a dict)
		self.colors = None

	def _check_cmap(self, cmap):
		self.cmap = None 

	def read_img(self,img_path=None,shape=None):
		#import matplotlib.image as mpimg  # mpimg.imread
		import PIL
		import requests
		from io import BytesIO
		if pd.isna(img_path):
			if shape is None:
				return None
			else:
				new_shape=tuple([shape[1],shape[0]]+list(shape[2:]))
				# print(shape, new_shape,type(shape), 'here')
				return np.full(new_shape, self.border_color)
		if os.path.exists(img_path):
			image = PIL.Image.open(img_path) #mpimg.imread(img_path)
		else: #remote file
			response = requests.get(img_path)
			# Open the image from bytes
			image = PIL.Image.open(BytesIO(response.content))
		if image.mode != self.mode:
			image = image.convert(self.mode)
		if not shape is None:
			image=image.resize(shape[:2]) #width, height
		if not self.rotate is None:
			image=image.rotate(self.rotate)
		# Convert the image to an array if needed
		image = np.array(image)
		return image

	def _add_border(self, img, width=1, color=0, axis=1):
		w = width
		if axis == 1:
			pad_width = ((0, 0), (w, w), (0, 0))
		else:
			pad_width = ((w, w), (0, 0), (0, 0))

		bordered_img = np.pad(img, pad_width=pad_width, 
						mode='constant', constant_values=color)
		return bordered_img

	def _type_specific_params(self):
		pass
	def plot(self, ax=None, axis=1):
		if ax is None:
			ax = plt.gca()
		if axis==1:
			imgfiles = list(self.plot_data.iloc[:,0]) #[::-1] #fix bug for the inverted yaxis
		else:
			imgfiles = list(self.plot_data.iloc[:, 0])[::-1]
		imgs = [self.read_img(img_path=imgfile) for imgfile in imgfiles]
		shapes = [img.shape for img in imgs if not img is None]  # (height,width, channel)
		if len(set(shapes)) > 1 or len(shapes) != len(imgs):  # None is in imgs
			# resize the images to make sure all images have the same height and wdith
			if len(shapes)>1:
				shape = np.min(np.vstack(shapes), axis=0)  # height,width, channel; height, width,*channel
			else:
				shape=shapes[0]
			new_shape = tuple([shape[1], shape[0]] + list(shape[2:]))
			imgs = [self.read_img(img_path=imgfile, shape=new_shape) for imgfile in imgfiles]
			shapes = [img.shape for img in imgs]
		# for img in imgs:
		# 	print(img.shape)
		img_shape = shapes[0]
		img_h = img_shape[0]  # shape: height,width, channel
		img_w = img_shape[1]
		if self.merge:
			origin = 'upper'
			assert self.plot_data.iloc[:,0].dropna().nunique()==1, "Not all file names in the list are identical"
			imgs = imgs[0]
			if axis==1: #columns annotation
				extent = [self.nrows/2-self.merge_width/2, self.nrows/2+self.merge_width/2, 0, img_h]
				# floats (left, right, bottom, top), optional
				# The bounding box in data coordinates that the image will fill
			else:
				extent = [0, img_w, self.nrows/2-self.merge_width/2, self.nrows/2+self.merge_width/2]
		else:
			if axis==1:
				imgs = np.hstack(tuple([self._add_border(img,width=self.border_width,
														 color=self.border_color, axis=axis) \
                            for img in imgs]))
				extent = [0, self.nrows, 0, img_h]
				origin='upper'
			else: #axis=0
				# ax.invert_yaxis()  # y is shared, invert has no effect (only useful when anno_imgCT on the most right side, main axes of sharey)
				# in default, if orientation=='right', x direction is: left -> right, orient='left', right -> left
				origin = 'lower'
				if self.orientation=='left':
					# ax.invert_xaxis() # no effect
					ax.set_xlim(img_w,0)
				# else:
				# 	# ax.set_ylim(self.nrows,0)
				imgs = np.vstack(tuple([self._add_border(img,
                                                width=self.border_width, color=self.border_color, axis=axis) \
                            for img in imgs[::-1]])) #bottom -> up? to invert: up -> bottom
				extent = [0,img_w, 0, self.nrows]
		self.plot_kws.setdefault('origin',origin)
		ax.imshow(imgs, aspect='auto', extent=extent, cmap=self.cmap, **self.plot_kws)
		ax.tick_params(axis='both',which='both',labelbottom=False, labelleft=False,
								labelright=False, labeltop=False,
					   			bottom=False, left=False,
								right=False, top=False)
		# ax.set_axis_off()
		self.ax = ax
		self.fig = self.ax.figure
		return self.ax

class anno_lineplotCT(anno_barplotCT):
	"""
	Annotate lineplot, all arguments are included in AnnotationBaseCellTools,
		parameter grid control whether to show grid (default is True),
		other arguments passed to plt.plot, including linewidth, marker and so on.
	"""

	def _check_df(self, df):
		if isinstance(df, pd.Series):
			self.df = df.to_frame(name=df.name)
		elif isinstance(df, pd.DataFrame):
			self.df = df
		else:
			raise TypeError("df must be a pandas DataFrame or Series.")

	def _set_default_plot_kws(self, plot_kws):
		self.plot_kws = plot_kws if plot_kws is not None else {}
		self.plot_kws.setdefault("grid", False)
		self.plot_kws.setdefault("zorder", 10)
		self.plot_kws.setdefault("linewidth", 1)

	def _check_cmap(self, cmap):
		self.cmap = "Set1"
		if cmap == "auto":
			pass
		elif type(cmap) == str:
			self.cmap = cmap
		else:
			print("WARNING: cmap for scatterplot is not a string!")
			self.cmap = cmap

	def _calculate_colors(self):  # add self.color_dict (each col is a dict)
		col_list = self.df.columns.tolist()
		self.color_dict = {}
		self.colors = [get_colormap(self.cmap)(col_list.index(v)) for v in col_list]
		for v, color in zip(col_list, self.colors):
			self.color_dict[v] = color

	def plot(self, ax=None, axis=1):  # add self.gs,self.fig,self.ax,self.axes
		if ax is None:
			ax = plt.gca()
		fig = ax.figure
		plot_kws = self.plot_kws.copy()
		grid = plot_kws.pop("grid", False)
		if grid:
			ax.grid(linestyle="--", zorder=-10)
		for col in self.color_dict:
			color=self.color_dict[col]
			if axis == 1:
				ax.set_xticks(ticks=np.arange(0.5, self.nrows, 1))
				ax.plot(
					np.arange(0.5, self.nrows, 1),
					self.plot_data[col].values,
					color=color,
					**plot_kws
				)
				ax.set_xlim(0, self.nrows)
				ax.set_ylim(*self.ylim)
			else:
				ax.set_yticks(ticks=np.arange(0.5, self.nrows, 1))
				ax.plot(
					self.plot_data[col].values,
                    np.arange(0.5, self.nrows, 1),
					color=color,
					**plot_kws
				)
				ax.set_ylim(0, self.nrows)
				ax.set_xlim(*self.ylim)
		if axis == 0:
			ax.tick_params(
				axis="both",
				which="both",
				left=False,
				right=False,
				labelleft=False,
				labelright=False,
			)
			# if self.orientation == 'left':
			# 	ax.invert_xaxis()
		else:
			ax.tick_params(
				axis="both",
				which="both",
				top=False,
				bottom=False,
				labeltop=False,
				labelbottom=False,
			)
			# if self.orientation=='down':
			# 	ax.invert_yaxis()
		self.fig = fig
		self.ax = ax
		return self.ax
# =============================================================================
class anno_dendrogramCT(AnnotationBaseCellTools):
	def __init__(
		self,
		df=None,
		cmap="auto",
		colors=None,
		add_text=False,
		majority=True,
		text_kws=None,
		height=None,
		dendrogram_kws=None,
		**plot_kws
	):
		"""
		Annotate and plot dendrogram. Please Note, when use anno_dendrogramCT
		together with heatmap, there may be some issue.

		Parameters
		----------
		df : DataFrame
			Calculate linkage for rows, to calculate linkage for columns, please
			provide df.T.
		cmap :
		colors :
		add_text :
		majority :
		text_kws :
		height :
		dendrogram_kws :
		plot_kws :
		"""
		self.add_text = add_text
		self.majority = majority
		self.text_kws = text_kws if not text_kws is None else {}
		self.plot_kws = plot_kws
		self.dendrogram_kws={} if dendrogram_kws is None else dendrogram_kws
		super().__init__(
			df=df,
			cmap=cmap,
			colors=colors,
			height=height,
			legend=False,
			**plot_kws
		)
		self.dend = DendrogramPlotterCellTools(
			self.plot_data,
			**self.dendrogram_kws
		)
		self.row_order = [
			self.dend.dendrogram["ivl"]
		]

	def _height(self, height):
		return 10 if height is None else height

	def _set_default_plot_kws(self, plot_kws):
		self.plot_kws = {} if plot_kws is None else plot_kws
		self.plot_kws.setdefault("invert", False)
		# self.dendrogram_kws.setdefault("label", False)

	def _check_cmap(self, cmap):
		if cmap == "auto":
			if self.df.shape[0] <= 10:
				self.cmap = "Set1"
			elif self.df.shape[0] <= 20:
				self.cmap = "tab20"
			else:
				self.cmap = "random50"
		elif type(cmap) == str:
			self.cmap = cmap
		else:
			print("WARNING: cmap is not a string!")
			self.cmap = cmap

	def _check_colors(self, colors):
		if isinstance(colors,str):
			colors=[colors]*self.nrows
		assert isinstance(colors,list)
		assert len(colors)==self.nrows
		self.colors=colors

	def _calculate_colors(self):  # add self.color_dict (each col is a dict)
		if self.cmap is None:
			self.colors = ['black'] * self.nrows
		else:
			self.colors = [
				get_colormap(self.cmap)(i) for i in range(self.nrows)
			]

	def _calculate_cmap(self):
		self.cmap = None
		pass

	def _type_specific_params(self):
		pass

	def plot(self, ax=None, axis=1):
		self.plot_kws.setdefault("tree_kws", dict(colors=self.colors))
		# inint the DendrogramPlotterCellTools class object
		ax.set_axis_off()
		self.dend.plot(ax=ax,axis=axis,**self.plot_kws)
		self.ax = ax
		self.fig = self.ax.figure
		return self.ax

# =============================================================================
class HeatmapAnnotationCellTools:
	"""
	Generate and plot heatmap annotations.

	Parameters
	----------
	self : Class
		HeatmapAnnotationCellTools
	df :  dataframe
		a pandas dataframe, each column will be converted to one anno_simpleCT class.
	axis : int
		1 for columns annotation, 0 for rows annotations.
	cmap : str
		colormap, such as Set1, Dark2, bwr, Reds, jet, hsv, rainbow and so on. Please see
		https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html for more information, or run
		matplotlib.pyplot.colormaps() to see all availabel cmap.
		default cmap is 'auto', it would be determined based on the dtype for each columns in df.
		If df is None, then there is no need to specify cmap, cmap and colors will only be used when
		df is provided.
		If cmap is a string, then all columns in the df would have the same cmap, cmap can also be
		a dict, keys are the column names from df, values should be cmap (matplotlib.pyplot.colormaps()).
	colors : dict
		a dict, keys are the column names of df, values are list, dict or string passed to
		AnnotationBaseCellTools.__subclasses__(), including anno_simpleCT, anno_boxplotCT,anno_labelCT and anno_scatter.
		colors must have the same length as the df.columns, if colors is not provided (default), else,
		colors would be calculated based on the given cmap.
		If colors is given, then the cmap would be invalid.
	label_side : str
		top or bottom when axis=1 (columns annotation), left or right when axis=0 (rows annotations).
	label_kws : dict
		kws passed to the labels of the annotation labels (would be df.columns if df is given).
		such as alpha, color, fontsize, fontstyle, ha (horizontalalignment),
		va (verticalalignment), rotation, rotation_mode, visible, rasterized and so on.
		For more information, see plt.gca().yaxis.label.properties() or ax.yaxis.label.properties()
	ticklabels_kws : dict
		label_kws is for the label of annotation, ticklabels_kws is for the label (text) in anno_labelCT,
		such as axis, which, direction, length, width,
		color, pad, labelsize, labelcolor, colors, zorder, bottom, top, left, right, labelbottom, labeltop,
		labelleft, labelright, labelrotation, grid_color, grid_linestyle and so on.
		For more information,see ?matplotlib.axes.Axes.tick_params
	plot_kws : dict
		kws passed to annotation functions, such as anno_simpleCT, anno_labelCT et.al.
	plot : bool
		whether to plot, when the annotation are included in clustermap, plot would be
		set to False automotially.
	legend : bool
		True or False, or dict (when df is no None), when legend is dict, keys are the
		columns of df.
	legend_side : str
		right or left
	legend_gap : float
		the vertical gap between two legends, default is 2 [mm]
	legend_width: float
		width of the legend, default is 4.5[mm]
	legend_hpad: float
		Horizonal space between heatmap and legend, default is 2 [mm].
	legend_vpad: float
		Vertical space between top of ax and legend, default is 2 [mm].
	orientation: str
		up or down, when axis=1
		left or right, when axis=0;
		When anno_labelCT shows up in annotation, the orientation would be automatically be assigned according
		to the position of anno_labelCT.
	wgap: float or int
		optional,  the space used to calculate wspace, default is [0.1] (mm),
		control the vertical gap between two annotations.
	hgap: float or int
		optional,  the space used to calculate hspace, default is [0.1] (mm),
		control the horizontal gap between two annotations.
	plot_legend : bool
		whether to plot legends.
	legend_order: str, bool or list
		control the order of legends, default is 'auto', sorted by length of legend.
		could also be True/False or a list (or tuple), if a list / tuple is provided,
		values should be the label (title) of each legend.
	args : name-value pair
		key is the annotation label (name), values can be a pandas dataframe,
		series, or annotation such as
		anno_simpleCT, anno_boxplotCT, anno_scatterCT, anno_labelCT, or anno_barplotCT.

	Returns
	-------
	Class HeatmapAnnotationCellTools.

	"""
	def __init__(
		self,
		df=None,
		axis=1,
		cmap="auto",
		colors=None,
		label_side=None,
		label_kws=None,
		ticklabels_kws=None,
		plot_kws=None,
		plot=False,
		legend=True,
		legend_side="right",
		legend_gap=5,
		legend_width=4.5,
		legend_hpad=2,
		legend_vpad=5,
		orientation=None,
		wgap=0.1,
		hgap=0.1,
		plot_legend=True,
		legend_order='auto',
		rasterized=False,
		verbose=1,
		**args
	):
		if df is None and len(args) == 0:
			raise ValueError("Please specify either df or other args")
		if not df is None and len(args) > 0:
			raise ValueError("df and Name-value pairs can only be given one, not both.")
		if not df is None:
			self._check_df(df)
		else:
			self.df = None
		self.axis = axis
		self.verbose = verbose
		self.label_side = label_side
		self.plot_kws = plot_kws if not plot_kws is None else {}
		self.args = args
		self._check_legend(legend)
		self.legend_side = legend_side
		self.legend_gap = legend_gap
		self.wgap = wgap
		self.hgap = hgap
		self.legend_width = legend_width
		self.legend_hpad = legend_hpad
		self.legend_vpad = legend_vpad
		self.plot_legend = plot_legend
		self.legend_order=legend_order
		self.rasterized = rasterized
		self.orientation = orientation
		self.plot = plot
		if colors is None:
			self._check_cmap(cmap)
			self.colors = None
		else:
			self._check_colors(colors)
		self._process_data()
		self.heights = [ann.height for ann in self.annotations]
		self.nrows = [ann.nrows for ann in self.annotations]
		self.label_kws, self.ticklabels_kws = label_kws, ticklabels_kws
		if self.plot:
			self.plot_annotations()

	def _check_df(self, df):
		if type(df) == list or isinstance(df, np.ndarray):
			df = pd.Series(df).to_frame(name="df")
		elif isinstance(df, pd.Series):
			name = df.name if not df.name is None else "df"
			df = df.to_frame(name=name)
		if not isinstance(df, pd.DataFrame):
			raise TypeError(
				"data type of df could not be recognized, should be a dataframe"
			)
		self.df = df

	def _check_legend(self, legend):
		if type(legend) == bool:
			if not self.df is None:
				self.legend = {col: legend for col in self.df.columns}
			if len(self.args) > 0:
				# self.legend = collections.defaultdict(lambda: legend)
				self.legend = {arg: legend for arg in self.args}
		elif type(legend) == dict:
			self.legend = legend
			for arg in self.args:
				if arg not in self.legend:
					self.legend[arg] = False
		else:
			raise TypeError("Unknow data type for legend!")

	def _check_cmap(self, cmap):
		if self.df is None:
			return
		self.cmap = {}
		if cmap == "auto":
			for col in self.df.columns:
				if self.df.dtypes[col] in [object,'category']:
					if self.df[col].nunique() <= 10:
						self.cmap[col] = "Set1"
					elif self.df[col].nunique() <= 20:
						self.cmap[col] = "tab20"
					else:
						self.cmap[col] = "random50"
				elif self.df.dtypes[col] == float or self.df.dtypes[col] == int:
					self.cmap[col] = "jet"
				else:
					raise TypeError(
						"Can not assign cmap for column %s, please specify cmap" % col
					)
		elif type(cmap) == str:
			self.cmap = {col: cmap for col in self.df.columns}
		elif type(cmap) == list:
			if len(cmap) == 1:
				cmap = cmap * len(self.df.shape[1])
			if len(cmap) != self.df.shape[1]:
				raise ValueError(
					"kind must have the same lengt with the number of columns with df"
				)
			self.cmap = {col: c for col, c in zip(self.df.columns, cmap)}
		elif type(cmap) == dict:
			if len(cmap) != self.df.shape[1]:
				raise ValueError(
					"kind must have the same length with number of columns with df"
				)
			self.cmap = cmap
		else:
			print("WARNING: unknown datatype for cmap!")
			self.cmap = cmap

	def _check_colors(self, colors):
		if self.df is None:
			return
		self.colors = {}
		if not isinstance(colors, dict):
			raise TypeError("colors must be a dict!")
		if len(colors) != self.df.shape[1]:
			raise ValueError("colors must have the same length as the df.columns!")
		self.colors = colors

	def _process_data(self):  # add self.annotations,self.names,self.labels
		self.annotations = []
		self.plot_kws["rasterized"] = self.rasterized
		if not self.df is None:
			for col in self.df.columns:
				plot_kws = self.plot_kws.copy()
				if self.colors is None:
					plot_kws.setdefault("cmap", self.cmap[col])  #
				else:
					plot_kws.setdefault("colors", self.colors[col])
				anno1 = anno_simpleCT(
					self.df[col], legend=self.legend.get(col, False), **plot_kws
				)
				anno1.set_label(col)
				anno1.set_orientation(self.orientation)
				self.annotations.append(anno1)
		elif len(self.args) > 0:
			# print(self.args)
			self.labels = []
			for arg in self.args:
				# print(arg)
				ann = self.args[arg] # Series, anno_* or DataFrame
				if type(ann) == list or isinstance(ann, np.ndarray):
					ann = pd.Series(ann).to_frame(name=arg)
				elif isinstance(ann, pd.Series):
					ann = ann.to_frame(name=arg)
				if isinstance(ann, pd.DataFrame):
					if ann.shape[1] > 1:
						for col in ann.columns:
							anno1 = anno_simpleCT(
								ann[col],
								legend=self.legend.get(col, False),
								**self.plot_kws
							)
							anno1.set_label(col)
							self.annotations.append(anno1)
					else:
						anno1 = anno_simpleCT(ann, **self.plot_kws)
						anno1.set_label(arg)
						anno1.set_legend(self.legend.get(arg, False))
						self.annotations.append(anno1)
				if hasattr(ann, "set_label") and AnnotationBaseCellTools.__subclasscheck__(
					type(ann)
				):
					self.annotations.append(ann)
					ann.set_label(arg)
					ann.set_legend(self.legend.get(arg, False))
					if type(ann) == anno_labelCT and self.orientation is None:
						if self.axis == 1 and len(self.labels) == 0:
							self.orientation = "up"
						elif self.axis == 1:
							self.orientation = "down"
						elif self.axis == 0 and len(self.labels) == 0:
							self.orientation = "left"
						elif self.axis == 0:
							self.orientation = "right"
					ann.set_orientation(self.orientation)
				self.labels.append(arg)

	def _set_orentation(self, orientation):
		if self.orientation is None:
			self.orientation = orientation

	def _set_label_kws(self, label_kws, ticklabels_kws):
		if self.label_side in ["left", "right"] and self.axis != 1:
			raise ValueError(
				"For row annotation, label_side must be top or bottom!"
			)
		if self.label_side in ["top", "bottom"] and self.axis != 0:
			raise ValueError("For columns annotation, label_side must be left or right!")
		if self.orientation is None:
			if self.axis == 1:
				self.orientation = "up"
			else:  # horizonal
				self.orientation = "left"
		self.label_kws = {} if label_kws is None else label_kws
		self.ticklabels_kws = {} if ticklabels_kws is None else ticklabels_kws
		self.label_kws.setdefault("rotation_mode", "anchor")
		if self.label_side is None:
			self.label_side = (
				"right" if self.axis == 1 else "top"
			)  # columns annotation, default ylabel is on the right
		ha, va = "left", "center"
		if self.orientation == "left":
			rotation, labelrotation = 90, 90
			ha = "right" if self.label_side == "bottom" else "left"
		elif self.orientation == "right":
			ha = "right" if self.label_side == "top" else "left"
			rotation, labelrotation = -90, -90
		else:  # self.orientation == 'up':
			rotation, labelrotation = 0, 0
			ha = "left" if self.label_side == "right" else "right"
		self.label_kws.setdefault("rotation", rotation)
		self.ticklabels_kws.setdefault("labelrotation", labelrotation)
		self.label_kws.setdefault("horizontalalignment", ha)
		self.label_kws.setdefault("verticalalignment", va)

		map_dict = {"right": "left", "left": "right", "top": "bottom", "bottom": "top"}
		self.ticklabels_side = map_dict[self.label_side]
		# label_kws: alpha,color,fontfamily,fontname,fontproperties,fontsize,fontstyle,fontweight,label,rasterized,
		# rotation,rotation_mode(default,anchor),visible, zorder,verticalalignment,horizontalalignment

	def set_axes_kws(self):
		if self.axis == 1 and self.label_side == "left":
			self.ax.yaxis.tick_right()
			for i in range(self.axes.shape[0]):
				self.axes[i, 0].yaxis.set_visible(True)
				self.axes[i, 0].yaxis.label.set_visible(True)
				self.axes[i, 0].tick_params(
					axis="y",
					which="both",
					left=False,
					labelleft=False,
					right=False,
					labelright=False,
				)
				self.axes[i, 0].set_ylabel(self.annotations[i].label)
				self.axes[i, 0].yaxis.set_label_position(self.label_side)
				self.axes[i, 0].yaxis.label.update(self.label_kws)
				# self.axes[i, -1].yaxis.tick_right()  # ticks
				if type(self.annotations[i]) not in [anno_simpleCT,anno_imgCT]:
					self.axes[i, -1].yaxis.set_visible(True)
					self.axes[i, -1].tick_params(
						axis="y", which="both", right=True, labelright=True
					)
					self.axes[i, -1].yaxis.set_tick_params(**self.ticklabels_kws)
		elif self.axis == 1 and self.label_side == "right":
			self.ax.yaxis.tick_left()
			for i in range(self.axes.shape[0]):
				self.axes[i, -1].yaxis.set_visible(True)
				self.axes[i, -1].yaxis.label.set_visible(True)
				self.axes[i, -1].tick_params(
					axis="y",
					which="both",
					left=False,
					labelleft=False,
					right=False,
					labelright=False,
				)
				self.axes[i, -1].set_ylabel(self.annotations[i].label)
				self.axes[i, -1].yaxis.set_label_position(self.label_side)
				self.axes[i, -1].yaxis.label.update(self.label_kws)
				# self.axes[i, 0].yaxis.tick_left()  # ticks
				if type(self.annotations[i]) not in [anno_simpleCT,anno_imgCT]:
					self.axes[i, 0].yaxis.set_visible(True)
					self.axes[i, 0].tick_params(
						axis="y", which="both", left=True, labelleft=True
					)
					self.axes[i, 0].yaxis.set_tick_params(**self.ticklabels_kws)
		elif self.axis == 0 and self.label_side == "top":
			self.ax.xaxis.tick_bottom()
			for j in range(self.axes.shape[1]):
				self.axes[0, j].xaxis.set_visible(True) #0, the top axes
				self.axes[0, j].xaxis.label.set_visible(True)
				self.axes[0, j].tick_params(
					axis="x",
					which="both",
					top=False,
					labeltop=False,
					bottom=False,
					labelbottom=False,
				)
				self.axes[0, j].set_xlabel(self.annotations[j].label)
				self.axes[0, j].xaxis.set_label_position(self.label_side)
				self.axes[0, j].xaxis.label.update(self.label_kws)
				# self.axes[-1, j].xaxis.tick_bottom()  # ticks
				if type(self.annotations[j]) not in [anno_simpleCT,anno_imgCT]:
					self.axes[-1, j].xaxis.set_visible(True) # show ticks
					self.axes[-1, j].tick_params(
						axis="x", which="both", bottom=True, labelbottom=True
					)
					self.axes[-1, j].xaxis.set_tick_params(**self.ticklabels_kws)
		elif self.axis == 0 and self.label_side == "bottom":
			self.ax.xaxis.tick_top()
			for j in range(self.axes.shape[1]):
				self.axes[-1, j].xaxis.set_visible(True)
				self.axes[-1, j].xaxis.label.set_visible(True)
				self.axes[-1, j].tick_params(
					axis="x",
					which="both",
					top=False,
					labeltop=False,
					bottom=False,
					labelbottom=False,
				)
				self.axes[-1, j].set_xlabel(self.annotations[j].label)
				self.axes[-1, j].xaxis.set_label_position(self.label_side)
				self.axes[-1, j].xaxis.label.update(self.label_kws)
				# self.axes[0, j].xaxis.tick_top()  # ticks
				if type(self.annotations[j]) not in [anno_simpleCT,anno_imgCT]:
					self.axes[0, j].xaxis.set_visible(True)
					self.axes[0, j].tick_params(
						axis="x", which="both", top=True, labeltop=True
					)
					self.axes[0, j].xaxis.set_tick_params(**self.ticklabels_kws)

	def get_legend_list(self):
		if len(self.legend_dict) > 1 and self.legend_order in [True,"auto"]:
			self.legend_list=[self.legend_dict[k] for k in self.legend_dict.keys()]
			self.legend_list = sorted(self.legend_list, key=lambda x: x[3])
		elif len(self.legend_dict) > 1 and isinstance(self.legend_order,(list,tuple)):
			self.legend_list = [self.legend_dict[k] for k in self.legend_order if k in self.legend_dict]
		elif len(self.legend_dict) > 1:
			self.legend_list = [self.legend_dict[k] for k in self.legend_dict.keys()]
		else:
			self.legend_list=[]

	def collect_legends(self):
		"""
		Collect legends.
		Returns
		-------
		None
		"""
		if self.verbose >= 1:
			print("Collecting annotation legends..")
		self.legend_dict = {}  # handles(dict) / cmap, title, kws
		for annotation in self.annotations:
			if not annotation.legend:
				continue
			legend_kws = annotation.legend_kws.copy()
			# print(annotation.cmap,annotation)
			if (
				(annotation.cmap is None)
				or (hasattr(annotation.cmap, "N") and annotation.cmap.N < 256)
				or (
					type(annotation.cmap) == str
					and get_colormap(annotation.cmap).N < 256
				)
			):
				color_dict = annotation.color_dict
				if color_dict is None:
					continue
				self.legend_dict[annotation.label]=tuple([
						annotation.color_dict,
						annotation.label,
						legend_kws,
						len(annotation.color_dict),
						"color_dict",
					])
			else:
				if annotation.df.shape[1] == 1:
					array = annotation.df.iloc[:, 0].values
				else:
					array = annotation.df.values
				vmax = np.nanmax(array)
				vmin = np.nanmin(array)
				# print(vmax,vmin,annotation)
				legend_kws.setdefault("vmin", round(vmin, 2))
				legend_kws.setdefault("vmax", round(vmax, 2))
				self.legend_dict[annotation.label]=tuple(
					[
						annotation.cmap,
						annotation.label,
						legend_kws, 4, "cmap"]
				)
		self.get_legend_list() #self.legend_list will be created

		if self.label_side == "right":
			self.label_max_width = max(
				[ann.get_max_label_width() for ann in self.annotations]
			)
		else:
			self.label_max_width = max(
				[ann.get_ticklabel_width() for ann in self.annotations]
			)
		# self.label_max_height = max([ann.ax.yaxis.label.get_window_extent().height for ann in self.annotations])

	def plot_annotations(
		self, ax=None, subplot_spec=None, idxs=None, wspace=None, hspace=None
	):
		"""
		Plot annotations

		Parameters
		----------
		ax : ax
			axes to plot the annotations.
		subplot_spec : ax.figure.add_gridspec
			object from ax.figure.add_gridspec or matplotlib.gridspec.GridSpecFromSubplotSpec.
		idxs : list
			index to reorder df and df of annotation class.
		wspace : float
			if wspace not is None, use wspace, else wspace would be calculated based on gap.
		hspace : float
			if hspace not is None, use hspace, else hspace would be calculated based on gap.

		Returns
		-------
		self.ax
		"""
		# print(ax.figure.get_size_inches())
		self._set_label_kws(self.label_kws, self.ticklabels_kws)
		if self.verbose >= 1:
			print("Starting plotting HeatmapAnnotationCellTools")
		if ax is None:
			self.ax = plt.gca()
		else:
			self.ax = ax
		if idxs is None:
			# search for ann.row_order in anno_dendrogramCT
			for ann in self.annotations:
				if hasattr(ann,"row_order"):
					idxs=ann.row_order
			if idxs is None:
				idxs = [self.annotations[0].plot_data.index.tolist()]
		# print(idxs)
		if self.axis == 1:
			nrows = len(self.heights)
			ncols = len(idxs)
			height_ratios = self.heights
			width_ratios = [len(idx) for idx in idxs]
			wspace = (
				self.wgap
				* mm2inch
				* self.ax.figure.dpi
				/ (self.ax.get_window_extent().width / ncols)
				if wspace is None
				else wspace
			)  # 1mm=mm2inch inch
			hspace = (
				self.hgap
				* mm2inch
				* self.ax.figure.dpi
				/ (self.ax.get_window_extent().height / nrows)
				if hspace is None
				else hspace
			)  # fraction of height
		else:
			nrows = len(idxs)
			ncols = len(self.heights)
			width_ratios = self.heights
			height_ratios = [len(idx) for idx in idxs]
			hspace = (
				self.hgap
				* mm2inch
				* self.ax.figure.dpi
				/ (self.ax.get_window_extent().height / nrows)
				if hspace is None
				else hspace
			)
			wspace = (
				self.wgap
				* mm2inch
				* self.ax.figure.dpi
				/ (self.ax.get_window_extent().width / ncols)
				if wspace is None
				else wspace
			)  # The amount of width reserved for space between subplots, expressed as a fraction of the average axis width
		# print(wspace,hspace)
		if subplot_spec is None:
			self.gs = self.ax.figure.add_gridspec(
				nrows,
				ncols,
				hspace=hspace,
				wspace=wspace,
				height_ratios=height_ratios,
				width_ratios=width_ratios,
			)
		else:  # this ax is a subplot of another bigger figure.
			self.gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
				nrows,
				ncols,
				hspace=hspace,
				wspace=wspace,
				subplot_spec=subplot_spec,
				height_ratios=height_ratios,
				width_ratios=width_ratios,
			)
		self.axes = np.empty(shape=(nrows, ncols), dtype=object)
		self.fig = self.ax.figure
		self.ax.set_axis_off()
		for j, idx in enumerate(idxs): # columns if axis=1, rows if axis=0
			for i, ann in enumerate(self.annotations): #rows for axis=1, columns if axis=0
				# axis=1: left -> right, axis=0: bottom -> top.
				ann.reorder(idx)
				gs = self.gs[i, j] if self.axis == 1 else self.gs[j, i]
				# sharex = self.axes[0, j] if self.axis == 1 else self.axes[0, i]
				# sharey = self.axes[i, 0] if self.axis == 1 else self.axes[j, 0]
				sharex = self.axes[0, j] if self.axis == 1 else None
				sharey = None if self.axis == 1 else self.axes[j, 0]
				ax1 = self.ax.figure.add_subplot(gs, sharex=sharex, sharey=sharey)
				if self.axis == 1:
					ax1.set_xlim([0, len(idx)])
				else:
					ax1.set_ylim([0, len(idx)])
				ann.plot(ax=ax1, axis=self.axis) #subplot_spec=gs
				if self.axis == 1:
					# ax1.yaxis.set_visible(False)
					ax1.yaxis.label.set_visible(False)
					ax1.tick_params(
						left=False, right=False, labelleft=False, labelright=False
					)
					self.ax.spines["top"].set_visible(False)
					self.ax.spines["bottom"].set_visible(False)
					self.axes[i, j] = ax1
					if self.orientation == "down":
						ax1.invert_yaxis()
				else:  # horizonal
					if type(ann) != anno_simpleCT:
						# if sharey, one y axis inverted will affect other y axis?
						ax1.invert_yaxis()  # 20230312 fix bug for inversed row order in anno_labelCT.
					ax1.xaxis.label.set_visible(False)
					ax1.tick_params(
						top=False, bottom=False, labeltop=False, labelbottom=False
					)
					self.ax.spines["left"].set_visible(False)
					self.ax.spines["right"].set_visible(False)
					self.axes[j, i] = ax1
					if self.orientation == "left":
						ax1.invert_xaxis()

		self.set_axes_kws()
		self.legend_list = None
		if self.plot and self.plot_legend:
			self.plot_legends(ax=self.ax)
		# _draw_figure(self.ax.figure)
		return self.ax

	def show_ticklabels(self, labels, **kwargs):
		ha, va = "left", "center"
		if self.axis == 1:
			ax = self.axes[-1, 0] if self.orientation == "up" else self.axes[0, 0]
			rotation = -45 if self.orientation == "up" else 45
			ax.xaxis.set_visible(True)
			ax.xaxis.label.set_visible(True)
			if self.orientation == "up":
				ax.xaxis.set_ticks_position("bottom")
				ax.tick_params(axis="both", which="both", bottom=True, labelbottom=True)
			else:
				ax.xaxis.set_ticks_position("top")
				ax.tick_params(axis="both", which="both", top=True, labeltop=True)
		else:
			ax = self.axes[0, -1] if self.orientation == "left" else self.axes[0, 0]
			rotation = 0
			ax.yaxis.set_visible(True)
			ax.yaxis.label.set_visible(True)
			if self.orientation == "left":
				ax.yaxis.set_ticks_position("right")
				ax.tick_params(axis="both", which="both", right=True, labelright=True)
			else:
				ha = "right"
				ax.yaxis.set_ticks_position("left")
				ax.tick_params(axis="both", which="both", left=True, labelleft=True)
		kwargs.setdefault("rotation", rotation)
		kwargs.setdefault("ha", ha)
		kwargs.setdefault("va", va)
		kwargs.setdefault("rotation_mode", "anchor")
		if self.axis == 1:
			ax.set_xticklabels(labels, **kwargs)
		else:
			ax.set_yticklabels(labels, **kwargs)

	def plot_legends(self, ax=None):
		"""
		Plot legends.
		Parameters
		----------
		ax : axes for the plot, is ax is None, then ax=plt.figure()

		Returns
		-------
		None
		"""
		if self.legend_list is None:
			self.collect_legends() #create self.legend_dict and self.legend_list
		if len(self.legend_list) > 0:
			# if the legend is on the right side
			space = (
				self.label_max_width
				if (self.legend_side == "right" and self.label_side == "right")
				else 0
			)
			legend_hpad = (
				self.legend_hpad * mm2inch * self.ax.figure.dpi
			)  # mm to inch to pixel
			self.legend_axes, self.cbars, self.boundry = plot_legend_list(
				self.legend_list,
				ax=ax,
				space=space + legend_hpad,
				legend_side="right",
				gap=self.legend_gap,
				legend_width=self.legend_width,
				legend_vpad=self.legend_vpad,
				verbose=self.verbose
			)

# =============================================================================
class DendrogramPlotterCellTools(object):
    def __init__(self, data, linkage, metric, method, axis, label, rotate, dendrogram_kws=None):
        """Plot a dendrogram of the relationships between the columns of data
        """
        self.axis = axis
        if self.axis == 1:  # default 1, columns, when calculating dendrogram, each row is a point.
            data = data.T
        self.check_array(data)
        self.shape = self.data.shape
        self.metric = metric
        self.method = method
        self.label = label
        self.rotate = rotate
        self.dendrogram_kws = dendrogram_kws if not dendrogram_kws is None else {}
        if linkage is None:
            self.linkage = self.calculated_linkage
        else:
            self.linkage = linkage
        self.dendrogram = self.calculate_dendrogram()
        # Dendrogram ends are always at multiples of 5, who knows why
        ticks = np.arange(self.data.shape[0]) + 0.5  # xticklabels

        if self.label:
            ticklabels = _index_to_ticklabels(self.data.index)
            ticklabels = [ticklabels[i] for i in self.reordered_ind]
            if self.rotate:  # horizonal
                self.xticks = []
                self.yticks = ticks
                self.xticklabels = []

                self.yticklabels = ticklabels
                self.ylabel = _index_to_label(self.data.index)
                self.xlabel = ''
            else:  # vertical
                self.xticks = ticks
                self.yticks = []
                self.xticklabels = ticklabels
                self.yticklabels = []
                self.ylabel = ''
                self.xlabel = _index_to_label(self.data.index)
        else:
            self.xticks, self.yticks = [], []
            self.yticklabels, self.xticklabels = [], []
            self.xlabel, self.ylabel = '', ''

        self.dependent_coord = np.array(self.dendrogram['dcoord'])
        self.independent_coord = np.array(self.dendrogram['icoord']) / 10

    def check_array(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        # To avoid missing values and infinite values and further error, remove missing values
        # nrow = data.shape[0]
        # keep_col = data.apply(np.isfinite).sum() == nrow
        # if keep_col.sum() < 3:
        #     raise ValueError("There are too many missing values or infinite values")
        # data = data.loc[:, keep_col[keep_col].index.tolist()]
        if data.isna().sum().sum() > 0:
            data = data.apply(lambda x: x.fillna(x.median()),axis=1)
        self.data = data
        self.array = data.values

    def _calculate_linkage_scipy(self):  # linkage is calculated by columns
        # print(type(self.array),self.method,self.metric)
        linkage = hierarchy.linkage(self.array, method=self.method, metric=self.metric)
        return linkage  # array is a distance matrix?

    def _calculate_linkage_fastcluster(self):
        import fastcluster
        # Fastcluster has a memory-saving vectorized version, but only
        # with certain linkage methods, and mostly with euclidean metric
        # vector_methods = ('single', 'centroid', 'median', 'ward')
        euclidean_methods = ('centroid', 'median', 'ward')
        euclidean = self.metric == 'euclidean' and self.method in euclidean_methods
        if euclidean or self.method == 'single':
            return fastcluster.linkage_vector(self.array, method=self.method, metric=self.metric)
        else:
            linkage = fastcluster.linkage(self.array, method=self.method, metric=self.metric)
            return linkage

    @property
    def calculated_linkage(self):
        try:
            return self._calculate_linkage_fastcluster()
        except ImportError:
            if np.product(self.shape) >= 1000:
                msg = ("Clustering large matrix with scipy. Installing "
                       "`fastcluster` may give better performance.")
                warnings.warn(msg)
        return self._calculate_linkage_scipy()

    def calculate_dendrogram(self):  # Z (linkage) shape = (n,4), then dendrogram icoord shape = (n,4)
        return hierarchy.dendrogram(self.linkage, no_plot=True, labels=self.data.index.tolist(),
                                    get_leaves=True, **self.dendrogram_kws)  # color_threshold=-np.inf,

    @property
    def reordered_ind(self):
        """Indices of the matrix, reordered by the dendrogram"""
        return self.dendrogram['leaves']  # idx of the matrix

    def plot(self, ax, tree_kws):
        """Plots a dendrogram of the similarities between data on the axes
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object upon which the dendrogram is plotted
        """
        tree_kws = {} if tree_kws is None else tree_kws
        tree_kws.setdefault("linewidth", .5)
        tree_kws.setdefault("colors", None)
        # tree_kws.setdefault("colors", tree_kws.pop("color", (.2, .2, .2)))
        if self.rotate and self.axis == 0:  # 0 is rows, 1 is columns (default)
            coords = zip(self.dependent_coord, self.independent_coord)  # independent is icoord (x), horizontal
        else:
            coords = zip(self.independent_coord, self.dependent_coord)  # vertical
        # lines = LineCollection([list(zip(x,y)) for x,y in coords], **tree_kws)  #
        # ax.add_collection(lines)
        colors = tree_kws.pop('colors')
        if colors is None:
            # colors=self.dendrogram['leaves_color_list']
            colors = ['black'] * len(self.dendrogram['ivl'])
        for (x, y), color in zip(coords, colors):
            ax.plot(x, y, color=color, **tree_kws)
        number_of_leaves = len(self.reordered_ind)
        max_dependent_coord = max(map(max, self.dependent_coord))  # max y
        # if self.axis==0: #TODO
        #     ax.invert_yaxis()  # 20230227 fix the bug for inverse order of row dendrogram

        if self.rotate:  # horizontal
            ax.yaxis.set_ticks_position('right')
            # Constants 10 and 1.05 come from
            # `scipy.cluster.hierarchy._plot_dendrogram`
            ax.set_ylim(0, number_of_leaves)
            # ax.set_xlim(0, max_dependent_coord * 1.05)
            ax.set_xlim(0, max_dependent_coord)
            ax.invert_xaxis()
            ax.invert_yaxis()
        else:  # vertical
            # Constants 10 and 1.05 come from
            # `scipy.cluster.hierarchy._plot_dendrogram`
            ax.set_xlim(0, number_of_leaves)
            ax.set_ylim(0, max_dependent_coord)
        despine(ax=ax, bottom=True, left=True)
        ax.set(xticks=self.xticks, yticks=self.yticks,
               xlabel=self.xlabel, ylabel=self.ylabel)
        xtl = ax.set_xticklabels(self.xticklabels)
        ytl = ax.set_yticklabels(self.yticklabels, rotation='vertical')
        # Force a draw of the plot to avoid matplotlib window error
        if len(ytl) > 0 and axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation="horizontal")
        if len(xtl) > 0 and axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation="vertical")
        return self


class ClusterMapPlotterCellTools:
    """
    Clustermap (Heatmap) plotter.
    Plot heatmap / clustermap with annotation and legends.

    Parameters
    ----------
    data : dataframe
        pandas dataframe or numpy array.
    z_score : int
        whether to perform z score scale, either 0 for rows or 1 for columns, after scale,
        value range would be from -1 to 1.
    standard_scale : int
        either 0 for rows or 1 for columns, after scale,value range would be from 0 to 1.
    top_annotation : annotation: class of HeatmapAnnotationCellTools.
    bottom_annotation : class AnnotationBaseCellTools
        the same as top_annotation.
    left_annotation : class AnnotationBaseCellTools
        the same as top_annotation.
    right_annotation : class AnnotationBaseCellTools
        the same as top_annotation.
    row_cluster :bool
        whether to perform cluster on rows/columns.
    col_cluster :bool
        whether to perform cluster on rows/columns.
    row_cluster_method :str
        cluster method for row/columns linkage, such single, complete, average,weighted,
        centroid, median, ward. see scipy.cluster.hierarchy.linkage or
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) for detail.
    row_cluster_metric : str
        Pairwise distances between observations in n-dimensional space for row/columns,
        such euclidean, minkowski, cityblock, seuclidean, cosine, correlation, hamming, jaccard,
        chebyshev, canberra, braycurtis, mahalanobis, kulsinski et.al.
        centroid, median, ward. see scipy.cluster.hierarchy.linkage or
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html
    col_cluster_method :str
        same as row_cluster_method
    col_cluster_metric :str
        same as row_cluster_metric
    show_rownames :bool
        True (default) or False, whether to show row ticklabels.
    show_colnames : bool
        True of False, same as show_rownames.
    row_names_side :str
        right or left.
    col_names_side :str
        top or bottom.
    row_dendrogram :bool
        True or False, whether to show dendrogram.
    col_dendrogram :bool
        True or False, whether to show dendrogram.
    row_dendrogram_size :int
        default is 10mm.
    col_dendrogram_size :int
        default is 10mm.
    row_split :int or pd.Series or pd.DataFrame
        number of cluster for hierarchical clustering or pd.Series or pd.DataFrame,
        used to split rows or rows into subplots.
    col_split :int or pd.Series or pd.DataFrame
        int or pd.Series or pd.DataFrame, used to split rows or columns into subplots.
    dendrogram_kws :dict
        kws passed to hierarchy.dendrogram.
    tree_kws :dict
        kws passed to DendrogramPlotterCellTools.plot()
    row_split_gap :float
        default are 0.5 and 0.2 mm for row and col.
        default are 0.5 and 0.2 mm for row and col.
    mask :dataframe or array
        mask the data in heatmap, the cell with missing values of infinite values will be masked automatically.
    subplot_gap :float
        the gap between subplots, default is 1mm.
    legend :bool
        True or False, whether to plot heatmap legend, determined by cmap.
    legend_kws :dict
        kws passed to plot legend. If one want to change the outline color and linewidth of cbar:
        ```
        for cbar in cm.cbars:
            if isinstance(cbar,matplotlib.colorbar.Colorbar):
                cbar.outline.set_color('white')
                cbar.outline.set_linewidth(2)
                cbar.dividers.set_color('red')
                cbar.dividers.set_linewidth(2)
        ```
    plot :bool
        whether to plot or not.
    plot_legend :bool
        True or False, whether to plot legend, if False, legends can be plot with
        ClusterMapPlotterCellTools.plot_legends()
    legend_anchor :str
        ax_heatmap or ax, the ax to which legend anchor.
    legend_gap :float
        the columns gap between different legends.
    legend_width: float [mm]
        width of the legend, default is None (infer from data automatically)
    legend_hpad: float
        Horizonal space between the heatmap and legend, default is 2 [mm].
    legend_vpad: float
        Vertical space between the top of legend_anchor and legend, default is 5 [mm].
    legend_side :str
        right of left.
    cmap :str
        default is 'jet', the colormap for heatmap colorbar, see plt.colormaps().
    label :str
        the title (label) that will be shown in heatmap colorbar legend.
    xticklabels_kws :dict
        xticklabels or yticklabels kws, such as axis, which, direction, length, width,
        color, pad, labelsize, labelcolor, colors, zorder, bottom, top, left, right, labelbottom, labeltop,
        labelleft, labelright, labelrotation, grid_color, grid_linestyle and so on.
        For more information,see ?matplotlib.axes.Axes.tick_params or ?ax.tick_params.
    yticklabels_kws :dict
        the same as xticklabels_kws.
    rasterized :bool
        default is False, when the number of rows * number of cols > 100000, rasterized would be suggested
        to be True, otherwise the plot would be very slow.
    kwargs :kws passed to heatmap.

    Returns
    -------
    Class ClusterMapPlotterCellTools.
    """
    def __init__(self, data, z_score=None, standard_scale=None,
                 top_annotation=None, bottom_annotation=None, left_annotation=None, right_annotation=None,
                 row_cluster=True, col_cluster=True, row_cluster_method='average', row_cluster_metric='correlation',
                 col_cluster_method='average', col_cluster_metric='correlation',
                 show_rownames=False, show_colnames=False, row_names_side='right', col_names_side='bottom',
                 row_dendrogram=False, col_dendrogram=False, row_dendrogram_size=10, col_dendrogram_size=10,
                 row_split=None, col_split=None, dendrogram_kws=None, tree_kws=None,
                 row_split_order=None, col_split_order=None, row_split_gap=0.5, col_split_gap=0.2, mask=None,
                 subplot_gap=1, legend=True, legend_kws=None, plot=True, plot_legend=True,
                 legend_anchor='auto', legend_gap=7, legend_width=None, legend_hpad=1, legend_vpad=5,
                 legend_side='right', cmap='jet', label=None, xticklabels_kws=None, yticklabels_kws=None,
                 rasterized=False, legend_delta_x=None, verbose=1, xlabel = None, ylabel = None, **kwargs):
        self.kwargs = kwargs if not kwargs is None else {}
        self.data2d = self.format_data(data, mask, z_score, standard_scale)
        #print("self.data2d", self.data2d)
        #print("self.mask", self.mask)
        self.verbose=verbose
        self._define_kws(xticklabels_kws, yticklabels_kws)
        self.top_annotation = top_annotation
        self.bottom_annotation = bottom_annotation
        self.left_annotation = left_annotation
        self.right_annotation = right_annotation
        self.row_dendrogram_size = row_dendrogram_size
        self.col_dendrogram_size = col_dendrogram_size
        self.row_cluster = row_cluster
        self.col_cluster = col_cluster
        self.row_cluster_method = row_cluster_method
        self.row_cluster_metric = row_cluster_metric
        self.col_cluster_method = col_cluster_method
        self.col_cluster_metric = col_cluster_metric
        self.show_rownames = show_rownames
        self.show_colnames = show_colnames
        self.row_names_side = row_names_side
        self.col_names_side = col_names_side
        self.row_dendrogram = row_dendrogram
        self.col_dendrogram = col_dendrogram
        self.subplot_gap = subplot_gap
        self.dendrogram_kws = dendrogram_kws
        self.tree_kws = {} if tree_kws is None else tree_kws
        self.row_split = row_split
        self.col_split = col_split
        self.row_split_gap = row_split_gap
        self.col_split_gap = col_split_gap
        self.row_split_order=row_split_order
        self.col_split_order = col_split_order
        self.rasterized = rasterized
        self.legend = legend
        self.legend_kws = legend_kws if not legend_kws is None else {}
        self.legend_side = legend_side
        self.cmap = cmap
        self.label = label if not label is None else 'heatmap'
        self.legend_gap = legend_gap
        self.legend_width = legend_width
        self.legend_hpad = legend_hpad
        self.legend_vpad = legend_vpad
        self.legend_anchor = legend_anchor
        self.legend_delta_x=legend_delta_x
        self.xlabel = xlabel 
        self.ylabel = ylabel
        if plot:
            self.plot()
            if plot_legend:
                if legend_anchor=='auto':
                    if not self.right_annotation is None and self.legend_side=='right':
                        legend_anchor='ax'
                    else:
                        legend_anchor='ax_heatmap'
                if legend_anchor == 'ax_heatmap':
                    self.plot_legends(ax=self.ax_heatmap)
                else:
                    self.plot_legends(ax=self.ax)

        self.post_processing()

    def _define_kws(self, xticklabels_kws, yticklabels_kws):
        self.yticklabels_kws = {} if yticklabels_kws is None else yticklabels_kws
        # self.yticklabels_kws.setdefault('labelrotation', 0)
        self.xticklabels_kws = {} if xticklabels_kws is None else xticklabels_kws
        # self.xticklabels_kws.setdefault('labelrotation', 90)

    def format_data(self, data, mask=None, z_score=None, standard_scale=None):
        data2d = data.copy()
        self.kwargs.setdefault('vmin', np.nanmin(data.values))
        self.kwargs.setdefault('vmax', np.nanmax(data.values))
        if z_score is not None and standard_scale is not None:
            raise ValueError('Cannot perform both z-scoring and standard-scaling on data')
        if z_score is not None:
            data2d = self.z_score(data, z_score)
        if standard_scale is not None:
            data2d = self.standard_scale(data, standard_scale)
        self.mask = _check_mask(data2d, mask)
        #print("self.mask", self.mask)
        return data2d

    def _define_gs_ratio(self):
        self.top_heights = []
        self.bottom_heights = []
        self.left_widths = []
        self.right_widths = []
        if self.col_dendrogram:
            self.top_heights.append(self.col_dendrogram_size * mm2inch * self.ax.figure.dpi)
        if self.row_dendrogram:
            self.left_widths.append(self.row_dendrogram_size * mm2inch * self.ax.figure.dpi)
        if not self.top_annotation is None:
            self.top_heights.append(sum(self.top_annotation.heights) * mm2inch * self.ax.figure.dpi)
        else:
            self.top_heights.append(0)
        if not self.left_annotation is None:
            self.left_widths.append(sum(self.left_annotation.heights) * mm2inch * self.ax.figure.dpi)
        else:
            self.left_widths.append(0)
        if not self.bottom_annotation is None:
            self.bottom_heights.append(sum(self.bottom_annotation.heights) * mm2inch * self.ax.figure.dpi)
        else:
            self.bottom_heights.append(0)
        if not self.right_annotation is None:
            self.right_widths.append(sum(self.right_annotation.heights) * mm2inch * self.ax.figure.dpi)
        else:
            self.right_widths.append(0)
        heatmap_h = self.ax.get_window_extent().height - sum(self.top_heights) - sum(self.bottom_heights)
        heatmap_w = self.ax.get_window_extent().width - sum(self.left_widths) - sum(self.right_widths)
        self.heights = [sum(self.top_heights), heatmap_h, sum(self.bottom_heights)]
        self.widths = [sum(self.left_widths), heatmap_w, sum(self.right_widths)]

    def _define_axes(self, subplot_spec=None):
        wspace = self.subplot_gap * mm2inch * self.ax.figure.dpi / (self.ax.get_window_extent().width / 3)
        hspace = self.subplot_gap * mm2inch * self.ax.figure.dpi / (self.ax.get_window_extent().height / 3)

        if subplot_spec is None:
            self.gs = self.ax.figure.add_gridspec(3, 3, width_ratios=self.widths, height_ratios=self.heights,
                                                  wspace=wspace, hspace=hspace)
        else:
            self.gs = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 3, width_ratios=self.widths,
                                                                  height_ratios=self.heights,
                                                                  wspace=wspace, hspace=hspace,
                                                                  subplot_spec=subplot_spec)

        #left -> right, top -> bottom
        self.ax_heatmap = self.ax.figure.add_subplot(self.gs[1, 1])
        self.ax_top = self.ax.figure.add_subplot(self.gs[0, 1], sharex=self.ax_heatmap)
        self.ax_bottom = self.ax.figure.add_subplot(self.gs[2, 1], sharex=self.ax_heatmap)
        self.ax_left = self.ax.figure.add_subplot(self.gs[1, 0], sharey=self.ax_heatmap)
        self.ax_right = self.ax.figure.add_subplot(self.gs[1, 2], sharey=self.ax_heatmap)
        self.ax_heatmap.set_xlim([0, self.data2d.shape[1]])
        self.ax_heatmap.set_ylim([0, self.data2d.shape[0]])
        self.ax.yaxis.label.set_visible(False)
        self.ax_heatmap.yaxis.set_visible(False)
        self.ax_heatmap.xaxis.set_visible(False)
        self.ax.tick_params(axis='both', which='both',
                            left=False, right=False, labelleft=False, labelright=False,
                            top=False, bottom=False, labeltop=False, labelbottom=False)
        self.ax_heatmap.tick_params(axis='both', which='both',
                                    left=False, right=False, top=False, bottom=False,
                                    labeltop=False, labelbottom=False, labelleft=False, labelright=False)
        
        self.ax.set_axis_off()
        if self.xlabel is not None:
            self.ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            self.ax.set_ylabel(self.ylabel)

    def _define_top_axes(self):
        self.top_gs = None
        if self.top_annotation is None and self.col_dendrogram:
            self.ax_col_dendrogram = self.ax_top
            self.ax_top_annotation = None
        elif self.top_annotation is None and not self.col_dendrogram:
            self.ax_top_annotation = None
            self.ax_col_dendrogram = None
        elif self.col_dendrogram:
            self.top_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1, hspace=0, wspace=0,
                                                                      subplot_spec=self.gs[0, 1],
                                                                      height_ratios=[self.col_dendrogram_size,
                                                                                     sum(self.top_annotation.heights)])
            self.ax_top_annotation = self.ax_top.figure.add_subplot(self.top_gs[1, 0])
            self.ax_col_dendrogram = self.ax_top.figure.add_subplot(self.top_gs[0, 0])
        else:
            self.ax_top_annotation = self.ax_top
            self.ax_col_dendrogram = None
        self.ax_top.set_axis_off()

    def _define_left_axes(self):
        self.left_gs = None
        if self.left_annotation is None and self.row_dendrogram:
            self.ax_row_dendrogram = self.ax_left
            self.ax_left_annotation = None
        elif self.left_annotation is None and not self.row_dendrogram:
            self.ax_left_annotation = None
            self.ax_row_dendrogram = None
        elif self.row_dendrogram:
            self.left_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, hspace=0, wspace=0,
                                                                       subplot_spec=self.gs[1, 0],
                                                                       width_ratios=[self.row_dendrogram_size,
                                                                                     sum(self.left_annotation.heights)])
            self.ax_left_annotation = self.ax_left.figure.add_subplot(self.left_gs[0, 1])
            self.ax_row_dendrogram = self.ax_left.figure.add_subplot(self.left_gs[0, 0])
            self.ax_row_dendrogram.set_axis_off()
        else:
            self.ax_left_annotation = self.ax_left
            self.ax_row_dendrogram = None
        self.ax_left.set_axis_off()

    def _define_bottom_axes(self):
        if self.bottom_annotation is None:
            self.ax_bottom_annotation = None
        else:
            self.ax_bottom_annotation = self.ax_bottom
        self.ax_bottom.set_axis_off()

    def _define_right_axes(self):
        if self.right_annotation is None:
            self.ax_right_annotation = None
        else:
            self.ax_right_annotation = self.ax_right
        self.ax_right.set_axis_off()

    @staticmethod
    def z_score(data2d, axis=1):
        """
        Standarize the mean and variance of the data axis

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        normalized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.

        """
        if axis == 1:
            z_scored = data2d
        else:
            z_scored = data2d.T

        z_scored = (z_scored - z_scored.mean()) / z_scored.std()
        if axis == 1:
            return z_scored
        else:
            return z_scored.T

    @staticmethod
    def standard_scale(data2d, axis=1):
        """
        Divide the data by the difference between the max and min

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        standardized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.

        """
        # Normalize these values to range from 0 to 1
        if axis == 1:
            standardized = data2d
        else:
            standardized = data2d.T

        subtract = standardized.min()
        standardized = (standardized - subtract) / (
                standardized.max() - standardized.min())
        if axis == 1:
            return standardized
        else:
            return standardized.T

    def calculate_row_dendrograms(self, data):
        if self.row_cluster:
            self.dendrogram_row = DendrogramPlotterCellTools(data, linkage=None, axis=0,
                                                    metric=self.row_cluster_metric, method=self.row_cluster_method,
                                                    label=False, rotate=True, dendrogram_kws=self.dendrogram_kws)
        if not self.ax_row_dendrogram is None:
            self.ax_row_dendrogram.set_axis_off()

    def calculate_col_dendrograms(self, data):
        if self.col_cluster:
            self.dendrogram_col = DendrogramPlotterCellTools(data, linkage=None, axis=1,
                                                    metric=self.col_cluster_metric, method=self.col_cluster_method,
                                                    label=False, rotate=False, dendrogram_kws=self.dendrogram_kws)
            # self.dendrogram_col.plot(ax=self.ax_col_dendrogram)
        # despine(ax=self.ax_col_dendrogram, bottom=True, left=True, top=True, right=True)
        if not self.ax_col_dendrogram is None:
            self.ax_col_dendrogram.set_axis_off()

    def _reorder_rows(self):
        if self.verbose >= 1:
            print("Reordering rows..")
        if self.row_split is None and self.row_cluster:
            self.calculate_row_dendrograms(self.data2d)  # xind=self.dendrogram_row.reordered_ind
            self.row_order = [self.dendrogram_row.dendrogram['ivl']]  # self.data2d.iloc[:, xind].columns.tolist()
            return None
        elif isinstance(self.row_split, int) and self.row_cluster:
            self.calculate_row_dendrograms(self.data2d)
            self.row_clusters = pd.Series(hierarchy.fcluster(self.dendrogram_row.linkage, t=self.row_split,
                                                             criterion='maxclust'),
                                          index=self.data2d.index.tolist()).to_frame(name='cluster')\
                .groupby('cluster').apply(lambda x: x.index.tolist()).to_dict()
            #index=self.dendrogram_row.dendrogram['ivl']).to_frame(name='cluster')

        elif isinstance(self.row_split, (pd.Series, pd.DataFrame)):
            if isinstance(self.row_split, pd.Series):
                self.row_split = self.row_split.to_frame(name=self.row_split.name)
            cols = self.row_split.columns.tolist()
            row_clusters = self.row_split.groupby(cols).apply(lambda x: x.index.tolist())
            if len(cols)==1 and self.row_split_order is None:
                # calculate row_split_order using the mean across all samples in this group of
                # values of mean values across all samples
                self.row_split_order = row_clusters.apply(lambda x: self.data2d.loc[x].mean(axis=1).mean())\
                    .sort_values(ascending=False).index.tolist()
            else:
                self.row_split_order=row_clusters.sort_index().index.tolist()
            self.row_clusters = row_clusters.loc[self.row_split_order].to_dict()
        elif not self.row_cluster:
            self.row_order = [self.data2d.index.tolist()]
            return None
        else:
            raise TypeError("row_split must be integar or dataframe or series")

        self.row_order = []
        self.dendrogram_rows = []
        for i, cluster in enumerate(self.row_clusters):
            rows = self.row_clusters[cluster]
            if len(rows) <= 1:
                self.row_order.append(rows)
                self.dendrogram_rows.append(None)
                continue
            if self.row_cluster:
                self.calculate_row_dendrograms(self.data2d.loc[rows])
                self.dendrogram_rows.append(self.dendrogram_row)
                self.row_order.append(self.dendrogram_row.dendrogram['ivl'])
            else:
                self.row_order.append(rows)

    def _reorder_cols(self):
        if self.verbose >= 1:
            print("Reordering cols..")
        if self.col_split is None and self.col_cluster:
            self.calculate_col_dendrograms(self.data2d)
            self.col_order = [self.dendrogram_col.dendrogram['ivl']]  # self.data2d.iloc[:, xind].columns.tolist()
            return None
        elif isinstance(self.col_split, int) and self.col_cluster:
            self.calculate_col_dendrograms(self.data2d)
            self.col_clusters = pd.Series(hierarchy.fcluster(self.dendrogram_col.linkage, t=self.col_split,
                                                             criterion='maxclust'),
                                          index=self.data2d.columns.tolist()).to_frame(name='cluster')\
                .groupby('cluster').apply(lambda x: x.index.tolist()).to_dict()
            #index=self.dendrogram_col.dendrogram['ivl']).to_frame(name='cluster')

        elif isinstance(self.col_split, (pd.Series, pd.DataFrame)):
            if isinstance(self.col_split, pd.Series):
                self.col_split = self.col_split.to_frame(name=self.col_split.name)
            cols = self.col_split.columns.tolist()
            col_clusters = self.col_split.groupby(cols).apply(lambda x: x.index.tolist())
            if len(cols)==1 and self.col_split_order is None:
                # calculate col_split_order using the mean across all samples in this group of
                # values of mean values across all samples
                self.col_split_order = col_clusters.apply(lambda x: self.data2d.loc[:,x].mean().mean())\
                    .sort_values(ascending=False).index.tolist()
            else:
                self.col_split_order=col_clusters.sort_index().index.tolist()
            self.col_clusters = col_clusters.loc[self.col_split_order].to_dict()
        elif not self.col_cluster:
            self.col_order = [self.data2d.columns.tolist()]
            return None
        else:
            raise TypeError("row_split must be integar or dataframe or series")

        self.col_order = []
        self.dendrogram_cols = []
        for i, cluster in enumerate(self.col_clusters):
            cols = self.col_clusters[cluster]
            if len(cols) <= 1:
                self.col_order.append(cols)
                self.dendrogram_cols.append(None)
                continue
            if self.col_cluster:
                self.calculate_col_dendrograms(self.data2d.loc[:, cols])
                self.dendrogram_cols.append(self.dendrogram_col)
                self.col_order.append(self.dendrogram_col.dendrogram['ivl'])
            else:
                self.col_order.append(cols)

    def plot_dendrograms(self, row_order, col_order):
        rcmap = self.tree_kws.pop('row_cmap', None)
        ccmap = self.tree_kws.pop('col_cmap', None)
        tree_kws = self.tree_kws.copy()

        if self.row_cluster and self.row_dendrogram:
            if self.left_annotation is None:
                gs = self.gs[1, 0]
            else:
                gs = self.left_gs[0, 0]
            self.row_dendrogram_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(len(row_order), 1, hspace=self.hspace,
                                                                                 wspace=0, subplot_spec=gs,
                                                                                 height_ratios=[len(rows) for rows
                                                                                                in row_order])
            self.ax_row_dendrogram_axes = []
            for i in range(len(row_order)):
                ax1 = self.ax_row_dendrogram.figure.add_subplot(self.row_dendrogram_gs[i, 0])
                ax1.set_axis_off()
                self.ax_row_dendrogram_axes.append(ax1)

            try:
                if rcmap is None:
                    colors = ['black'] * len(self.dendrogram_rows)
                else:
                    colors = [plt.get_cmap(rcmap)(i) for i in range(len(self.dendrogram_rows))]
                for ax_row_dendrogram, dendrogram_row, color in zip(self.ax_row_dendrogram_axes, self.dendrogram_rows,
                                                                    colors):
                    if dendrogram_row is None:
                        continue
                    tree_kws['colors'] = [color] * len(dendrogram_row.dendrogram['ivl'])
                    dendrogram_row.plot(ax=ax_row_dendrogram, tree_kws=tree_kws)
            except:
                self.dendrogram_row.plot(ax=self.ax_row_dendrogram, tree_kws=self.tree_kws)

        if self.col_cluster and self.col_dendrogram:
            if self.top_annotation is None:
                gs = self.gs[0, 1]
            else:
                gs = self.top_gs[0, 0]
            self.col_dendrogram_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, len(col_order), hspace=0,
                                                                                 wspace=self.wspace, subplot_spec=gs,
                                                                                 width_ratios=[len(cols) for cols
                                                                                               in col_order])
            self.ax_col_dendrogram_axes = []
            for i in range(len(col_order)):
                ax1 = self.ax_col_dendrogram.figure.add_subplot(self.col_dendrogram_gs[0, i])
                ax1.set_axis_off()
                self.ax_col_dendrogram_axes.append(ax1)

            try:
                if ccmap is None:
                    colors = ['black'] * len(self.dendrogram_cols)
                else:
                    colors = [plt.get_cmap(ccmap)(i) for i in range(len(self.dendrogram_cols))]
                for ax_col_dendrogram, dendrogram_col, color in zip(self.ax_col_dendrogram_axes, self.dendrogram_cols,
                                                                    colors):
                    if dendrogram_col is None:
                        continue
                    tree_kws['colors'] = [color] * len(dendrogram_col.dendrogram['ivl'])
                    dendrogram_col.plot(ax=ax_col_dendrogram, tree_kws=tree_kws)
            except:
                self.dendrogram_col.plot(ax=self.ax_col_dendrogram, tree_kws=self.tree_kws)

                
                
    def plot_matrix(self, row_order, col_order):
        if self.verbose >= 1:
            print("Plotting matrix..")

        nrows = len(row_order)
        ncols = len(col_order)
        self.wspace = self.col_split_gap * mm2inch * self.ax.figure.dpi / (
                self.ax_heatmap.get_window_extent().width / ncols)  # 1mm=mm2inch inch
        self.hspace = self.row_split_gap * mm2inch * self.ax.figure.dpi / (
                self.ax_heatmap.get_window_extent().height / nrows)  # height

        self.heatmap_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows, ncols, hspace=self.hspace,
                                                                      wspace=self.wspace,
                                                                      subplot_spec=self.gs[1, 1],
                                                                      height_ratios=[len(rows) for rows in row_order],
                                                                      width_ratios=[len(cols) for cols in col_order])

        annot = self.kwargs.pop("annot", None)

        # Initialize annot_data to None by default
        annot_data = None

        if annot is not None and annot is not False:
            if isinstance(annot, bool):
                annot_data = self.data2d
            else:
                annot_data = annot.copy()
                if annot_data.shape != self.data2d.shape:
                    err = "`data` and `annot` must have same shape."
                    raise ValueError(err)

        self.heatmap_axes = np.empty(shape=(nrows, ncols), dtype=object)
        self.ax_heatmap.set_axis_off()
        for i, rows in enumerate(row_order):
            for j, cols in enumerate(col_order):
                ax1 = self.ax_heatmap.figure.add_subplot(self.heatmap_gs[i, j],
                                                          sharex=self.heatmap_axes[0, j],
                                                          sharey=self.heatmap_axes[i, 0])
                ax1.set_xlim([0, len(rows)])
                ax1.set_ylim([0, len(cols)])
                
                if self.xlabel is not None:
                    ax1.set_xlabel(self.xlabel)
                if self.ylabel is not None:
                    ax1.set_ylabel(self.ylabel)

                # Ensure annot_data is valid before accessing
                annot1 = None
                if annot_data is not None:
                    annot1 = annot_data.loc[rows, cols]

                plot_heatmap(self.data2d.loc[rows, cols], ax=ax1, cmap=self.cmap,
                             mask=self.mask.loc[rows, cols], rasterized=self.rasterized,
                             xticklabels='auto', yticklabels='auto', annot=annot1, 
                             xlabel  = self.xlabel, 
                             ylabel = self.ylabel,
                             **self.kwargs)
                self.heatmap_axes[i, j] = ax1
                ax1.yaxis.label.set_visible(False)
                ax1.xaxis.label.set_visible(False)
                ax1.tick_params(left=False, right=False, labelleft=False, labelright=False,
                                top=False, bottom=False, labeltop=False, labelbottom=False)
               # Draw purple border around main diagonal cells if the matrix is square
                if ncols == nrows and i == j:
                    # Add a basic red rectangle at (0, 0) with width and height matching the subplot
                    rect = patches.Rectangle((0, 0), len(cols), len(rows), fill=False, edgecolor='brown', lw=4)
                    ax1.add_patch(rect)

    def set_axes_labels_kws(self):
        # ax.set_xticks(ticks=np.arange(1, self.nrows + 1, 1), labels=self.plot_data.index.tolist())
        self.ax_heatmap.yaxis.set_tick_params(**self.yticklabels_kws)
        self.ax_heatmap.xaxis.set_tick_params(**self.xticklabels_kws)
        # self.ax_heatmap.tick_params(axis='both', which='both',
        #                             left=False, right=False, top=False, bottom=False)
        self.yticklabels = []
        self.xticklabels = []
        if (self.show_rownames and self.left_annotation is None and not self.row_dendrogram) \
                and ((not self.right_annotation is None) or (
                self.right_annotation is None and self.row_names_side == 'left')):  # tick left
            self.row_names_side='left'
            self.yticklabels_kws.setdefault('labelrotation', 0)
            for i in range(self.heatmap_axes.shape[0]):
                self.heatmap_axes[i, 0].yaxis.set_visible(True)
                self.heatmap_axes[i, 0].tick_params(axis='y', which='both', left=False, labelleft=True)
                self.heatmap_axes[i, 0].yaxis.set_tick_params(**self.yticklabels_kws)  # **self.ticklabels_kws
                plt.setp(self.heatmap_axes[i, 0].get_yticklabels(), rotation_mode='anchor',
                         ha='right', va='center')
                self.yticklabels.extend(self.heatmap_axes[i, 0].get_yticklabels())
        elif self.show_rownames and self.right_annotation is None:  # tick right
            self.row_names_side = 'right'
            self.yticklabels_kws.setdefault('labelrotation', 0)
            for i in range(self.heatmap_axes.shape[0]):
                self.heatmap_axes[i, -1].yaxis.tick_right()  # set_ticks_position('right')
                self.heatmap_axes[i, -1].yaxis.set_visible(True)
                self.heatmap_axes[i, -1].tick_params(axis='y', which='both', right=False, labelright=True)
                self.heatmap_axes[i, -1].yaxis.set_tick_params(**self.yticklabels_kws)
                plt.setp(self.heatmap_axes[i, -1].get_yticklabels(), rotation_mode='anchor',
                         ha='left', va='center')
                self.yticklabels.extend(self.heatmap_axes[i, -1].get_yticklabels())
        if self.show_colnames and self.top_annotation is None and not self.col_dendrogram and \
                ((not self.bottom_annotation is None) or (
                        self.bottom_annotation is None and self.col_names_side == 'top')):
            self.xticklabels_kws.setdefault('labelrotation', 90)
            for j in range(self.heatmap_axes.shape[1]):
                self.heatmap_axes[0, j].xaxis.tick_top()  # ticks
                self.heatmap_axes[0, j].xaxis.set_visible(True)
                self.heatmap_axes[0, j].tick_params(axis='x', which='both', top=False, labeltop=True)
                self.heatmap_axes[0, j].xaxis.set_tick_params(**self.xticklabels_kws)
                plt.setp(self.heatmap_axes[0, j].get_xticklabels(), rotation_mode = 'anchor',
                         ha = 'left',va='center') #rotation=90,ha=left is bottom, va is horizonal
                self.xticklabels.extend(self.heatmap_axes[0, j].get_xticklabels())
        elif self.show_colnames and self.bottom_annotation is None:  # tick bottom
            self.xticklabels_kws.setdefault('labelrotation', -90)
            for j in range(self.heatmap_axes.shape[1]):
                self.heatmap_axes[-1, j].xaxis.tick_bottom()  # ticks
                self.heatmap_axes[-1, j].xaxis.set_visible(True)
                self.heatmap_axes[-1, j].tick_params(axis='x', which='both', bottom=False, labelbottom=True)
                self.heatmap_axes[-1, j].xaxis.set_tick_params(**self.xticklabels_kws)
                plt.setp(self.heatmap_axes[-1, j].get_xticklabels(), rotation_mode='anchor',
                         ha='left', va='center')
                self.xticklabels.extend(self.heatmap_axes[-1, j].get_xticklabels())


    def collect_legends(self):
        if self.verbose >= 1:
            print("Collecting legends..")
        self.legend_list = []
        self.label_max_width = 0
        for annotation in [self.top_annotation, self.bottom_annotation, self.left_annotation, self.right_annotation]:
            if not annotation is None:
                annotation.collect_legends()
                if annotation.plot_legend and len(annotation.legend_list) > 0:
                    self.legend_list.extend(annotation.legend_list)
                # print(annotation.label_max_width,self.label_max_width)
                if annotation.label_max_width > self.label_max_width:
                    self.label_max_width = annotation.label_max_width
        if self.legend:
            vmax = self.kwargs.get('vmax', np.nanmax(self.data2d[self.data2d != np.inf]))
            vmin = self.kwargs.get('vmin', np.nanmin(self.data2d[self.data2d != -np.inf]))
            self.legend_kws.setdefault('vmin', round(vmin, 2))
            self.legend_kws.setdefault('vmax', round(vmax, 2))
            self.legend_list.append([self.cmap, self.label, self.legend_kws, 4,'cmap'])
            heatmap_label_max_width = max([label.get_window_extent().width for label in self.yticklabels]) if len(
                self.yticklabels) > 0 and self.row_names_side=='right' else 0
            # heatmap_label_max_height = max([label.get_window_extent().height for label in self.yticklabels]) if len(
            #     self.yticklabels) > 0 else 0
            if heatmap_label_max_width >= self.label_max_width or self.legend_anchor == 'ax_heatmap':
                self.label_max_width = heatmap_label_max_width #* 1.1
            if len(self.legend_list) > 1:
                self.legend_list = sorted(self.legend_list, key=lambda x: x[3])

    def plot_legends(self, ax=None):
        if self.verbose >= 1:
            print("Plotting legends..")
        if len(self.legend_list) > 0:
            if self.legend_side == 'right' and not self.right_annotation is None:
                space = self.label_max_width
            elif self.legend_side == 'right' and self.show_rownames and self.row_names_side=='right':
                space = self.label_max_width
            else:
                space=0
            # if self.right_annotation:
            #     space+=sum(self.right_widths)
            legend_hpad = self.legend_hpad * mm2inch * self.ax.figure.dpi
            self.legend_axes, self.cbars,self.boundry = \
                plot_legend_list(self.legend_list, ax=ax, space=space + legend_hpad,
                                  legend_side=self.legend_side, gap=self.legend_gap,
                                  delta_x=self.legend_delta_x,legend_width=self.legend_width,
                                 legend_vpad=self.legend_vpad)

    def plot(self, ax=None, subplot_spec=None, row_order=None, col_order=None):
        if self.verbose >= 1:
            print("Starting plotting..")
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self._define_gs_ratio()
        self._define_axes(subplot_spec)
        self._define_top_axes()
        self._define_left_axes()
        self._define_bottom_axes()
        self._define_right_axes()
        if row_order is None:
            if self.verbose >= 1:
                print("Starting calculating row orders..")
            self._reorder_rows()
            row_order = self.row_order
        if col_order is None:
            if self.verbose >= 1:
                print("Starting calculating col orders..")
            self._reorder_cols()
            col_order = self.col_order
        self.plot_matrix(row_order=row_order, col_order=col_order)
        if not self.top_annotation is None:
            gs = self.gs[0, 1] if not self.col_dendrogram else self.top_gs[1, 0]
            self.top_annotation._set_orentation('up')
            self.top_annotation.plot_annotations(ax=self.ax_top_annotation, subplot_spec=gs,
                                                 idxs=col_order, wspace=self.wspace)
        if not self.bottom_annotation is None:
            self.bottom_annotation._set_orentation('down')
            self.bottom_annotation.plot_annotations(ax=self.ax_bottom_annotation, subplot_spec=self.gs[2, 1],
                                                    idxs=col_order, wspace=self.wspace)
        if not self.left_annotation is None:
            gs = self.gs[1, 0] if not self.row_dendrogram else self.left_gs[0, 1]
            self.left_annotation._set_orentation('left')
            self.left_annotation.plot_annotations(ax=self.ax_left_annotation, subplot_spec=gs,
                                                  idxs=row_order, hspace=self.hspace)
        if not self.right_annotation is None:
            self.right_annotation._set_orentation('right')
            self.right_annotation.plot_annotations(ax=self.ax_right_annotation, subplot_spec=self.gs[1, 2],
                                                   idxs=row_order, hspace=self.hspace)
        if self.row_cluster or self.col_cluster:
            if self.row_dendrogram or self.col_dendrogram:
                self.plot_dendrograms(row_order, col_order)
        self.set_axes_labels_kws()
        self.collect_legends()
        # _draw_figure(self.ax_heatmap.figure)
        return self.ax

    def tight_layout(self, **tight_params):
        tight_params = dict(h_pad=.02, w_pad=.02) if tight_params is None else tight_params
        left = 0
        right = 1
        if self.legend and self.legend_side == 'right':
            right = self.boundry
        elif self.legend and self.legend_side == 'left':
            left = self.boundry
        tight_params.setdefault("rect", [left, 0, right, 1])
        self.ax.figure.tight_layout(**tight_params)

    def set_height(self, fig, height):
        matplotlib.figure.Figure.set_figheight(fig, height)  # convert mm to inches

    def set_width(self, fig, width):
        matplotlib.figure.Figure.set_figwidth(fig, width)  # convert mm to inches

    def post_processing(self):
        pass
# =============================================================================
def compositeCellTools(cmlist=None, main=0, ax=None, axis=1, row_gap=15, col_gap=15,
              legend_side='right', legend_gap=5, legend_y=0.8, legend_hpad=None,
              legend_width=None):
    """
    Assemble multiple ClusterMapPlotterCellTools objects vertically or horizontally together.

    Parameters
    ----------
    cmlist: list
        a list of ClusterMapPlotterCellTools (with plot=False).
    axis: int
        1 for columns (align the cmlist horizontally), 0 for rows (vertically).
    main: int
        use which as main ClusterMapPlotterCellTools, will influence row/col order. main is the index
        of cmlist.
    row/col_gap: float
        the row or columns gap between subplots, unit is mm [15].
    legend_side: str
        right,left [right].
    legend_gap: float
        row gap between two legends, unit is mm.

    Returns
    -------
    tuple:
        ax,legend_axes

    """
    if ax is None:
        ax = plt.gca()
    n = len(cmlist)
    wspace, hspace = 0, 0
    if axis == 1:  # horizontally
        wspace = col_gap * mm2inch * ax.figure.dpi / (ax.get_window_extent().width / n)
        nrows = 1
        ncols = n
        width_ratios = [cm.data2d.shape[1] for cm in cmlist]
        height_ratios = None
    else:  # vertically
        hspace = row_gap * mm2inch * ax.figure.dpi / (ax.get_window_extent().height / n)
        nrows = n
        ncols = 1
        width_ratios = None
        height_ratios = [cm.data2d.shape[0] for cm in cmlist]
    gs = ax.figure.add_gridspec(nrows, ncols, width_ratios=width_ratios,
                                height_ratios=height_ratios,
                                wspace=wspace, hspace=hspace)
    axes = []
    for i, cm in enumerate(cmlist):
        sharex = axes[0] if axis == 0 and i > 0 else None
        sharey = axes[0] if axis == 1 and i > 0 else None
        gs1 = gs[i, 0] if axis == 0 else gs[0, i]
        ax1 = ax.figure.add_subplot(gs1, sharex=sharex, sharey=sharey)
        ax1.set_axis_off()
        axes.append(ax1)
    cm_1 = cmlist[main]
    ax1 = axes[main]
    gs1 = gs[main, 0] if axis == 0 else gs[0, main]
    cm_1.plot(ax=ax1, subplot_spec=gs1, row_order=None, col_order=None)
    legend_list = cm_1.legend_list
    legend_names = [L[1] for L in legend_list]
    label_max_width = ax.figure.get_window_extent().width * cm_1.label_max_width / cm_1.ax.figure.get_window_extent().width
    for i, cm in enumerate(cmlist):
        if i == main:
            continue
        gs1 = gs[i, 0] if axis == 0 else gs[0, i]
        cm.plot(ax=axes[i], subplot_spec=gs1, row_order=cm_1.row_order, col_order=cm_1.col_order)
        for L in cm.legend_list:
            if L[1] not in legend_names:
                legend_names.append(L[1])
                legend_list.append(L)
        w = ax.figure.get_window_extent().width * cm.label_max_width / cm.ax.figure.get_window_extent().width
        if w > label_max_width:
            label_max_width = w
    if len(legend_list) == 0:
        return None
    legend_list = sorted(legend_list, key=lambda x: x[3])
    if legend_hpad is None:
        space = col_gap * mm2inch * ax.figure.dpi + label_max_width
    else:
        space = legend_hpad * ax.figure.dpi / 72
    legend_axes, cbars,boundry = \
        plot_legend_list(legend_list, ax=ax, space=space,
                        legend_side=legend_side, gap=legend_gap,
                         y0=legend_y,legend_width=legend_width)
    ax.set_axis_off()
    return ax,legend_axes

# def calculate_accuracy(subset_df):
#     """Calculate accuracy for the subset dataframe."""
#     correct_preds = (subset_df['truth'] == subset_df['pred']).sum()
#     total_preds = len(subset_df)
#     return correct_preds / total_preds if total_preds > 0 else 0


# def assign_soft_colors(hierarchy_df):
#     """
#     Assigns a color to each unique group in hierarchy_df if a 'colors' column does not exist.
    
#     Parameters:
#     - hierarchy_df (pd.DataFrame): Dataframe containing hierarchical group information.
    
#     Returns:
#     - hierarchy_df (pd.DataFrame): Updated dataframe with a 'colors' column assigning colors to each group.
#     """
#     # Define a set of soft colors
#     soft_colors = ["yellow", "#ADFF2F", "#87CEEB", "#FFB6C1", "#FFD700", "#FF6347", "#EEE8AA", "#98FB98", "#FF69B4", "#FFDAB9", "#E0FFFF"]
    
#     # Check if 'colors' column exists
#     if 'colors' not in hierarchy_df.columns:
#         unique_groups = hierarchy_df['Group'].unique()
#         color_map = {group: soft_colors[i % len(soft_colors)] for i, group in enumerate(unique_groups)}
        
#         # Map colors to groups
#         hierarchy_df['colors'] = hierarchy_df['Group'].map(color_map)
    
#     return hierarchy_df