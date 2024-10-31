#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pycomplexheatmapcelltools import (ClusterMapPlotterCellTools, compositeCellTools,
                         DendrogramPlotterCellTools, HeatmapAnnotationCellTools,
                          anno_lineplotCT, anno_imgCT,
	anno_scatterplotCT, anno_barplotCT, anno_boxplotCT,
	anno_labelCT, anno_simpleCT, AnnotationBaseCellTools,
	anno_dendrogramCT)


# from .clustermap import (heatmap, ClusterMapPlotter, composite,
#                          DendrogramPlotter)
from .oncoPrint import oncoprint, oncoPrintPlotter
# from .annotations import (
# 	HeatmapAnnotation,anno_lineplot,anno_img,
# 	anno_scatterplot,anno_barplot,anno_boxplot,
# 	anno_label,anno_simple,AnnotationBase,
# 	anno_dendrogram
# )
from .dotHeatmap import DotClustermapPlotter,dotHeatmap2d
from .colors import define_cmap
from .utils import use_pch_style
# __all__=['*']
from ._version import version as __version__
# __version__ = "1.6.5"

_ROOT = os.path.abspath(os.path.dirname(__file__))
