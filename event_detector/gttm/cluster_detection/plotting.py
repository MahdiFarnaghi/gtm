# -*- coding: UTF8 -*-
import os
import sys

from matplotlib.patches import Ellipse, Circle
from mpl_toolkits.mplot3d import art3d
from scipy.linalg import norm

# os.environ["PROJ_LIB"] = os.path.dirname(sys.executable) + os.sep + "Library\\share"
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
from matplotlib import pylab, transforms
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, box
from geopandas import GeoDataFrame
import shapely
from gttm.word_cloud.wordcloud import generate_wordcloud
import multiprocessing


class Plotter:
    noise_color = 'tab:gray'
    _chart_color = None

    @classmethod
    def generate_3d_chart_map_wordcloud_(cls, iter_num: int, gdf: GeoDataFrame, study_area: shapely.geometry.box,
                                         label=None,
                                         label_codes=None, cmap='brg',  # cmap='tab20b',
                                         show_plot=False,
                                         save_plot=False, plot_title="", plot_folder_path="", plot_file_name_prefix="",
                                         one_plot_per_cluster=False, multiprocess=False, verbose=False):
        if verbose:
            print('\tStart plotting ...')
        s_time = datetime.now()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # xdata = gdf.geometry.x.values
            # ydata = gdf.geometry.y.values
            xdata = gdf['x'].values
            ydata = gdf['y'].values
            tdata = ((gdf['t'].values - np.min(gdf['t'].values)) * 0.001).astype(float)
            cdata = gdf['c']
            colors = cls.get_colors(label_codes)

            plot_num = iter_num * 1000
            n_row = 2
            n_col = 2

            plt.close()
            plt.style.use('ggplot')

            # overall plot
            # print('\t\tData view')
            plot_num = plot_num + 1
            fig = cls.get_plot(plot_num, "Overall-{}".format(plot_title))
            cls.get_3d_chart_overall(cmap, fig, n_col, n_row, 1, '', xdata, ydata, zdata=tdata,
                                     study_area=study_area)
            cls.get_base_map_overall(cmap, fig, n_col, n_row, 3, '', study_area, xdata, ydata, zdata=tdata)
            cls.get_wordcloud_overall(cdata, fig, n_col, n_row, 4, '')
            plt.subplots_adjust(wspace=.3, hspace=.3)
            file_path = plot_folder_path + os.sep + "{}_overall.png".format(plot_file_name_prefix)
            cls._save_plot(file_path, plot_folder_path, save_plot, show_plot)

            # clusters plot
            # print('\t\tPlot all clusters')
            plot_num = plot_num + 1
            fig = cls.get_plot(plot_num, "All clusters")
            cls.get_3d_chart_all_clusters(colors, fig, label, label_codes, n_col, n_row, 1, '', xdata, ydata,
                                          zdata=tdata, add_cylinder=False)
            cls.get_3d_chart_all_clusters(colors, fig, label, label_codes, n_col, n_row, 2, '', xdata, ydata,
                                          zdata=tdata, add_cylinder=True)
            cls.get_base_map_all_clusters(colors, fig, label, label_codes, n_col, n_row, 3, '', study_area,
                                          xdata, ydata)
            plt.subplots_adjust(wspace=.3, hspace=.3)
            file_path = plot_folder_path + os.sep + "{}_all_clusters.png".format(plot_file_name_prefix)
            cls._save_plot(file_path, plot_folder_path, save_plot, show_plot)

            n_row = 2
            n_col = 3
            if not multiprocess:
                # plot for each cluster ==> iterative
                s_time_simple = datetime.now()
                if label is not None and label_codes is not None and one_plot_per_cluster:
                    for idx, l_code in enumerate(label_codes):
                        if l_code != -1:
                            # print('\t\tPlot cluster number {}'.format(l_code))
                            plot_num += 1
                            cls.plot_a_cluster(cdata, colors, idx, l_code, label, n_col, n_row, plot_file_name_prefix,
                                               plot_folder_path, plot_num, '', save_plot, show_plot, study_area,
                                               tdata, xdata, ydata)
                dur_simple = datetime.now() - s_time_simple
            else:
                # plot for each cluster ==> multi process
                s_time_multiprocessing = datetime.now()
                with warnings.catch_warnings():
                    if label is not None and label_codes is not None and one_plot_per_cluster:
                        proc_lst = []
                        for idx, l_code in enumerate(label_codes):
                            if l_code != -1:
                                plot_num += 1
                                p = multiprocessing.Process(target=cls.plot_a_cluster, args=(
                                    cdata, colors, idx, l_code, label, n_col, n_row, plot_file_name_prefix,
                                    plot_folder_path, plot_num, '', save_plot, show_plot, study_area,
                                    tdata, xdata, ydata,))
                                proc_lst.append(p)
                                p.start()

                                # cls.plot_a_cluster(cdata, colors, idx, l_code, label, n_col, n_row,
                                #                    plot_file_name_prefix,
                                #                    plot_folder_path, plot_num, plot_title, save_plot, show_plot,
                                #                    study_area,
                                #                    tdata, xdata, ydata)
                        for p in proc_lst:
                            p.join()
                dur_multiprocessing = datetime.now() - s_time_multiprocessing

            # print('-----------------')
            # with ThreadPoolExecutor() as executor:
            #     # plot for each cluster
            #     tasks = []
            #     if label is not None and label_codes is not None and one_plot_per_cluster:
            #         for idx, l_code in enumerate(label_codes):
            #             if l_code != -1:
            #                 # input_args = (cdata, colors, idx, l_code, label, n_col, n_row, plot_file_name_prefix,
            #                 #               plot_folder_path, plot_num, plot_title, save_plot, show_plot, study_area,
            #                 #               tdata, xdata, ydata)
            #                 executor.submit(cls.plot_a_cluster,
            #                                 cdata, colors, idx, l_code, label, n_col, n_row, plot_file_name_prefix,
            #                                 plot_folder_path, plot_num, plot_title, save_plot, show_plot, study_area,
            #                                 tdata, xdata, ydata)
            # print('................')

            # # plot for each cluster ==> multi process
            # s_time_multiprocessing_with_pool = datetime.now()
            # if label is not None and label_codes is not None and one_plot_per_cluster:
            #     pool = multiprocessing.Pool(multiprocessing.cpu_count())
            #     input_lst = []
            #     for idx, l_code in enumerate(label_codes):
            #         if l_code != -1:
            #             # proc.append(multiprocessing.Process(target=cls.plot_a_cluster1, args=(idx,)))
            #             input_lst.append(
            #                 (cdata, colors, idx, l_code, label, n_col, n_row, plot_file_name_prefix + 'mpwp',
            #                  plot_folder_path, plot_num, plot_title, save_plot, show_plot, study_area,
            #                  tdata, xdata, ydata,))
            #     pool.starmap(cls.plot_a_cluster, input_lst)
            #     pool.close()
            #     pool.join()
            # dur_multiprocessing_with_pool = datetime.now() - s_time_multiprocessing_with_pool

            # print('Simple: {}'.format(dur_simple.seconds))
            # print('Multiprocessing: {}'.format(dur_multiprocessing.seconds))
            # print('Multiprocessing with pool: {}'.format(dur_multiprocessing_with_pool.seconds))

            if show_plot:
                plt.show()
            else:
                plt.close('all')

            dur = datetime.now() - s_time
            if verbose:
                print('\tPlotting was finished ({} seconds).'.format(dur.seconds))

    # @classmethod
    # def plot_a_cluster1(cls, num):
    #     time.sleep(3)
    #     print('plot cluster ....... {}'.format(num))

    @classmethod
    def plot_a_cluster(cls, cdata, colors, idx, l_code, label, n_col, n_row, plot_file_name_prefix, plot_folder_path,
                       plot_num, plot_title, save_plot, show_plot, study_area, tdata, xdata, ydata):
        color = colors[idx]
        fig = cls.get_plot(plot_num, "Cluster number: {}".format(l_code))
        cls.get_3d_chart_one_cluster(color, fig, l_code, label, n_col, n_row, 1, plot_title, xdata,
                                     ydata, tdata)
        cls.get_3d_chart_one_cluster(color, fig, l_code, label, n_col, n_row, 2, plot_title, xdata,
                                     ydata, tdata, study_area=study_area)
        cls.get_3d_chart_one_cluster(color, fig, l_code, label, n_col, n_row, 3, plot_title, xdata,
                                     ydata, tdata, study_area=study_area, add_cylinder=True)
        cls.get_base_map_one_cluster(color, colors, fig, idx, l_code, label, n_col, n_row, 4,
                                     plot_title,
                                     study_area,
                                     xdata, ydata)
        cls.get_chart2d_one_cluster(color, colors, fig, idx, l_code, label, n_col, n_row, 5,
                                    plot_title,
                                    study_area,
                                    xdata, ydata, True)
        cls.get_wordcloud_one_cluster(cdata, fig, l_code, label, n_col, n_row, 6, plot_title)
        file_path = plot_folder_path + os.sep + "{}_cluster_num_{:02d}.png".format(
            plot_file_name_prefix,
            int(l_code))
        plt.subplots_adjust(wspace=.3, hspace=.3)
        cls._save_plot(file_path, plot_folder_path, save_plot, show_plot)

    @classmethod
    def _save_plot(cls, file_path, plot_folder_path, save_plot, show_plot):
        if save_plot and plot_folder_path != "":
            if not Path(file_path).parent.exists():
                os.makedirs(Path(file_path).parent.absolute())
            # TODO: It does not save the image with 300 dpi! It needs to be addressed.
            plt.savefig(file_path, dpi=300)

    @staticmethod
    def get_plot(plot_num=1, fig_title=""):
        if fig_title == "":
            fig_title = plot_num
        fig = plt.figure(num=plot_num, figsize=(20, 10))
        fig.suptitle(fig_title)
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 5),
                  'axes.labelsize': 'medium',
                  'axes.titlesize': 'medium',
                  'xtick.labelsize': 'small',
                  'ytick.labelsize': 'small'}
        pylab.rcParams.update(params)
        return fig

    @classmethod
    def generate_3d_chart_map_wordcloud_in_a_single_plot(cls, gdf: GeoDataFrame, study_area: shapely.geometry.box,
                                                         label=None,
                                                         label_codes=None, cmap='tab20b',
                                                         show_plot=False,
                                                         save_plot=False, plot_title="", plot_file_path="",
                                                         one_plot_per_cluster=False):
        # xdata = gdf.geometry.x.values
        # ydata = gdf.geometry.y.values
        xdata = gdf['x'].values
        ydata = gdf['y'].values
        tdata = ((gdf['t'].values - np.min(gdf['t'].values)) * 0.001).astype(float)
        cdata = gdf['c']

        plt.close()
        fig = plt.figure(figsize=(20, 10))

        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 5),
                  'axes.labelsize': 'medium',
                  'axes.titlesize': 'medium',
                  'xtick.labelsize': 'small',
                  'ytick.labelsize': 'small'}
        pylab.rcParams.update(params)

        n_row = 1
        if label is not None and label_codes is not None:
            n_row = n_row + 1
            if one_plot_per_cluster:
                n_row = n_row + len(label_codes[label_codes >= 0])

        n_col = 3
        # 3d chart of the input data
        cls.generate_3d_chart_ax(n_row, n_col, 1, cmap, fig, plot_title, xdata, ydata, tdata, label=label,
                                 label_codes=label_codes, one_plot_per_cluster=one_plot_per_cluster)
        # map of the input data
        cls.generate_map_ax(n_row, n_col, 2, cmap, fig, plot_title, study_area, xdata, ydata, tdata, label=label,
                            label_codes=label_codes, one_plot_per_cluster=one_plot_per_cluster)
        # word could of input data
        cls.generate_wordcloud_ax(n_row, n_col, 3, cdata, fig, plot_title, label=label,
                                  label_codes=label_codes, one_plot_per_cluster=one_plot_per_cluster)

        plt.tight_layout()
        plt.subplots_adjust(hspace=.8, wspace=.3, left=.05)
        plt.draw()

        if save_plot and plot_file_path != "":
            if not Path(plot_file_path).parent.exists():
                os.makedirs(Path(plot_file_path).parent.absolute())
            # TODO: It does not save the image with 300 dpi! It needs to be addressed.
            plt.savefig(plot_file_path, dpi=300)

        if show_plot:
            plt.show()
        else:
            plt.close()

    @classmethod
    def generate_wordcloud_ax(cls, n_row, n_col, col_num, cdata, fig, plot_title, label=None,
                              label_codes=None, one_plot_per_cluster=False):
        row_num = 0
        num = row_num * n_col + col_num

        cls.get_wordcloud_overall(cdata, fig, n_col, n_row, num, plot_title)

        # word cloud of clusters - 2nd row: IS NOT NEEDED
        row_num = 1
        num = row_num * n_col + col_num

        # word cloud of each cluster
        if label is not None and label_codes is not None and one_plot_per_cluster:
            for idx, l_code in enumerate(label_codes):
                if l_code != -1:
                    row_num = row_num + 1
                    num = row_num * n_col + col_num
                    cls.get_wordcloud_one_cluster(cdata, fig, l_code, label, n_col, n_row, num, plot_title)

    @classmethod
    def get_wordcloud_one_cluster(cls, cdata, fig, l_code, label, n_col, n_row, num, plot_title):
        title = '{}: Cluster number {}'.format(plot_title, l_code)
        Plotter.get_wordcloud_on_axis(cdata[label == l_code], fig, n_col, n_row, num, title)

    @classmethod
    def get_wordcloud_overall(cls, cdata, fig, n_col, n_row, num, plot_title):
        # overall world cloud - 1st row
        title = '{}: Word Cloud of Tweets'.format(plot_title)
        Plotter.get_wordcloud_on_axis(cdata, fig, n_col, n_row, num, title)

    @staticmethod
    def get_wordcloud_on_axis(cdata, fig, n_col, n_row, num, title):
        wordcloud = generate_wordcloud(cdata)
        ax_wordcloud = fig.add_subplot(n_row, n_col, num)
        imgplot = plt.imshow(wordcloud)
        ax_wordcloud.set_title('{} Word cloud'.format(title))
        ax_wordcloud.get_xaxis().set_visible(False)
        ax_wordcloud.get_yaxis().set_visible(False)

    @classmethod
    def generate_map_ax(cls, n_row, n_col, col_num, cmap, fig, plot_title, study_area, xdata, ydata, zdata, label=None,
                        label_codes=None, one_plot_per_cluster=False):
        row_num = 0
        num = row_num * n_col + col_num

        cls.get_base_map_overall(cmap, fig, n_col, n_row, num, plot_title, study_area, xdata, ydata, zdata)

        # map of clusters - 2nd row
        row_num = 1
        num = row_num * n_col + col_num

        colors = cls.get_colors(label_codes)

        cls.get_base_map_all_clusters(colors, fig, label, label_codes, n_col, n_row, num, plot_title, study_area, xdata,
                                      ydata)

        # one map for each cluster
        if label is not None and label_codes is not None and one_plot_per_cluster:
            for idx, l_code in enumerate(label_codes):
                if l_code != -1:
                    col = colors[idx]
                    title = '{}: Cluster number {} ({})'.format(plot_title, l_code, col)
                    row_num = row_num + 1
                    num = row_num * n_col + col_num
                    ax_2d, m = cls.get_base_map_on_axis(fig, n_col, n_row, num, title, study_area)
                    x, y = m(xdata, ydata)
                    m.scatter(x[label == l_code], y[label == l_code], marker='D', c=col)

    @classmethod
    def get_base_map_all_clusters(cls, colors, fig, label, label_codes, n_col, n_row, num, plot_title, study_area,
                                  xdata, ydata, add_cluster_ellipse=False):

        if label is not None and label_codes is not None:
            ax_2d, m = cls.get_base_map_on_axis(fig, n_col, n_row, num, plot_title, study_area)
            colors = cls.get_colors(label_codes)
            for idx, l_code in enumerate(label_codes):
                if l_code != -1:
                    color = colors[idx]  # chart_color[idx]
                    m.scatter(xdata[label == l_code], ydata[label == l_code], marker='D', c=color)

            # cls.get_base_map_one_cluster(color, colors, fig, idx, l_code, label, n_col, n_row, num, plot_title,
            #                              study_area,
            #                              xdata, ydata, add_cluster_ellipse=add_cluster_ellipse)

    @classmethod
    def get_chart2d_one_cluster(cls, noise_color, cluster_colors, fig, idx, l_code, label, n_col, n_row, num,
                                plot_title, study_area,
                                xdata,
                                ydata, add_cluster_ellipse=False):
        ax_2d = fig.add_subplot(n_row, n_col, num)

        if l_code != -1:
            if add_cluster_ellipse:
                ell = ShapeDrawer.generate_3_confidence_ellipse2d(xdata[label == l_code], ydata[label == l_code], ax_2d)
            color = cluster_colors[idx]  # chart_color[idx]
            ax_2d.scatter(xdata[label == l_code], ydata[label == l_code], marker='D', c=color)
        else:
            ax_2d.scatter(xdata[label == l_code], ydata[label == l_code], marker='+', c=noise_color)
        return ax_2d

    @classmethod
    def get_base_map_one_cluster(cls, noise_color, cluster_colors, fig, idx, l_code, label, n_col, n_row, num,
                                 plot_title, study_area,
                                 xdata,
                                 ydata):
        title = plot_title
        sa = study_area
        ax_2d, m = cls.get_base_map_on_axis(fig, n_col, n_row, num, title, sa)
        x, y = m(xdata, ydata)
        if l_code != -1:
            noise_color = cluster_colors[idx]  # chart_color[idx]
            m.scatter(x[label == l_code], y[label == l_code], marker='D', c=noise_color)
        else:
            m.scatter(x[label == l_code], y[label == l_code], marker='+', c=noise_color)
        return ax_2d, m

    @classmethod
    def get_colors(cls, label_codes):
        spectral = plt.cm.get_cmap("Spectral")
        colors = [mpl_colors.to_hex(spectral(each))
                  for each in np.linspace(0, 1, len(label_codes))]
        # if len(label_codes[label_codes >= 0]) < len(cls.get_chart_color()):
        #     colors = cls.get_chart_color()
        return colors

    @classmethod
    def get_base_map_overall(cls, cmap, fig, n_col, n_row, num, plot_title, study_area, xdata, ydata, zdata):
        # overall map - 1st row
        title = '{}: Map of Tweets'.format(plot_title)
        ax_2d, m = cls.get_base_map_on_axis(fig, n_col, n_row, num, title, study_area)
        x, y = m(xdata, ydata)
        m.scatter(x, y, marker='D', c=zdata, cmap=cmap)

    @classmethod
    def get_base_map_on_axis(cls, fig, n_col, n_row, num, map_title, study_area):

        ax_2d = fig.add_subplot(n_row, n_col, num)
        ax_2d.set_title(map_title)
        m = None
        if study_area is not None:
            m = Basemap(epsg=4326, lat_0=0, lon_0=3, resolution='h', llcrnrlon=study_area.envelope.bounds[0],
                        llcrnrlat=study_area.envelope.bounds[
                            1], urcrnrlon=study_area.envelope.bounds[2], urcrnrlat=study_area.envelope.bounds[3])
            # m = Basemap(projection='merc', lat_0=0, lon_0=3, resolution='h', llcrnrlon=study_area.envelope.bounds[0],
            #             llcrnrlat=study_area.envelope.bounds[
            #                 1], urcrnrlon=study_area.envelope.bounds[2], urcrnrlat=study_area.envelope.bounds[3])
        else:
            m = Basemap(epsg=4326, lat_0=0, lon_0=3, resolution='h')
            # m = Basemap(projection='merc', lat_0=0, lon_0=3, resolution='h')

        m.fillcontinents(color='beige', lake_color='aqua', zorder=0)
        m.drawcoastlines(color='gray')
        m.drawcountries(color='gray')
        m.drawstates(color='gray')
        return ax_2d, m

    @classmethod
    def add_wms(cls, m):
        # m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1500, verbose=True)
        ## Add wms layer
        wms_server = 'http://ows.mundialis.de/services/service?'
        # wms_layername = 'OSM-WMS'
        wms_layername = 'OSM-Overlay-WMS'
        # wms_layername = 'SRTM30-Hillshade'
        # wms_layername = 'TOPO-OSM-WMS'
        m.wmsimage(wms_server, layers=[wms_layername], verbose=False, xpixels=500)

    @classmethod
    def generate_3d_chart_ax(cls, n_row, n_col, col_num, cmap, fig, plot_title, xdata, ydata, zdata, label=None,
                             label_codes=None, one_plot_per_cluster=False):
        row_num = 0
        num = row_num * n_col + col_num

        cls.get_3d_chart_overall(cmap, fig, n_col, n_row, num, plot_title, xdata, ydata, zdata, study_area=study_area)

        # 3d scatter plot of clusters - 2nd row: IS NOT NEEDED
        row_num = 1
        num = row_num * n_col + col_num

        colors = cls.get_colors(label_codes)
        cls.get_3d_chart_all_clusters(colors, fig, label, label_codes, n_col, n_row, num, plot_title, xdata, ydata,
                                      zdata)

        # one scatter plot for each cluster
        if label is not None and label_codes is not None and one_plot_per_cluster:
            for idx, l_code in enumerate(label_codes):
                if l_code != -1:
                    color = colors[idx]
                    row_num = row_num + 1
                    num = row_num * n_col + col_num
                    cls.get_3d_chart_one_cluster(color, fig, l_code, label, n_col, n_row, num, plot_title, xdata, ydata,
                                                 zdata)

    @classmethod
    def get_3d_chart_one_cluster(cls, color, fig, l_code, label, n_col, n_row, num, plot_title, xdata, ydata, zdata,
                                 add_cylinder=False,
                                 study_area=None):
        title = ''  # '{}: Scatter plot of cluster number {} ({})'.format(plot_title, l_code, color)

        ax_3d = cls.get_3d_chart_on_axis(fig, n_col, n_row, num, title)

        if add_cylinder:
            ShapeDrawer.plot_cluster_as_cylinder(ax_3d, xdata[label == l_code], ydata[label == l_code],
                                                 zdata[label == l_code],
                                                 c=color, alpha=0.3)

        ax_3d.scatter3D(xdata[label == l_code], ydata[label == l_code], zdata[label == l_code], c=color)

        if study_area is not None:
            cls.add_base_map_to_3d_chart(ax_3d, study_area)

    @classmethod
    def get_3d_chart_all_clusters(cls, colors, fig, label, label_codes, n_col, n_row, num, plot_title, xdata, ydata,
                                  zdata, add_cylinder=False, study_area=None):
        if label is not None and label_codes is not None:
            title = ''  # '{}: Scatter plot of clusters'.format(plot_title)

            ax_3d = cls.get_3d_chart_on_axis(fig, n_col, n_row, num, title)

            for idx, l_code in enumerate(label_codes):
                col = cls.noise_color
                if l_code != -1:
                    col = colors[idx]

                    if add_cylinder:
                        ShapeDrawer.plot_cluster_as_cylinder(ax_3d, xdata[label == l_code], ydata[label == l_code],
                                                             zdata[label == l_code], c=col)

                    ax_3d.scatter3D(xdata[label == l_code], ydata[label == l_code], zdata[label == l_code], c=col,
                                    marker='D')
                else:
                    ax_3d.scatter3D(xdata[label == l_code], ydata[label == l_code], zdata[label == l_code], c=col,
                                    marker='+', alpha=0.3)
            if study_area is not None:
                cls.add_base_map_to_3d_chart(ax_3d, study_area)

    @classmethod
    def get_3d_chart_overall(cls, cmap, fig, n_col, n_row, num, plot_title, xdata, ydata, zdata, study_area):
        # overall 3d scatter plot - 1st row
        title = ''  # '{} 3D Chart'.format(plot_title)
        ax_3d = cls.get_3d_chart_on_axis(fig, n_col, n_row, num, title)

        # ax_3d.scatter3D(xdata, ydata, zdata, c=zdata, cmap=cmap, marker='D')
        ax_3d.scatter3D(xdata, ydata, zdata)

        cls.add_base_map_to_3d_chart(ax_3d, study_area)

    @classmethod
    def add_base_map_to_3d_chart(cls, ax_3d, study_area):
        m = Basemap(epsg=4326, lat_0=0, lon_0=3, resolution='h', llcrnrlon=study_area.envelope.bounds[0],
                    llcrnrlat=study_area.envelope.bounds[
                        1], urcrnrlon=study_area.envelope.bounds[2], urcrnrlat=study_area.envelope.bounds[3])
        # m = Basemap(projection='merc', lat_0=0, lon_0=3, resolution='h', llcrnrlon=study_area.envelope.bounds[0],
        #             llcrnrlat=study_area.envelope.bounds[
        #                 1], urcrnrlon=study_area.envelope.bounds[2], urcrnrlat=study_area.envelope.bounds[3])
        ax_3d.add_collection3d(m.drawcoastlines(linewidth=0.25, color='gray'))
        ax_3d.add_collection3d(m.drawstates(linewidth=0.25, color='gray'))
        ax_3d.add_collection3d(m.drawcountries(linewidth=0.35, color='gray'))

    @classmethod
    def get_3d_chart_on_axis(cls, fig, n_col, n_row, num, title):
        ax_3d = fig.add_subplot(n_row, n_col, num, aspect='equal', projection='3d')  # Axes3D(fig)
        ax_3d.set_title(title)
        ax_3d.azim = -80
        ax_3d.elev = 25
        ax_3d.dist = 6
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('T')
        # ax_3d.set_xticklabels('')
        # ax_3d.set_yticklabels('')
        # ax_3d.set_zticklabels('')
        return ax_3d

    @classmethod
    def generate_3d_map(cls, gdf: GeoDataFrame, show_map=False, save_map=False, map_title="", map_filepath=""):
        # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

        if show_map or save_map:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')  # Axes3D(fig)

            xdata = gdf['x'].values
            ydata = gdf['y'].values
            zdata = gdf['timestamp'].values - np.min(gdf['timestamp'].values)

            ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='BuGn')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Time')
            plt.draw()

        if show_map:
            plt.show()

        if save_map and map_filepath != "":
            if not Path(map_filepath).parent.exists():
                Path(map_filepath).parent.absolute().mkdir()
            plt.savefig(map_filepath, dpi=300)

    # def _add_world_polygons_to_xy_plate(ax):
    #     polys = []
    #     for polygon in map.landpolygons:
    #         polys.append(polygon.get_coords())

    #     lc = PolyCollection(polys, edgecolor='black',
    #                         facecolor='#DDDDDD', closed=False)

    #     ax.add_collection3d(lc)

    # chart_color = ['tab:gray', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink',
    #                'tab:olive', 'tab:cyan', 'aqua', 'aquamarine', 'azure', 'beige', 'black', 'blue', 'brown',
    #                'chartreuse', 'chocolate', 'coral', 'crimson', 'cyan', 'darkblue', 'darkgreen', 'fuchsia', 'gold',
    #                'goldenrod',
    #                'green', 'grey', 'indigo', 'ivory', 'khaki', 'lavender', 'lightblue', 'lightgreen', 'lime', 'magenta',
    #                'maroon', 'navy', 'olive', 'orange', 'orangered', 'orchid', 'pink', 'plum', 'purple', 'red', 'salmon',
    #                'sienna', 'silver', 'tan', 'teal', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'yellow',
    #                'yellowgreen']
    @classmethod
    def get_chart_color(cls):
        if cls._chart_color is None:
            numb_of_colors = 200.0
            cmap = cls.rand_cmap(numb_of_colors, verbose=False)
            cls._chart_color = [cmap(i / numb_of_colors) for i in range(numb_of_colors)]
        return cls._chart_color

    # Generate random colormap
    def rand_cmap(cls, nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
        """
        Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
        :param nlabels: Number of labels (size of colormap)
        :param type: 'bright' for strong colors, 'soft' for pastel colors
        :param first_color_black: Option to use first color as black, True or False
        :param last_color_black: Option to use last color as black, True or False
        :param verbose: Prints the number of labels and shows the colormap. True or False
        :return: colormap for matplotlib
        """
        from matplotlib.colors import LinearSegmentedColormap
        import colorsys
        import numpy as np

        if type not in ('bright', 'soft'):
            print('Please choose "bright" or "soft" for type')
            return

        if verbose:
            print('Number of labels: ' + str(nlabels))

        # Generate color map for bright colors, based on hsv
        if type == 'bright':
            randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                              np.random.uniform(low=0.2, high=1),
                              np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

            # Convert HSV list to RGB
            randRGBcolors = []
            for HSVcolor in randHSVcolors:
                randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

            if first_color_black:
                randRGBcolors[0] = [0, 0, 0]

            if last_color_black:
                randRGBcolors[-1] = [0, 0, 0]

            random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

        # Generate soft pastel colors, by limiting the RGB spectrum
        if type == 'soft':
            low = 0.6
            high = 0.95
            randRGBcolors = [(np.random.uniform(low=low, high=high),
                              np.random.uniform(low=low, high=high),
                              np.random.uniform(low=low, high=high)) for i in xrange(nlabels)]

            if first_color_black:
                randRGBcolors[0] = [0, 0, 0]

            if last_color_black:
                randRGBcolors[-1] = [0, 0, 0]
            random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

        # Display colorbar
        if verbose:
            from matplotlib import colors, colorbar
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

            bounds = np.linspace(0, nlabels, nlabels + 1)
            norm = colors.BoundaryNorm(bounds, nlabels)

            cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                       boundaries=bounds, format='%1i', orientation=u'horizontal')

        return random_colormap

    # @classmethod
    # def plot_clusters_old(cls, xy, labels, label_codes, space, reachability, show_plot=False, save_plot=False,
    #                       plot_title="", plot_filepath=""):
    #     if not show_plot and not save_plot:
    #         return
    #
    #     plt.figure(figsize=(10, 7))
    #     plt.title(plot_title)
    #     G = gridspec.GridSpec(2, 1)
    #     ax1 = plt.subplot(G[0, :])
    #     ax2 = plt.subplot(G[1, 0])
    #
    #     ## Plotting
    #
    #     # label_codes = label_codes[label_codes >= 0]
    #     for idx, l_code in enumerate(label_codes):
    #         Xk = xy[labels == l_code]
    #
    #         c = np.random.rand(3, 1)
    #         if idx < len(cls.get_chart_color()):
    #             c = cls.get_chart_color()[idx]
    #
    #         ax2.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3)
    #     ax2.plot(xy[labels == -1, 0], xy[labels == -1, 1], 'k+', alpha=0.1)
    #     ax2.set_title('Clustered points - Optics')
    #
    #     ## Generating the reachability
    #     for idx, l_code in enumerate(label_codes):
    #         # for k, c in zip(range(0, 5), color):
    #         Xk = space[labels == l_code]
    #         Rk = reachability[labels == l_code]
    #
    #         c = np.random.rand(3, 1)
    #         if idx < len(cls.get_chart_color()):
    #             c = cls.get_chart_color()[idx]
    #
    #     ax1.plot(Xk, Rk, c, alpha=0.3)
    #     ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    #     ax1.plot(space, np.full_like(space, 0.75, dtype=float), 'k-', alpha=0.5)
    #     ax1.plot(space, np.full_like(space, 0.25, dtype=float), 'k-.', alpha=0.5)
    #     ax1.set_ylabel('Reachability (epsilon distance)')
    #     ax1.set_title('Reachability Plot - Optics')
    #
    #     if save_plot and plot_filepath != "":
    #         if not Path(plot_filepath).parent.exists():
    #             os.makedirs(Path(plot_filepath).parent.absolute())
    #         plt.savefig(plot_filepath, dpi=300)
    #
    #     if show_plot:
    #         plt.show()
    #     else:
    #         plt.close()


class ShapeDrawer:
    @classmethod
    def plot_cluster_as_multiple_cylinder(cls, ax, xdata, ydata, zdata, alpha=.6, c='blue', expansion_scale=1,
                                          add_to_upper_z_limit=1,
                                          add_to_lower_z_limit=-1, step=.4):
        import math
        z_min = math.floor(np.min(zdata))
        z_max = math.ceil(np.max(zdata))
        for i in np.arange(z_min, z_max, step):
            ShapeDrawer.plot_cluster_as_cylinder(ax,
                                                 xdata[np.logical_and(zdata >= i, zdata < i + 1)],
                                                 ydata[np.logical_and(zdata >= i, zdata < i + 1)],
                                                 zdata[np.logical_and(zdata >= i, zdata < i + 1)],
                                                 alpha=alpha, c=c,
                                                 expansion_scale=expansion_scale,
                                                 add_to_upper_z_limit=add_to_upper_z_limit,
                                                 add_to_lower_z_limit=add_to_lower_z_limit)
            print(i)

        pass

    @classmethod
    def plot_cluster_as_cylinder(cls, ax, xdata, ydata, zdata, alpha=.6, c='blue', expansion_scale=1,
                                 add_to_upper_z_limit=1,
                                 add_to_lower_z_limit=-1):
        if len(xdata) < 1:
            return
        try:
            import math
            # set z limit
            ax.set_zlim(math.floor(np.min(zdata)) + add_to_lower_z_limit,
                        math.ceil(np.max(zdata)) + add_to_upper_z_limit)
            # axis and radius
            p0 = np.array([np.mean(xdata), np.mean(ydata), np.min(zdata)])
            p1 = np.array([np.mean(xdata), np.mean(ydata), np.max(zdata)])
            R = max([np.max(xdata) - np.min(xdata), np.max(ydata) - np.min(ydata)]) / 2 * expansion_scale
            # vector in direction of axis
            v = p1 - p0
            if np.isnan(v).any():
                return
            # find magnitude of vector
            mag = norm(v)
            # unit vector in direction of axis
            v = v / mag
            # make some vector not in the same direction as v
            not_v = np.array([1, 0, 0])
            if (v == not_v).all():
                not_v = np.array([0, 1, 0])
            # make vector perpendicular to v
            n1 = np.cross(v, not_v)
            # normalize n1
            n1 /= norm(n1)
            # make unit vector perpendicular to v and n1
            n2 = np.cross(v, n1)
            # surface ranges over t from 0 to length of axis and 0 to 2*pi
            t = np.linspace(0, mag, 100)
            theta = np.linspace(0, 2 * np.pi, 100)
            # use meshgrid to make 2d arrays
            t, theta = np.meshgrid(t, theta)
            # generate coordinates for surface
            X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
            # Draw parameters
            rstride = 20
            cstride = 10
            ax.plot_surface(X, Y, Z, alpha=alpha, color=c)
        except:
            print('%' * 60)
            print('Unable to draw the cylinder around the cluster points.')
            print('%' * 60)
            pass

    @classmethod
    def generate_3_confidence_ellipse2d(cls, x, y, ax, facecolor='none', **kwargs):
        cls.generate_confidence_ellipse2d(x, y, ax, 1.0, facecolor, edgecolor='firebrick')
        cls.generate_confidence_ellipse2d(x, y, ax, 2.0, facecolor, edgecolor='fuchsia', linestyle='--')
        ell = cls.generate_confidence_ellipse2d(x, y, ax, 3.0, facecolor, edgecolor='blue', linestyle=':')
        return ell

    @classmethod
    def generate_confidence_ellipse2d(cls, x, y, ax, n_std=3.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of `x` and `y`

        Parameters
        ----------
        x, y : array_like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        Returns
        -------
        matplotlib.patches.Ellipse

        Other parameters
        ----------------
        kwargs : `~matplotlib.patches.Patch` properties
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
        if x.size <= 0:
            return None

        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2,
                          height=ell_radius_y * 2,
                          facecolor=facecolor,
                          **kwargs)

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    @classmethod
    def plot_cluster_as_ellipse_rings(cls, ax, xdata, ydata, zdata, alpha=.6, c='blue',
                                      add_to_upper_z_limit=1,
                                      add_to_lower_z_limit=-1, step=2):
        import math
        z_min = math.floor(np.min(zdata))
        z_max = math.ceil(np.max(zdata))
        for i in np.arange(z_min, z_max, step):
            cls.generate_confidence_ellipse3d(xdata[np.logical_and(zdata >= i, zdata < i + 1)],
                                              ydata[np.logical_and(zdata >= i, zdata < i + 1)],
                                              zdata[np.logical_and(zdata >= i, zdata < i + 1)],
                                              ax, facecolor=c)

    pass

    @classmethod
    def generate_confidence_ellipse3d(cls, xdata, ydata, zdata, ax_3d, n_std=3.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of `x` and `y`

        Parameters
        ----------
        xdata, ydata, zdata : array_like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        Returns
        -------
        matplotlib.patches.Ellipse

        Other parameters
        ----------------
        kwargs : `~matplotlib.patches.Patch` properties
        """
        if xdata.size != ydata.size or xdata.size != zdata.size or ydata.size != zdata.size:
            raise ValueError("x and y must be the same size")

        if xdata.size <= 0:
            return

        cov = np.cov(xdata, ydata)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(xdata)

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(ydata)

        ellipse = Ellipse((mean_x, mean_y),
                          angle=45,
                          width=ell_radius_x * 2 * scale_x,
                          height=ell_radius_y * 2 * scale_y,
                          facecolor=facecolor,
                          fill=False,
                          **kwargs)

        ax_3d.add_patch(ellipse)
        art3d.pathpatch_2d_to_3d(ellipse, z=np.mean(zdata), zdir="z")

        # p = Circle((mean_x, mean_y), 10)
        # ax_3d.add_patch(p)
        # art3d.pathpatch_2d_to_3d(p, z=np.mean(zdata), zdir="z")

        pass
