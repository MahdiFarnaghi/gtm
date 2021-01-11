import os
from datetime import datetime
from pathlib import Path
from pprint import pprint

import sys
from sklearn.cluster import DBSCAN
import geopandas as gp
import pandas as pd
import numpy as np
# from sklearn.cluster.optics_ import OPTICS

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from gttm.nlp.vectorize import VectorizerUtil
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import euclidean_distances, silhouette_score, pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from gttm.mathematics import math_func


class ClusterHandler:

    def __init__(self, alpha, beta, gama, temporal_extent: float, spatial_extent: float, min_cluster_size=10,
                 dbscan_eps=4.5, metric='default', min_textual_distance=0.5):
        self.labels = None
        self.label_codes = None
        self.space = None
        self.reachability = None
        self.silhouette_coefficient = None
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.temporal_extent = temporal_extent
        self.temporal_extent_boundary = 3 * 60 * 60  # seconds
        self.spatial_extent = spatial_extent
        self.spatial_extent_divide_by_factor = 20
        self.spatial_extent_boundary = 40000  # meters based on kNN graph
        self.min_cluster_size = min_cluster_size
        self.dbscan_eps = dbscan_eps
        self.min_textual_distance = min_textual_distance
        # self.defaut_metric = self.metric_xyt_euc_times_c_cos_boundary_on_c
        self.defaut_metric = self.metric_07_weighted_sum_xy_euc_t_euc_c_cos_normalized_with_extents_boundary_on_c
        if metric != 'default':
            if metric == '01':
                self.defaut_metric = self.metric_01_c_cos
            if metric == '02':
                self.defaut_metric = self.metric_02_xy_euc
            if metric == '03':
                self.defaut_metric = self.metric_03_t_euc
            if metric == '04':
                self.defaut_metric = self.metric_04_norm1_xy_euc_t_euc_c_cos
            if metric == '05':
                self.defaut_metric = self.metric_05_weighted_sum_xy_euc_t_euc_c_cos
            if metric == '06':
                self.defaut_metric = self.metric_06_weighted_sum_xy_euc_t_euc_c_cos_normalized_with_extents
            if metric == '07':
                self.defaut_metric = self.metric_07_weighted_sum_xy_euc_t_euc_c_cos_normalized_with_extents_boundary_on_c
            if metric == '08':
                self.defaut_metric = self.metric_08_xyt_euc_times_c_cos
            if metric == '09':
                self.defaut_metric = self.metric_09_xyt_euc_times_c_cos_boundary_on_c

    def clear(self):
        self.labels = None
        self.label_codes = None
        self.space = None
        self.reachability = None
        self.silhouette_coefficient = None

    def cluster_optics(self, gdf: gp.GeoDataFrame, vect_method, lang, verbose=False):
        """

        :type gdf: object
        """
        if verbose:
            print('\tStart cluster_detection (optics)...')
        s_time = datetime.now()

        self.space = None
        self.reachability = None

        xytc = self.generate_xy_projected_t_c_matrix(gdf, vect_method, lang)

        # metric = self.weighted_xy_euc_t_euc_c_cos_with_boundaries
        clust = OPTICS(metric=self.defaut_metric, min_cluster_size=self.min_cluster_size)

        clust.fit(xytc)

        self.labels = clust.labels_
        # labels_ordered = clust.labels_[clust.ordering_]
        self.label_codes = np.unique(self.labels)
        if verbose:
            print('\t\tNumber of records: {}'.format(len(xytc)))
            print('\t\tCluster labels: {}'.format(str(self.label_codes)))
        self.space = np.arange(len(ClusterHandler.generate_xy_matrix(gdf)))
        self.reachability = clust.reachability_[clust.ordering_]

        self.silhouette_coefficient = ClusteringQuality.silhouette_coefficient(xytc, self.labels, self.label_codes,
                                                                               self.defaut_metric)

        dur = datetime.now() - s_time
        if verbose:
            print('\tClustering was finished ({} seconds).'.format(dur.seconds))
        return self.labels, self.label_codes, self.silhouette_coefficient

    def cluster_dbscan(self, gdf: gp.GeoDataFrame, vect_method, lang, verbose=False):
        if verbose:
            print('\tStart cluster_detection (dbscan)...')
        s_time = datetime.now()

        xytc = self.generate_xy_projected_t_c_matrix(gdf, vect_method, lang)

        clust = DBSCAN(metric=self.defaut_metric, eps=self.dbscan_eps, min_samples=self.min_cluster_size)
        clust.fit(xytc)

        self.labels = clust.labels_
        self.label_codes = np.unique(self.labels)
        self.silhouette_coefficient = ClusteringQuality.silhouette_coefficient(xytc, self.labels, self.label_codes,
                                                                               self.defaut_metric)

        dur = datetime.now() - s_time
        if verbose:
            print('\tClustering was finished ({} seconds).'.format(dur.seconds))
        return self.labels, self.label_codes, self.silhouette_coefficient

    def generate_kNN_plot(self, gdf: gp.GeoDataFrame, vect_method, lang, file_path):
        print('\tGenerating kNN plot started (at {}) ...'.format(datetime.now().strftime("%Y%m%d-%H%M")))
        print('vect_method: {}, metric: {}'.format(vect_method, str(self.defaut_metric)))
        s_time = datetime.now()
        xytc = self.generate_xy_projected_t_c_matrix(gdf, vect_method, lang)

        nbrs = NearestNeighbors(n_neighbors=self.min_cluster_size, metric=self.defaut_metric).fit(xytc)
        distances, indices = nbrs.kneighbors(xytc)

        def get_col(arr, col):
            return map(lambda x: x[col], arr)

        dist_col = list(get_col(distances, self.min_cluster_size - 1))
        dist_col_sorted = sorted(dist_col)
        num_of_points = len(dist_col_sorted)
        import matplotlib.pyplot as plt
        plt.plot(list(range(1, num_of_points + 1)), dist_col_sorted)
        plt.ylabel('Distance (spatial, temporal and textual)')
        plt.savefig(file_path, dpi=300)
        plt.close()
        dur = datetime.now() - s_time
        print('\tGenerating kNN plot finished ({} seconds).'.format(dur.seconds))
        pass

    @staticmethod
    def generate_xy_projected_t_c_matrix(gdf, vect_method, lang):
        x = np.asarray(gdf.geometry.x)[:, np.newaxis]
        y = np.asarray(gdf.geometry.y)[:, np.newaxis]

        t = np.asarray(gdf[['t']])

        c = np.asarray(gdf['c'])
        c_vect_dense = None

        if vect_method == 'tfidf':
            c_vect_dense = VectorizerUtil.vectorize_tfidf(c, lang)
        elif vect_method == 'bow':
            c_vect_dense = VectorizerUtil.vectorize_count(c, lang)
        elif vect_method == 'w2v':
            c_vect_dense = VectorizerUtil.vectorize_word2vec(c, lang)
        elif vect_method == 'fasttext':
            c_vect_dense = VectorizerUtil.vectorize_fasttext(c, lang)
        elif vect_method == 'glove':
            c_vect_dense = VectorizerUtil.vectorize_glove(c, lang)

        xytc = np.concatenate((x, y, t, c_vect_dense), axis=1)  # , c[:, np.newaxis]
        return xytc

    @staticmethod
    def metric_01_c_cos(a, b):
        try:
            c_distance = np.absolute(cosine_distances(a[np.newaxis, 3:], b[np.newaxis, 3:]))
            return c_distance
        except Exception as ex:
            print(ex)
            return sys.maxsize

    @staticmethod
    def metric_02_xy_euc(a, b):
        try:
            xy_distance = euclidean_distances(a[np.newaxis, 0:2], b[np.newaxis, 0:2])
            return xy_distance
        except Exception as ex:
            print(ex)
            return sys.maxsize

    @staticmethod
    def metric_03_t_euc(a, b):
        try:
            t_distance = euclidean_distances(a[np.newaxis, 2:3], b[np.newaxis, 2:3])
            return t_distance
        except Exception as ex:
            print(ex)
            return sys.maxsize

    def metric_04_norm1_xy_euc_t_euc_c_cos(self, a, b):
        xy_distance = euclidean_distances(a[np.newaxis, 0:2], b[np.newaxis, 0:2])
        t_distance = euclidean_distances(a[np.newaxis, 2:3], b[np.newaxis, 2:3])
        c_distance = np.absolute(cosine_distances(a[np.newaxis, 3:4], b[np.newaxis, 3:4]))
        dist = self.alpha * xy_distance + self.beta * t_distance + self.gama * c_distance

        return dist

    def metric_05_weighted_sum_xy_euc_t_euc_c_cos(self, a, b):
        try:
            xy_distance = euclidean_distances(a[np.newaxis, 0:2], b[np.newaxis, 0:2])
            xy_distance_norm = math_func.linear(xy_distance, 3000)
            t_distance = euclidean_distances(a[np.newaxis, 2:3], b[np.newaxis, 2:3])
            t_distance_norm = math_func.linear(t_distance, 8 * 60 * 60)  # x0 in second
            c_distance = np.absolute(cosine_distances(a[np.newaxis, 3:], b[np.newaxis, 3:]))
            dist = self.alpha * xy_distance_norm + self.beta * t_distance_norm + self.gama * c_distance
            return dist
        except Exception as ex:
            print(ex)
            return sys.maxsize

    def metric_06_weighted_sum_xy_euc_t_euc_c_cos_normalized_with_extents(self, a, b):
        try:
            xy_distance = euclidean_distances(a[np.newaxis, 0:2], b[np.newaxis, 0:2])
            xy_distance_norm = math_func.linear(xy_distance, self.spatial_extent / self.spatial_extent_divide_by_factor)

            t_distance = euclidean_distances(a[np.newaxis, 2:3], b[np.newaxis, 2:3])
            t_distance_norm = math_func.linear(t_distance, self.temporal_extent * 60 * 60)  # x0 in second

            c_distance = np.absolute(cosine_distances(a[np.newaxis, 3:], b[np.newaxis, 3:]))

            return self.alpha * xy_distance_norm + self.beta * t_distance_norm + self.gama * c_distance

        except Exception as ex:
            print(ex)
            return sys.maxsize

    def metric_07_weighted_sum_xy_euc_t_euc_c_cos_normalized_with_extents_boundary_on_c(self, a, b):
        try:
            xy_distance = euclidean_distances(a[np.newaxis, 0:2], b[np.newaxis, 0:2])
            xy_distance_norm = math_func.linear(xy_distance, self.spatial_extent / self.spatial_extent_divide_by_factor)

            t_distance = euclidean_distances(a[np.newaxis, 2:3], b[np.newaxis, 2:3])
            t_distance_norm = math_func.linear(t_distance, self.temporal_extent * 60 * 60)  # x0 in second

            c_distance = np.absolute(cosine_distances(a[np.newaxis, 3:], b[np.newaxis, 3:]))
            if c_distance > self.min_textual_distance:
                return sys.maxsize

            return self.alpha * xy_distance_norm + self.beta * t_distance_norm + self.gama * c_distance

        except Exception as ex:
            print(ex)
            return sys.maxsize

    @staticmethod
    def metric_08_xyt_euc_times_c_cos(a, b):
        try:
            xyt_distance = euclidean_distances(a[np.newaxis, 0:3], b[np.newaxis, 0:3])

            c_distance = np.absolute(cosine_distances(a[np.newaxis, 3:], b[np.newaxis, 3:]))

            return xyt_distance * c_distance
        except Exception as ex:
            print(ex)
            return sys.maxsize

    def metric_09_xyt_euc_times_c_cos_boundary_on_c(self, a, b):
        try:
            xyt_distance = euclidean_distances(a[np.newaxis, 0:3], b[np.newaxis, 0:3])

            c_distance = np.absolute(cosine_distances(a[np.newaxis, 3:], b[np.newaxis, 3:]))
            if c_distance > self.min_textual_distance:
                return sys.maxsize

            return xyt_distance * c_distance
        except Exception as ex:
            print(ex)
            return sys.maxsize

    @staticmethod
    def calculate_distance(gdf, norm):
        xy = np.asarray(
            gdf[['x', 'y']] * 10000)  # pd.merge(gdf[geom_col].x, gdf[geom_col].y, left_index=True, right_index=True)
        spatial_distance = euclidean_distances(xy)
        norm_spatial_distance = preprocessing.normalize(spatial_distance, norm=norm)
        t = np.asarray(gdf[['t']])
        temporal_distance = euclidean_distances(t)
        norm_temporal_distance = preprocessing.normalize(temporal_distance, norm=norm)
        c = np.asarray(gdf['c'])
        vectorizer = TfidfVectorizer()
        c_vect = vectorizer.fit_transform(c)
        content_distance = np.absolute(cosine_distances(c_vect))
        norm_content_distance = preprocessing.normalize(content_distance, norm=norm)
        distances = alpha * norm_spatial_distance + beta * norm_content_distance + gama * norm_temporal_distance
        return distances

    @staticmethod
    def link_clusters(gdf_new, gdf_new_label, gdf_old, global_lable_codes, linking_coef=.8, verbose=False):
        if verbose:
            print('\tStart linking clusters ...')
        s_time = datetime.now()

        gdf_new['l'] = gdf_new_label
        if gdf_old is None:
            gdf_new['label'] = gdf_new_label
            [global_lable_codes.append(x) for x in np.unique(gdf_new_label)]
            changed_labels = gdf_new_label
        else:
            # gdf_new['label'] = -1
            changed_labels = - np.ones(gdf_new_label.shape)
            # np.intersect1d(n, o).count()
            # print("\t\tNumber of common tweets: {}".format(len(np.intersect1d(gdf_new.id.values, gdf_old.id.values))))
            inter_over_union = len(np.intersect1d(gdf_new.id.values, gdf_old.id.values)) / len(
                np.union1d(gdf_new.id.values, gdf_old.id.values))
            # print("\t\tNumber of common tweets over all tweets: {}".format(inter_over_union))
            # print("\t\told cluster code: {}".format(gdf_new[gdf_new.l >= 0].l.unique()))
            # print("\t\tnew cluster code: {}".format(gdf_old[gdf_old.l >= 0].l.unique()))
            # np.intersect1d(gdf_new[gdf_new.l == n], gdf_old[gdf_old.l == o]).count()
            for n in gdf_new[gdf_new.l >= 0].l.unique():
                max_rel_strength = 0
                max_rel_strength_label = -1
                for o in gdf_old[gdf_old.label >= 0].label.unique():
                    rel_strength = len(
                        np.intersect1d(gdf_new[gdf_new.l == n].id.values, gdf_old[gdf_old.label == o].id.values)) / len(
                        np.union1d(gdf_new[gdf_new.l == n].id.values, gdf_old[gdf_old.label == o].id.values))
                    if rel_strength > max_rel_strength:
                        max_rel_strength = rel_strength
                        max_rel_strength_label = o
                if max_rel_strength > (inter_over_union * linking_coef):
                    # gdf_new[gdf_new.l == n].label = o
                    changed_labels[gdf_new_label == n] = max_rel_strength_label
                else:
                    new_label = np.max(global_lable_codes) + 1
                    # gdf_new[gdf_new.l == n].label = new_label
                    changed_labels[gdf_new_label == n] = new_label
                    global_lable_codes.append(new_label)

            gdf_new['label'] = changed_labels

            # intersect = [
            #     [[n, o, len(np.intersect1d(gdf_new[gdf_new.l == n].id.values, gdf_old[gdf_old.l == o].id.values)) / len(
            #         np.union1d(gdf_new[gdf_new.l == n].id.values, gdf_old[gdf_old.l == o].id.values))] for o in
            #      gdf_old[gdf_old.l >= 0].l.unique()] for n in gdf_new[gdf_new.l >= 0].l.unique()]
            # for i_n in range(len(intersect)):
            #     idx = np.argmax(intersect[i_n])
            #     if intersect[i_n][idx] >= inter_over_union * linking_coef:
            #         gdf_new[]
            # pprint(intersect)
        changed_label_code = np.unique(changed_labels)

        dur = datetime.now() - s_time
        if verbose:
            print('\tLinking clusters was finished ({} seconds).'.format(dur.seconds))
        return changed_labels, changed_label_code

    @staticmethod
    def generate_xy_matrix(gdf):
        xy = np.asarray(
            gdf[['x', 'y']] * 10000)  # pd.merge(gdf[geom_col].x, gdf[geom_col].y, left_index=True, right_index=True)
        return xy

    @staticmethod
    def generate_xytc_matrix(gdf):
        xy = np.asarray(
            gdf[['x', 'y']] * 10000)  # pd.merge(gdf[geom_col].x, gdf[geom_col].y, left_index=True, right_index=True)
        t = np.asarray(gdf[['t']])
        c = np.asarray(gdf['c'])
        vectorizer = TfidfVectorizer()
        c_vect = vectorizer.fit_transform(c)
        c_vect_dense = np.asarray(c_vect.todense())
        xytc = np.concatenate((xy, t, c_vect_dense), axis=1)
        return xytc

    @staticmethod
    def generate_xytc_0_1_scaled_matrix(gdf, norm='l2'):
        scaler = MinMaxScaler()
        x = np.asarray(gdf[['x']])
        x_norm = scaler.fit(x).transform(x)

        y = np.asarray(gdf[['y']])
        y_norm = scaler.fit(y).transform(y)

        t = np.asarray(gdf[['t']])
        t_norm = scaler.fit(t).transform(t)

        c = np.asarray(gdf['c'])
        vectorizer = TfidfVectorizer()
        c_vect = vectorizer.fit_transform(c)
        c_vect_dense = np.asarray(c_vect.todense())
        xytc = np.concatenate((x_norm, y_norm, t_norm, c_vect_dense), axis=1)
        return xytc


class ClusteringQuality:
    def __init__(self):
        pass

    @classmethod
    def silhouette_coefficient(cls, xytc, labels, label_codes, metric):
        if len(label_codes[label_codes != -1]) <= 1:
            print('$' * 60)
            print('Unable to calculate silhouette coefficient. Cluster codes: {}'.format(str(label_codes)))
            print('$' * 60)
            return sys.maxsize, sys.maxsize, sys.maxsize, sys.maxsize
        else:
            dist = pairwise_distances(xytc[labels != -1], metric=metric)
            sillhouete_dist = silhouette_score(xytc[:, 0:2][labels != -1], labels[labels != -1], metric='euclidean')
            sillhouete_time = silhouette_score(xytc[:, 3:4][labels != -1], labels[labels != -1], metric='euclidean')
            sillhouete_content = silhouette_score(xytc[:, 4:][labels != -1], labels[labels != -1], metric='cosine')
            sillhouete_overall = silhouette_score(dist, labels[labels != -1], metric='precomputed')
            # print('\tSilhouette score: {}'.format(res))
            return sillhouete_dist, sillhouete_time, sillhouete_content, sillhouete_overall
