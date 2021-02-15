# -*- coding: UTF8 -*-
import getopt
import os
import sys
import traceback
import warnings
from logging import warning

from gttm.ioie import reader
from gttm.ioie.writer import ResultWritter
from gttm.nlp.identify_topic import HDPTopicIdentification
from gttm.nlp.vectorize import VectorizerUtil

# os.environ["PROJ_LIB"] = os.path.dirname(sys.executable) + os.sep + "Library\\share"
from gttm.cluster_detection.plotting import Plotter
from gttm.cluster_detection.cluster_handler import ClusterHandler
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from gttm.ioie.geodata import add_geometry
from gttm.ioie.reader import read_data_from_postgres, read_json_files_insert_to_postgres
import geopandas as gp

alpha = 0.3
beta = 0.2
gamma = 0.5


def execute(lang: str, clust_method: str, alpha, beta, gamma, metric, min_textual_distance,
            vect_method: str, study_area_name,
            table_name, start_date, duration, steps, end_date, analysis_id,
            analysis_output_folder, plot_results, min_cluster_size=None, dbscan_eps=None,
            filter_list_for_plotting=None,
            study_area_bbox=None,
            verbose=False, multiprocess=True, iternation_num=None):
    if not Path(analysis_output_folder).exists():
        os.makedirs(analysis_output_folder)

    write_execution_setting(lang, clust_method, alpha, beta, gamma, metric, min_textual_distance, vect_method,
                            study_area_name,
                            study_area_bbox, table_name,
                            start_date, duration, steps, end_date, analysis_id,
                            min_cluster_size, dbscan_eps,
                            analysis_output_folder, plot_results, verbose)

    current_date = start_date

    iter_num = 1
    iterations_info = []
    clusters_info = None
    cluster_labels = {}
    print("=" * 60)
    print("=" * 60)
    print("Clustering of {} tweets".format(study_area_name))
    print('Vectorization method: {}, Clustering method: {}'.format(
        vect_method, clust_method))
    print("=" * 60)
    print("=" * 60)
    prev_iter_gdf = None
    prev_label_codes = []

    VectorizerUtil.init_models([vect_method])

    while current_date + timedelta(hours=duration) <= end_date:
        if iternation_num is None or iternation_num == iter_num:
            iter_id, iter_info, iter_labels, iter_clusters_info, iter_gdf = execute_single_step(
                iter_num,
                prev_iter_gdf,
                prev_label_codes,
                lang,
                clust_method,
                alpha,
                beta,
                gamma,
                metric,
                min_textual_distance,
                vect_method,
                table_name,
                current_date,
                duration,
                analysis_output_folder,
                plot_results,
                min_cluster_size=min_cluster_size,
                dbscan_eps=dbscan_eps,
                filter_list_for_plotting=filter_list_for_plotting,
                study_area_bbox=study_area_bbox,
                multiprocess=multiprocess,
                verbose=False)

            if iter_id is not None:
                if iter_gdf is not None:
                    prev_iter_gdf = iter_gdf

                cluster_labels[iter_id] = iter_labels
                clusters_info = iter_clusters_info if clusters_info is None else \
                    clusters_info.append(iter_clusters_info, ignore_index=True)
                iterations_info.append(iter_info)
        iter_num += 1
        current_date = current_date + timedelta(hours=steps)

    print('=' * 60)
    print('Finalizing the analysis')
    # Saving the iteration information
    print('Saving the iterations information')
    iterations_info_path = analysis_output_folder + \
        os.sep + 'iterations_information.csv'
    ResultWritter.write_list_to_file(iterations_info, iterations_info_path,
                                     header=['iter_id', 'iter_num', 'start_date', 'end_date', 'clust_method',
                                             'lang', 'num_of_tweets', 'num_of_clusters', 'silhouette_dist',
                                             'silhouette_time', 'silhouette_content', 'silhouette_overall',
                                             'execution time (seconds)'])

    # Saving the cluster labels
    print('Saving the cluster labels')
    cluster_labels_path = analysis_output_folder + os.sep + 'cluster_labels.csv'
    ResultWritter.write_cluster_labels_to_file(
        cluster_labels, cluster_labels_path)

    # Saving the cluster topics
    print('Saving the clusters information')
    clusters_info_path = analysis_output_folder + os.sep + 'cluster_information.csv'
    clusters_info.to_csv(clusters_info_path, sep=',', encoding='utf-8')

    # Saving the cluster labels in a diagonal file
    print('Saving the cluster labels in a diagonal file')
    cluster_labels_path_diag = analysis_output_folder + \
        os.sep + 'cluster_labels_diag.csv'
    ResultWritter.write_cluster_labels_to_file_diagnal(
        cluster_labels, cluster_labels_path_diag)

    print('%' * 60)
    print('Analysis finished successfully')
    print('Check the output folder: {}'.format(analysis_output_folder))
    print('%' * 60)

    pass


def execute_single_step(iter_num, prev_iter_gdf, prev_lable_codes, lang, clust_method, alpha, beta, gama, metric,
                        min_textual_distance,
                        vect_method,
                        table_name, start_date, duration, output_folder, plot_results,
                        min_cluster_size=None, dbscan_eps=None, filter_list_for_plotting=None, study_area_bbox=None,
                        verbose=False, multiprocess=True):
    if not Path(output_folder).exists():
        os.makedirs(output_folder)

    current_date = start_date

    iter_info = []
    iter_gdf = None
    l = None
    VectorizerUtil.init_models([vect_method])

    s_time = datetime.now()
    iter_id = "{}-{}-{}-{}".format(current_date.strftime("%Y%m%d-%H%M"), lang, clust_method,
                                   vect_method)
    iter_folder = output_folder + os.sep + iter_id

    iter_info.append(iter_id)
    iter_info.append(iter_num)
    iter_info.append(current_date.strftime("%Y%m%d-%H%M"))
    iter_info.append(
        (current_date + timedelta(hours=duration)).strftime("%Y%m%d-%H%M"))
    iter_info.append(clust_method)
    iter_info.append(lang)

    if verbose:
        print("-" * 60)
    print(
        "Iteration number: {}, from {} to {}".format(iter_num,
                                                     current_date.strftime("%Y-%m-%d %H:%M"), (
                                                         current_date + timedelta(
                                                             hours=duration)).strftime(
                                                         "%Y-%m-%d %H:%M")))
    if verbose:
        print("Iteration Id: {}".format(iter_id))
        print("Clustering method: {}".format(clust_method))
        print("Vectorizing method: {}".format(vect_method))
        print("Language: {}".format(lang))
        print('Process started at: {}'.format(
            s_time.strftime("%Y:%m:%d-%H:%M")))

    # read the data
    print("1. read the data {}".format(iter_num))
    if iter_gdf is not None:
        prev_iter_gdf = iter_gdf
    # iter_df, num = read_data_from_files(start_date=current_date,
    #                                     end_date=current_date + timedelta(hours=duration),
    #                                     folder=data_path)

    iter_df, num = read_data_from_postgres(start_date=current_date,
                                           end_date=current_date +
                                           timedelta(hours=duration),
                                           table_name=table_name)
    if num <= 0:
        print('There was no record for processing.')
        return None, None, None, None, None

    # convert to geodataframe
    print("2. convert to GeoDataFrame {}".format(iter_num))
    iter_gdf = add_geometry(iter_df[iter_df.lang == lang].copy(),
                            study_area=study_area_bbox) if lang != 'multi' else add_geometry(
        iter_df.copy(), study_area=study_area_bbox)
    num_of_tweets = iter_gdf.shape[0]
    iter_info.append(num_of_tweets)
    if num_of_tweets <= 0:
        print('There was no record for processing.')
        return None, None, None, None, None
    if verbose:
        print("\tNumber of tweets: {}".format(num_of_tweets))

    # cluster_detection
    print("3. Clustering {}".format(iter_num))
    cluster_handler = ClusterHandler(alpha, beta, gama, temporal_extent=duration,
                                     spatial_extent=get_spatial_extent(iter_gdf), min_cluster_size=min_cluster_size,
                                     dbscan_eps=dbscan_eps, metric=metric, min_textual_distance=min_textual_distance) \
        if min_cluster_size is not None and dbscan_eps is not None else ClusterHandler(
        alpha, beta, gama, temporal_extent=duration,
        spatial_extent=get_spatial_extent(iter_gdf), metric=metric, min_textual_distance=min_textual_distance)
    cluster_handler.clear()
    if clust_method == 'dbscan':
        l, l_codes, silh_coefs = cluster_handler.cluster_dbscan(iter_gdf, vect_method,
                                                                lang)
    elif clust_method == 'optics':
        l, l_codes, silh_coefs = cluster_handler.cluster_optics(iter_gdf, vect_method,
                                                                lang)
    elif clust_method == 'hdbscan':
        # todo: Add hdbscan method
        pass
    num_of_clusters = len(l_codes[l_codes >= 0])
    iter_info.append(num_of_clusters)
    [iter_info.append(sil) for sil in silh_coefs]
    iter_gdf['l'] = l

    # linking clusters
    print("4. Link clusters {}".format(iter_num))
    iter_gdf_label, iter_gdf_label_code = ClusterHandler.link_clusters(iter_gdf, l, prev_iter_gdf,
                                                                       prev_lable_codes)
    if verbose:
        print("\tClusters information - Code (linked): Number of tweets:")
        label_codes = np.unique(iter_gdf.label.values)
        [print("\t\t{}: {}".format(code, iter_gdf[iter_gdf.label == code].shape[0])) for code in
         label_codes]

    # topic identification
    print("5. Identify topics {}".format(iter_num))
    # identTopic = LDATopicIndentifier()
    identTopic = HDPTopicIdentification()
    identTopic.identify_topics(iter_gdf)
    if verbose:
        identTopic.print_cluster_topics()
    iter_clust_topics = identTopic.get_cluster_topics()

    # plotting the results
    print("6. Plot results {}".format(iter_num))
    if plot_results:
        iter_gdf_label_code_for_plotting = iter_gdf_label_code
        if filter_list_for_plotting is not None:
            iter_gdf_label_code_for_plotting = []
            for topic in iter_clust_topics:
                if len(list(set(topic[4].split()) & set(filter_list_for_plotting))) > 0 and topic[0] > -1:
                    iter_gdf_label_code_for_plotting.append(topic[0])

        Plotter.generate_3d_chart_map_wordcloud_(iter_num,
                                                 iter_gdf,
                                                 label=iter_gdf_label,
                                                 label_codes=iter_gdf_label_code_for_plotting,
                                                 study_area=study_area_bbox,
                                                 show_plot=False, save_plot=True,
                                                 plot_title=iter_id,
                                                 plot_folder_path=iter_folder,
                                                 plot_file_name_prefix=iter_id,
                                                 one_plot_per_cluster=True,
                                                 multiprocess=multiprocess)
        identTopic.save_wordcloud_of_cluster_topics(iter_folder,
                                                    iter_id,
                                                    label_code_for_plotting=iter_gdf_label_code_for_plotting,
                                                    multiprocess=multiprocess)

    # todo: find important clusters
    # print('6. find important clusters')
    #
    # todo: cluster localization
    # print('7. cluster localization')
    #
    # todo: draw bounding circle or cylinder around the points of a cluster
    # http://kylebarbary.com/nestle/examples/plot_ellipsoids.html
    # https://stackoverflow.com/questions/26989131/add-cylinder-to-plot
    # https: // www.nayuki.ioie / res / smallest - enclosing - circle / smallestenclosingcircle.py

    # save clusters
    print("7. Save results {}".format(iter_num))
    file_path = iter_folder + os.sep + "labeled_tweets_{}.csv".format(iter_id)
    save_labeled_tweets(iter_gdf, file_path)
    file_path = iter_folder + os.sep + \
        "clusters_information_{}.csv".format(iter_id)
    iter_clusters_info = compose_clustering_result(clust_method, current_date, duration, identTopic, iter_clust_topics,
                                                   iter_folder, iter_gdf, iter_id, iter_num, lang, save=True,
                                                   file_path=file_path)

    dur = datetime.now() - s_time
    iter_info.append(dur.seconds)
    print('Iteration {} was finished at {} ({} minutes).'.format(iter_id, datetime.now().strftime("%Y:%m:%d-%H:%M"),
                                                                 int(dur.seconds / 60)))
    print('Iteration output folder: {}'.format(iter_folder))

    if verbose:
        print('=' * 60)

    return iter_id, iter_info, iter_gdf_label_code, iter_clusters_info, iter_gdf
    pass


def save_labeled_tweets(gdf: gp.GeoDataFrame, file_path, labels=None):
    print('\tStart saving labeled tweets ...')
    s_time = datetime.now()

    if file_path != "":
        if not Path(file_path).parent.exists():
            os.makedirs(Path(file_path).parent.absolute())
        if labels is not None:
            data = gdf[['id', 'userid', 'x', 'y',
                        't_datetime', 't', 'c']].copy()
            data['label'] = labels
        else:
            data = gdf[['id', 'userid', 'x', 'y',
                        't_datetime', 't', 'c', 'label']].copy()

        data.to_csv(file_path, sep=',', encoding='utf-8')

    dur = datetime.now() - s_time
    print('\tSaving clusters was finished ({} seconds).'.format(dur.seconds))
    pass


def get_spatial_extent(iter_gdf: gp.GeoDataFrame):
    b = iter_gdf.geometry.bounds.agg(
        {'minx': np.min, 'maxx': np.max, 'miny': np.min, 'maxy': np.max})
    return min(abs(b['minx'] - b['maxx']), abs(b['miny'] - b['maxy']))
    pass


def compose_clustering_result(clust_method, current_date, duration, identTopic, iter_clust_topics, iter_folder,
                              iter_gdf, iter_id, iter_num, lang, save=False, file_path=''):
    print('\tStart composing clusters information ...')
    s_time = datetime.now()

    cluster_groups = iter_gdf[iter_gdf.label > -1].groupby('label')
    clusters_info = cluster_groups.agg(
        {'x': {'x_min': np.min, 'x_max': np.max, 'x_center': np.mean},
         'y': {'y_min': np.min, 'y_max': np.max, 'y_center': np.mean}, 'id': {'num_tweets': np.size}})
    clusters_info.columns = ['x_min', 'x_max', 'x_center',
                             'y_min', 'y_max', 'y_center', 'num_tweets']
    clusters_info = pd.DataFrame(clusters_info.to_records())
    clusters_info['iteration_num'] = iter_num
    clusters_info['iteration_id'] = iter_id
    clusters_info['start_date'] = current_date.strftime("%Y%m%d-%H%M")
    clusters_info['end_date'] = (
        current_date + timedelta(hours=duration)).strftime("%Y%m%d-%H%M")
    clusters_info['clustering_method'] = clust_method
    clusters_info['vectorization_method'] = clust_method
    clusters_info['lang'] = lang
    clust_topics_df = pd.DataFrame.from_records(iter_clust_topics,
                                                columns=['label', 'topic_code', 'topic_relevance_percentage',
                                                         'topic', 'topic_words'])
    clusters_info = clusters_info.merge(clust_topics_df, on='label')
    # identTopic.save_cluster_topics(iter_folder, iter_id)
    if save and file_path != "":
        if not Path(file_path).parent.exists():
            os.makedirs(Path(file_path).parent.absolute())
        clusters_info.to_csv(file_path, sep=',', encoding='utf-8')

    dur = datetime.now() - s_time
    print('\tComposing clusters information was finished ({} seconds).'.format(dur.seconds))
    return clusters_info


def write_execution_setting(lang, clustering_method, alpha, beta, gama, metric, min_textual_distance,
                            vectorizing_method, study_area_name,
                            study_area_bbox,
                            table_name, start_date, duration, steps, end_date, analysis_id, min_cluster_size,
                            dbscan_eps,
                            analysis_output_folder, plot_results, verbose=False):
    exec_setting_file = analysis_output_folder + os.sep + 'execution_setting.txt'
    lines = ['Languages: {}'.format(lang),
             'Clustering methods: {}'.format(clustering_method),
             'Alph - distance multiplier: {}'.format(alpha),
             'Beta - time multiplier: {}'.format(beta),
             'Gama - text multiplier: {}'.format(gama),
             'Metric: {}'.format(str(metric)),
             'Minmum textual distance: {}'.format(str(min_textual_distance)),
             'Min cluster size: {}'.format(
                 min_cluster_size if min_cluster_size is not None else ''),
             'DBSCAN Eps: {}'.format(
                 dbscan_eps if min_cluster_size is not None else ''),
             'Vectorization method: {}'.format(vectorizing_method),
             'Study area: {} - {}'.format(study_area_name,
                                          study_area_bbox if study_area_bbox is not None else 'Not specified'),
             'Table name in the database: {}'.format(table_name),
             'Start date: {}'.format(start_date),
             'End date: {}'.format(end_date),
             'Time windows length: {} (hours)'.format(duration),
             'Temporal steps: {}'.format(steps),
             'Analysis Id: {}'.format(analysis_id),
             'Output folder: {}'.format(analysis_output_folder),
             'Plot results: {}'.format(plot_results),
             'Verbose: {}'.format(verbose)]
    lines = list(map(lambda x: str(x) + '\n', lines))
    with open(exec_setting_file, 'w+') as of:
        of.writelines(lines)
    pass


def import_data_from_textfile(file_path, import_to_table_name):
    reader.create_table(import_to_table_name)
    reader.read_csv_file_insert_to_postgres(import_to_table_name, file_path)


def run(study_area_name, table_name, output_folder, vect_method, clust_method, metric, prefix, iternation_num, min_textual_distance):
    # todo: Check if the database exists
    # todo: Check if the table exists

    lang = 'en'  # ['sv']    # , 'sv', 'multi']
    # datetime(year=2018, month=9, day=12, hour=0)
    start_date = datetime(year=2018, month=9, day=12, hour=0)
    # hours,  cluster_detection will be applied on [duration] hours
    duration = 24
    steps = 12  # hours, analysis will run every [steps] hours
    # datetime(year=2018, month=9, day=19, hour=0)
    end_date = datetime(year=2018, month=9, day=19, hour=13)
    plot_results = True
    verbose = False
    if clust_method not in ['optics', 'dbscan', 'hdbscan']:
        clust_method = 'optics'
    if vect_method not in ['bow', 'tfidf', 'w2v', 'glove', 'fasttext']:
        vect_method = 'tfidf'

    min_cluster_size = 10
    dbscan_eps = 0.45

    filter_list_for_plotting = [
        'huricaneflorence',
        'hurricane',
        'huricane',
        'florence',
        'tstorm',
        'safe',
        'rest',
        'storm',
        'beach',
        'alert',
        'condition',
        'weather',
        'traffic',
        'flooding',
        'rain',
        'wind',
        'forecast',
        'today',
        'tonight',
        'heavy',
        'warning',
        'shower',
        'cloudy',
        'come',
        'damage',
        'flood',
        'tornado',
        'tropical',
        'acident',
        'rainbow',
        'tropical',
        'physician'
    ]

    study_area_bbox = study_area[study_area_name] if study_area.index.contains(
        study_area_name) else None

    analysis_id = '{}_{}_{}_{}'.format(study_area_name, datetime.now().strftime("%Y%m%d_%H%M%S"), clust_method,
                                       vect_method)
    analysis_output_folder = output_folder + os.sep + (
        analysis_id if len(prefix) == 0 else '{}_{}'.format(prefix, analysis_id))

    execute(lang, clust_method, alpha, beta, gamma, metric, min_textual_distance, vect_method,
            study_area_name,
            table_name, start_date, duration, steps, end_date, analysis_id,
            analysis_output_folder, plot_results,
            min_cluster_size=min_cluster_size,
            dbscan_eps=dbscan_eps,
            filter_list_for_plotting=filter_list_for_plotting,
            study_area_bbox=study_area_bbox,
            verbose=verbose,
            multiprocess=True,
            iternation_num=iternation_num)

    pass

def extract_clusters():


def initialization():
    import nltk
    nltk.download('wordnet', '/data/nltk')
