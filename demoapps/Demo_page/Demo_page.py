###=========================================================================================
# Import the libraries and then add the path.

import dash_core_components as dcc
# from dash import dcc
import dash_html_components as html
# from dash import html
# from dash.dash_table.Format import Group
import dash_table
# from dash import dash_table
import shap
from dash.dependencies import Input, Output
# from dash import ctx
from extremum_config import app
from dash import ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from pathlib import Path
from math import log10
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
# import random


from sktime.classification.interval_based import RandomIntervalSpectralEnsemble
# from sktime.classification.dictionary_based import ContractableBOSS, BOSSEnsemble
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.datatypes._panel._convert import from_2d_array_to_nested

# import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sktime.transformations.panel.rocket import Rocket
# from sklearn import metrics
import sys
import time

# from wildboar.datasets import list_datasets, load_dataset
from wildboar.datasets import load_dataset
from wildboar.ensemble import ShapeletForestClassifier
from wildboar.explain.counterfactual import counterfactuals
from xgboost import XGBClassifier

# import pywt
from pywt import wavedec

# from colors import colors, markers
# from colors import colors, markers
# , waverec
# from tqdm.notebook import tqdm
colors = {
    "1": "darkblue",
    "2": "orangered",
    "3": "aqua",
    "4": "aquamarine",
    "5": "azure",
    "6": "beige",
    "7": "bisque",
    "8": "black",
    "9": "blanchedalmond",
    "10": "blue",
    "11": "blueviolet",
    "12": "brown",
    "13": "burlywood",
    "14": "cadetblue",
    "15": "chartreuse",
    "16": "chocolate",
    "17": "coral",
    "18": "cornflowerblue",
    "19": "cornsilk",
    "20": "crimson",
    "21": "cyan",
    "22": "turquoise",
    "23": "darkcyan",
    "24": "darkgoldenrod",
    "25": "darkgray",
    "26": "darkgrey",
    "27": "darkgreen",
    "28": "darkkhaki",
    "29": "darkmagenta",
    "30": "darkolivegreen",
    "31": "darkorange",
    "32": "darkorchid",
    "33": "darkred",
    "34": "darksalmon",
    "35": "darkseagreen",
    "36": "darkslateblue",
    "37": "darkslategray",
    "38": "darkslategrey",
    "39": "darkturquoise",
    "40": "darkviolet",
    "41": "deeppink",
    "42": "deepskyblue",
    "43": "dimgray",
    "44": "dimgrey",
    "45": "dodgerblue",
    "46": "firebrick",
    "47": "floralwhite",
    "48": "forestgreen",
    "49": "fuchsia",
    "50": "gainsboro",
    "51": "ghostwhite",
    "52": "gold",
    "53": "goldenrod",
    "54": "gray",
    "55": "grey",
    "56": "green",
    "57": "greenyellow",
    "58": "honeydew",
    "59": "hotpink",
    "60": "indianred",
    # "1":"indigo",
    # "1":"ivory",
    # "1":"khaki",
    # "1":"lavender",
    # "1":"lavenderblush",
    # "1":"lawngreen",
    # "1":"lemonchiffon",
    # "1":"lightblue",
    # "1":"lightcoral",
    # "1":"lightcyan",
    # "1":"lightgoldenrodyellow",
    # "1":"lightgray",
    # "1":"lightgrey",
    # "1":"lightgreen",
    # "1":"lightpink",
    # "1":"lightsalmon",
    # "1":"lightseagreen",
    # "1":"lightskyblue",
    # "1":"lightslategray",
    # "1":"lightslategrey",
    # "1":"lightsteelblue",
    # "1":"lightyellow",
    # "1":"lime",
    # "1":"limegreen",
    # "1":"linen",
    # "1":"magenta",
    # "1":"maroon",
    # "1":"mediumaquamarine",
    # "1":"mediumblue",
    # "1":"mediumorchid",
    # "1":"mediumpurple",
    # "1":"mediumseagreen",
    # "1":"mediumslateblue",
    # "1":"mediumspringgreen",
    # "1":"mediumturquoise",
    # "1":"mediumvioletred",
    # "1":"midnightblue",
    # "1":"mintcream",
    # "1":"mistyrose",
    # "1":"moccasin",
    # "1":"navajowhite",
    # "1":"navy",
    # "1":"oldlace",
    # "1":"olive",
    # "1":"olivedrab",
    # "1":"orange",
    # "1":"antiquewhite",
    # "1":"orchid",
    # "1":"palegoldenrod",
    # "1":"palegreen",
    # "1":"paleturquoise",
    # "1":"palevioletred",
    # "1":"papayawhip",
    # "1":"peachpuff",
    # "1":"peru",
    # "1":"pink",
    # "1":"plum",
    # "1":"powderblue",
    # "1":"purple",
    # "1":"red",
    # "1":"rosybrown",
    # "1":"royalblue",
    # "1":"rebeccapurple",
    # "1":"saddlebrown",
    # "1":"salmon",
    # "1":"sandybrown",
    # "1":"seagreen",
    # "1":"seashell",
    # "1":"sienna",
    # "1":"silver",
    # "1":"skyblue",
    # "1":"slateblue",
    # "1":"slategray",
    # "1":"slategrey",
    # "1":"snow",
    # "1":"springgreen",
    # "1":"steelblue",
    # "1":"tan",
    # "1":"teal",
    # "1":"thistle",
    # "1":"tomato",
    # "1":"aliceblue",
    # "1":"violet",
    # "1":"wheat",
    # "1":"white",
    # "1":"whitesmoke",
    # "1":"yellow",
    # "1":"yellowgreen"
}
markers = {
    "1": "circle",
    "2": "star",
    "3": "line-ew",
    "4": "diamond",
    "5": "x-thin",
    "6": "cross",
    "7": "triangle-up",
    "8": "star",
    "9": "hourglass",
    "10": "cross-thin"
}

plt.style.reload_library()
THIS_FILEPATH = os.path.dirname(__file__)
sys.path.append(os.path.join(THIS_FILEPATH, "."))

###=========================================================================================
###=========================================================================================
### LOAD DATASETS and store them in a pickle file whenever you start the page

# UCR_datasets = list_datasets(repository='wildboar/ucr')
#
# dataset_info = pd.DataFrame(columns=['size', 'classes', 'length'], index=UCR_datasets, dtype=float)
#
# for dataset in UCR_datasets:
#     x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')
#
#     # remove rows with missing values
#     x = x_all[~np.isnan(x_all).any(axis=1)]
#     y = y_all[~np.isnan(x_all).any(axis=1)]
#
#     classes = np.unique(y)  # all class labels
#     total_examples, ts_length = x.shape
#
#     dataset_info.loc[dataset] = [total_examples, len(classes), ts_length]
#
# ## Drop rows that have a value of zero in a column
# dataset_info = dataset_info.loc[~(dataset_info == 0).any(axis=1)]
# dataset_info = dataset_info.loc[~(dataset_info == 1).any(axis=1)]
#
# dataset_info.to_pickle('demoapps/demo_page/data/datasets_information.pkl')
# dataset_info = pd.read_pickle('demoapps/demo_page/data/datasets_information.pkl')

# def load_datasets(path):
# 'demoapps/Demo_page/data/datasets_information.pkl'
dataset_info = pd.read_pickle('demoapps/demo_page/data/datasets_information.pkl')
dataset_info.reset_index(inplace=True)
dataset_info.rename(columns={'index': 'Dataset'}, inplace=True)


# selected_datasets = dataset_info.copy()

# print("Read the dataSets and reset the indexes")
# return dataset_info

def transform_value(value):
    """
    Make logarithmic value into normal value
    """
    return 10 ** value


###=========================================================================================
###=========================================================================================
## Select DATASETS in a function based on inputs size, length, classes

def filter_dataset(sizes, classes, lengths):
    """
    The min and max of each variable (after loading and dropping NAN and zero and one values) is:
    lb_size = 40
    up_size = 24000
    lb_length = 15
    up_length = 2844
    lb_class = 2
    up_class = 60
    """
    lb_size = transform_value(sizes[0])
    up_size = transform_value(sizes[1] + 0.01)
    lb_length = transform_value(lengths[0])
    up_length = transform_value(lengths[1] + 0.01)
    lb_class = transform_value(classes[0])
    up_class = transform_value(classes[1] + 0.1)
    # lb_class = 2
    # up_class = 60
    # dataset_info = pd.read_pickle('demoapps/Demo_page/data/datasets_information.pkl')
    # 'demoapps/Demo_page/data/datasets_information.pkl'
    selected_datasets = dataset_info.loc[(dataset_info['size'] >= lb_size) &
                                         (dataset_info['size'] <= up_size) &
                                         (dataset_info['classes'] >= lb_class) &
                                         (dataset_info['classes'] <= up_class) &
                                         (dataset_info['length'] >= lb_length) &
                                         (dataset_info['length'] <= up_length)]
    return selected_datasets


###=========================================================================================
###=========================================================================================
## Mouna's part visualization
def update_plot(x_all, y_all, classes, btn_id, nb_lines, fig, nb_row):
    x = np.arange(1, x_all.shape[1] + 1)
    rows = []

    if btn_id == ctx.triggered_id:
        for j in range(0, len(classes)):
            current = classes[j]
            for i in range(0, len(x_all)):
                if y_all[i] == current:
                    rows.append(i)
            instance = np.random.choice(rows, nb_lines, replace=False)
            for line in range(nb_lines):
                fig.add_trace(go.Scatter(
                    x=x,
                    y=x_all[instance[line]],
                    marker=
                    dict(
                        color=colors[f"{int(current)}"]
                    ),
                    name=f"class {str(int(current))}"),
                    row=nb_row, col=1)
    else:
        for i in range(0, len(x_all)):
            if y_all[i] in classes:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=x_all[i],
                    marker=
                    dict(color=colors[f"{int(y_all[i])}"]
                         ),
                    name=f"class {str(int(y_all[i]))}"),
                    row=nb_row, col=1)
    return fig


def kmeans_clustering(x_all, y_all, classes, nb_clusters, fig, nb_row):
    x = np.arange(1, x_all.shape[1] + 1)
    data = []
    for i in range(0, len(classes)):
        current = classes[i]
        data.clear()
        for j in range(0, len(x_all)):
            if y_all[j] == current:
                data.append(x_all[j])

        km = KMeans(n_clusters=nb_clusters, random_state=0).fit(data)
        labels = km.labels_
        percentages = []
        total = len(labels)
        for l in range(nb_clusters):
            count = labels.tolist().count(l)
            percentages.append(round(((count / total) * 100)))

        centroids = km.cluster_centers_
        for k in range(len(centroids)):
            fig.add_trace(go.Scatter(
                x=x,
                y=centroids[k],
                mode='lines+markers',
                marker=
                dict(
                    symbol=markers[str(k + 1)],
                    size=6,
                    color=colors[f"{current}"]),
                line=dict(
                    color=colors[str(current)],
                    width=1
                ),
                name=f"class {str(int(current))} ({k + 1}) {percentages[k]}%",
            ), row=nb_row, col=1)
    return fig


###=========================================================================================
###=========================================================================================
## Classifiers performance (Arvin)
classifiers_lst = ["kNN", "RSF", "Rocket", "RISE", "BOSS", "DFT", "DWT"]
# knn_metrics = sklearn.neighbors.VALID_METRICS['brute']
knn_metrics = ['cityblock', 'cosine', 'euclidean', 'nan_euclidean']
rsf_metrics = ['euclidean', 'scaled_euclidean', 'scaled_dtw']
counter_metrics = ['euclidean', 'scaled_dtw']


def classifiers_performance(x_all, y_all, len_coef_DFT, len_coef_DWT, classifiers, splits, knn_metric,
                            rsf_metric, split_size, shapelet_size,
                            max_depth, n_estimator):
    # classifiers = ["kNN"]

    # selected_dataset = pd.read_pickle('demoapps/Demo_page/data/selected_dataset.pkl')
    # selected_dataset = selected_dataset["Dataset"].to_list()
    # # splits = 1
    # selected_dataset = selected_dataset[0]

    # iterables = [[selected_dataset], np.arange(splits)]
    # m_index = pd.MultiIndex.from_product(iterables, names=["dataset", "split"])
    m_index = np.arange(splits)
    # print("The indexes are printed: ", m_index)
    accuracies = pd.DataFrame(index=m_index, columns=classifiers, dtype=float)
    complexity = pd.DataFrame(index=m_index, columns=classifiers, dtype=float)
    # print("accuracies: ", accuracies)
    # print("complexity:", complexity)
    # x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')

    x_all, y_all = x_all, y_all

    x = x_all[~np.isnan(x_all).any(axis=1)]
    y = y_all[~np.isnan(x_all).any(axis=1)]

    classes = np.unique(y)  # all class labels
    total_examples, ts_length = x.shape

    x_ind = np.arange(total_examples)

    max_len_coef_DFT = int(ts_length / 2 + 1)  # maximum number of DFT coefficients
    max_len_coef_DWT = ts_length  # maximum number of DWT coefficients

    # for i in tqdm(range(splits), desc = selected_dataset, leave=False):
    for i in range(splits):
        # implement same split across all
        np.random.seed(i)

        x_train_ind, x_test_ind, y_train, y_test = train_test_split(x_ind, y, test_size=split_size,
                                                                    random_state=i, shuffle=True, stratify=None)
        # x_train2_ind, x_val_ind, y_train2, y_val = train_test_split(x_train_ind, y_train, test_size=.20, random_state=i,
        #                                                             shuffle=True, stratify=None)
        x_train = x[x_train_ind, :]
        x_test = x[x_test_ind, :]

        np.random.seed(i)

        ## kNN
        if 'kNN' in classifiers:
            knn_time_start = time.time()
            # sklearn.metrics.pairwise.distance_metrics()
            clf_kNN = KNeighborsClassifier(metric=knn_metric)  # 'euclidean'
            clf_kNN.fit(x_train, y_train)
            acc = clf_kNN.score(x_test, y_test)

            knn_time_end = time.time()
            pickle.dump(clf_kNN, open('demoapps/demo_page/data/clf_kNN.pkl', 'wb'))

            accuracies.loc[i, "kNN"] = acc
            complexity.loc[i, "kNN"] = knn_time_end - knn_time_start

            # y_train_knn = y_train[y_train != 1.0] = -1.0
            # y_test_knn = y_test[y_test != 1.0] = -1.0

        # ----------

        ## RSF
        if 'RSF' in classifiers:
            rsf_time_start = time.time()

            clf_RSF = ShapeletForestClassifier(n_estimators=n_estimator, metric=rsf_metric,
                                               max_shapelet_size=shapelet_size,
                                               max_depth=max_depth,
                                               random_state=1)  # 'scaled_euclidean' n_estimators=50,
            # clf_RSF = ShapeletForestClassifier(
            #     n_estimators=20,
            #     metric='euclidean',
            #     max_depth=5,
            #     max_shapelet_size=0.4,  # INTERACTION: Make this as input from user
            #     random_state=1,
            # )
            clf_RSF.n_features_in_ = x_train.shape[-1]
            clf_RSF.fit(x_train, y_train)
            acc = clf_RSF.score(x_test, y_test)

            rsf_time_end = time.time()
            accuracies.loc[i, "RSF"] = acc
            complexity.loc[i, "RSF"] = rsf_time_end - rsf_time_start
            pickle.dump(clf_RSF, open('demoapps/demo_page/data/clf_RSF.pkl', 'wb'))

            # y_train_rsf = y_train[y_train != 1.0] = -1.0
            # y_test_rsf = y_test[y_test != 1.0] = -1.0

        # ----------

        ## ROCKET
        if 'Rocket' in classifiers:
            rocket_time_start = time.time()

            rocket = Rocket(num_kernels=20000, normalise=False, n_jobs=1,
                            random_state=None)  # by default, ROCKET uses 10,000 kernels
            rocket.fit(np.expand_dims(x_train, axis=1))

            # transform training set and train classifier
            x_training_transform = rocket.transform(np.expand_dims(x_train, axis=1))

            classifierRocket = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))

            classifierRocket.fit(x_training_transform, y_train)

            # transform test set and predict
            x_test_transform = rocket.transform(np.expand_dims(x_test, axis=1))
            # predictions = classifierRocket.predict(x_test_transform)
            acc = classifierRocket.score(x_test_transform, y_test)

            rocket_time_end = time.time()

            accuracies.loc[i, "Rocket"] = acc
            complexity.loc[i, "Rocket"] = rocket_time_end - rocket_time_start
        # -------------

        ## RISE
        if 'RISE' in classifiers:
            rise_time_start = time.time()
            classifierRISE = RandomIntervalSpectralEnsemble(random_state=i)

            x_train_nested = from_2d_array_to_nested(x_train)
            x_test_nested = from_2d_array_to_nested(x_test)

            classifierRISE.fit(x_train_nested, y_train)
            # predictions = classifierRISE.predict(x_test_nested)
            acc = classifierRISE.score(x_test_nested, y_test)
            rise_time_end = time.time()

            accuracies.loc[i, "RISE"] = acc
            complexity.loc[i, "RISE"] = rise_time_end - rise_time_start
        # --------------

        ## BOSS
        if 'BOSS' in classifiers:
            boss_time_start = time.time()
            classifierBOSS = BOSSEnsemble(random_state=i)

            x_train_nested = from_2d_array_to_nested(x_train)
            x_test_nested = from_2d_array_to_nested(x_test)

            classifierBOSS.fit(x_train_nested, y_train)
            # predictions = classifierBOSS.predict(x_test_nested)
            acc = classifierBOSS.score(x_test_nested, y_test)
            boss_time_end = time.time()

            accuracies.loc[i, "BOSS"] = acc
            complexity.loc[i, "BOSS"] = boss_time_end - boss_time_start
        # --------------

        ## DFT
        ## Changing the number of lower frequencies features using a sliding bar
        if 'DFT' in classifiers:
            dft_time_start = time.time()

            # Freq domain
            X_F = np.fft.rfft(x)

            # combine amp and phase into one array
            # X_F_train = np.empty((x_train2_ind.shape[0], 2 * max_len_coef_DFT))
            X_F_train = np.empty((x_train_ind.shape[0], 2 * max_len_coef_DFT))
            # X_F_val = np.empty((x_val_ind.shape[0], 2 * max_len_coef_DFT))
            X_F_test = np.empty((x_test.shape[0], 2 * max_len_coef_DFT))

            # X_F_train[:, 0::2] = normalize(np.absolute(X_F[x_train2_ind, :]), axis=1)  # amplitude
            # X_F_train[:, 1::2] = normalize(np.angle(X_F[x_train2_ind, :]), axis=1)  # phase
            X_F_train[:, 0::2] = normalize(np.absolute(X_F[x_train_ind, :]), axis=1)  # amplitude
            X_F_train[:, 1::2] = normalize(np.angle(X_F[x_train_ind, :]), axis=1)  # phase
            # X_F_val[:, 0::2] = normalize(np.absolute(X_F[x_val_ind, :]), axis=1)  # amplitude
            # X_F_val[:, 1::2] = normalize(np.angle(X_F[x_val_ind, :]), axis=1)  # phase
            X_F_test[:, 0::2] = normalize(np.absolute(X_F[x_test_ind, :]), axis=1)  # amplitude
            X_F_test[:, 1::2] = normalize(np.angle(X_F[x_test_ind, :]), axis=1)  # phase

            # fracs = [.1,.25,.5,.75,1]
            # acc_val = np.empty(len(fracs))
            # acc_test = np.empty(len(fracs))

            #             for idx, L_DFT_frac in enumerate(fracs):

            #                 L_DFT = int(L_DFT_frac*max_len_coef_DFT)
            #                 clf = ExtraTreesClassifier(random_state=1)
            #                 clf.fit(X_F_train[:,:2*L_DFT], y_train2)
            #                 acc_val[idx] = clf.score(X_F_val[:,:2*L_DFT], y_val)
            #                 acc_test[idx] = clf.score(X_F_test[:,:2*L_DFT], y_test)

            L_DFT = int(len_coef_DFT)
            # clf = ExtraTreesClassifier(random_state=1)
            xgbc_DFT = XGBClassifier(random_state=1)
            # clf.fit(X_F_train[:, :2 * L_DFT], y_train2)
            xgbc_DFT.fit(X_F_train[:, :2 * L_DFT], y_train)
            pickle.dump(X_F_train[:, :2 * L_DFT], open('demoapps/demo_page/data/X_F_train_DFT.pkl', 'wb'))
            # acc_val = clf.score(X_F_val[:, :2 * L_DFT], y_val)
            pickle.dump(xgbc_DFT, open('demoapps/demo_page/data/xgbc_DFT.pkl', 'wb'))
            acc_test = xgbc_DFT.score(X_F_test[:, :2 * L_DFT], y_test)

            dft_time_end = time.time()

            # accuracies.loc[i, "DFT"] = acc_test[np.argmax(acc_val)]

            accuracies.loc[i, "DFT"] = acc_test
            complexity.loc[i, "DFT"] = dft_time_end - dft_time_start

        # --------------
        ## Changing the number of lower frequencies features using a sliding bar

        if 'DWT' in classifiers:
            dwt_time_start = time.time()

            # Time-Freq domain
            level = 3  # no of DWT decomposition levels
            waveletname = 'db2'

            X_TF = wavedec(x, waveletname, level=level)  ## If using DWT
            X_TF_stacked = np.hstack(X_TF)

            # X_TF_train = X_TF_stacked[x_train2_ind, :]
            X_TF_train = X_TF_stacked[x_train_ind, :]
            # X_TF_val = X_TF_stacked[x_val_ind, :]
            X_TF_test = X_TF_stacked[x_test_ind, :]

            #             fracs = [.1,.25,.5,.75,.1]
            #             acc_val = np.empty(len(fracs))
            #             acc_test = np.empty(len(fracs))
            #             for idx, L_DWT_frac in enumerate(fracs):

            #                 L_DWT = int(L_DWT_frac*max_len_coef_DWT)
            #                 L_DWT = int(len_coef_DWT)
            #                 clf = ExtraTreesClassifier(random_state=1)

            #                 clf.fit(X_TF_train[:,:L_DWT], y_train2)
            #                 acc_val[idx] = clf.score(X_TF_val[:,:L_DWT], y_val)
            #                 acc_test[idx] = clf.score(X_TF_test[:,:L_DWT], y_test)

            L_DWT = int(len_coef_DWT)
            # clf = ExtraTreesClassifier(random_state=1)
            xgbc_DWT = XGBClassifier(random_state=1)
            # clf.fit(X_TF_train[:, :L_DWT], y_train2)
            xgbc_DWT.fit(X_TF_train[:, :L_DWT], y_train)
            # acc_val = clf.score(X_TF_val[:, :L_DWT], y_val)
            pickle.dump(X_TF_train[:, :L_DWT], open('demoapps/demo_page/data/X_F_train_DWT.pkl', 'wb'))
            pickle.dump(xgbc_DWT, open('demoapps/demo_page/data/xgbc_DWT.pkl', 'wb'))
            acc_test = xgbc_DWT.score(X_TF_test[:, :L_DWT], y_test)

            dwt_time_end = time.time()

            # accuracies.loc[i, "DWT"] = acc_test[np.argmax(acc_val)]

            accuracies.loc[i, "DWT"] = acc_test
            complexity.loc[i, "DWT"] = dwt_time_end - dwt_time_start
            # --------------

            # save results
            # accuracies.to_pickle('data/accuracies_new.pkl')
            # complexity.to_pickle('data/complexity_new.pkl')
            # accuracies = dash_table.DataTable(
            #     columns=[
            #         {"name": i, "id": i, "deletable": True, "selectable": True} for i in accuracies.columns],
            #         data=accuracies.to_dict('records'),
            #         page_current= 0,
            #         page_size= 10,
            #     )
            # complexity = dash_table.DataTable(
            #     columns=[
            #         {"name": i, "id": i, "deletable": True, "selectable": True} for i in complexity.columns],
            #         data=complexity.to_dict('records'),
            #         page_current= 0,
            #         page_size= 10,
            #     )
    accuracies.loc['Average'] = accuracies.mean()
    complexity.loc['Average'] = complexity.mean()
    accuracies.reset_index(inplace=True)
    accuracies.rename(columns={'index': 'split'}, inplace=True)
    complexity.reset_index(inplace=True)
    complexity.rename(columns={'index': 'split'}, inplace=True)
    return accuracies, complexity, max_len_coef_DFT, max_len_coef_DWT


###=========================================================================================
###=========================================================================================
# We have the entire layout here 

layout = html.Div(children=[
    html.Div(
        children=[
            html.H2(
                id="banner-title",
                children=[
                    html.A(
                        "Post-hoc Explainability for Time Series Classification",
                        # href="https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1643492&dswid=4165",
                        style={
                            "text-decoration": "none",
                            "color": "inherit",
                        },
                    )
                ],
            ),
            html.A(),
        ],
        style={"text-align": "center", "padding": "20px 0px"}
    ),
    html.Div([
        html.H5('Motivation'),
        dcc.Markdown("Time series data correspond to observations of phenomena that are recorded over time. "
                     "Such data is encountered regularly in a wide range of applications such as speech and"
                     "music recognition, monitoring health and medical diagnosis, financial analysis, motion"
                     "tracking, and shape identification, to name a few. With such a diversity of applications"
                     "and the large variations in their characteristics, time series classification is a complex and"
                     "challenging task. One of the fundamental steps in the design of time series classification is"
                     "that of defining or constructing the discriminant features that help differentiate between"
                     "classes. This is typically achieved by designing novel representation techniques that"
                     "transform the raw time series data to a new data domain where subsequently a classifier"
                     "is trained on the transformed data, such as one-nearest neighbors or random forests."
                     "In recent time series classification approaches, deep neural network models have been"
                     "employed which are able to jointly learn a representation of time series and perform"
                     "classification. In many of these sophisticated approaches, the discriminant features"
                     "tend to be complicated to analyze and interpret, given the high degree of non-linearity.")
    ],
        style={"float": "left", "width": '100%', "margin": '.5%', 'border': '0 lightgrey solid',
               'display': 'inline-block', 'height': 'auto',
               "padding": 10, "borderRadius": 5, 'flex': 0},
    ),
    html.Div(
        id="range_sliders",
        children=[
            html.H4('Filters'),
            html.H5('Sizes'),
            dcc.RangeSlider(id='sizes',
                            min=log10(40),
                            max=log10(24000),
                            step=log10(1),
                            marks={
                                log10(137): {"label": '137 [0.1]', "style": {"transform": "rotate(30deg)"}},
                                log10(200): {"label": '200[0.2]', "style": {"transform": "rotate(30deg)"}},
                                log10(312): {"label": '312[0.3]', "style": {"transform": "rotate(30deg)"}},
                                log10(462): {"label": '462[0.4]', "style": {"transform": "rotate(30deg)"}},
                                log10(724): {"label": '724[0.5]', "style": {"transform": "rotate(30deg)"}},
                                log10(888): {"label": '888[0.6]', "style": {"transform": "rotate(30deg)"}},
                                log10(1035): {"label": '1035[0.7]', "style": {"transform": "rotate(30deg)"}},
                                log10(2250): {"label": '2250[0.8]', "style": {"transform": "rotate(30deg)"}},
                                log10(4362): {"label": '4362[0.9]', "style": {"transform": "rotate(30deg)"}},
                            },
                            value=[log10(40), log10(24000)],
                            # tooltip={"placement": "bottom", "always_visible": True},
                            allowCross=False,
                            # pad={"t": 50},
                            # style = {'height': '200px'}
                            updatemode='drag'
                            ),
            # html.Br(),
            # html.Br(),
            html.H5('Classes'),
            dcc.RangeSlider(id='classes',
                            min=log10(2),
                            max=log10(60),
                            step=log10(1),
                            marks={
                                log10(2): {"label": '2[0.3]', "style": {"transform": "rotate(30deg)"}},
                                log10(3): {"label": '3[0.5]', "style": {"transform": "rotate(30deg)"}},
                                log10(5): {"label": '5[0.6]', "style": {"transform": "rotate(30deg)"}},
                                log10(6): {"label": '6[0.7]', "style": {"transform": "rotate(30deg)"}},
                                log10(10): {"label": '10[0.8]', "style": {"transform": "rotate(30deg)"}},
                                log10(16): {"label": '16[0.9]', "style": {"transform": "rotate(30deg)"}},
                            },
                            value=[log10(2), log10(60)],
                            # tooltip={"placement": "bottom", "always_visible": False},
                            allowCross=False,
                            updatemode='drag'
                            ),
            # html.Br(),
            # html.Br(),
            html.H5('Lengths'),
            dcc.RangeSlider(id='lengths',
                            min=log10(24),
                            max=log10(2844),
                            step=log10(1),
                            marks={
                                log10(80): {"label": '80[0.1]', "style": {"transform": "rotate(30deg)"}},
                                log10(128): {"label": '128[0.2]', "style": {"transform": "rotate(30deg)"}},
                                log10(150): {"label": '150[0.3]', "style": {"transform": "rotate(30deg)"}},
                                log10(270): {"label": '270[0.4]', "style": {"transform": "rotate(35deg)"}},
                                log10(301): {"label": '301[0.5]', "style": {"transform": "rotate(35deg)"}},
                                log10(441): {"label": '441[0.6]', "style": {"transform": "rotate(30deg)"}},
                                log10(571): {"label": '571[0.7]', "style": {"transform": "rotate(30deg)"}},
                                log10(870): {"label": '870[0.8]', "style": {"transform": "rotate(30deg)"}},
                                log10(1476): {"label": '1476[0.9]', "style": {"transform": "rotate(30deg)"}},
                            },
                            value=[log10(24), log10(2844)],
                            # tooltip={"placement": "bottom", "always_visible": False},
                            allowCross=False,
                            updatemode='drag'
                            ),
        ],
        style={"width": '100%', "margin": '0.5%',
               'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
               "padding": 10, "borderRadius": 5}
    ),  ## End of range sliders
    # html.Br(),
    # html.Br(),
    html.Div(id='datatable-filtered',
             style={"width": '100%', "margin": '0.5%',
                    'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                    "padding": 10, "borderRadius": 5}
             ),
    html.Div(children=[
        html.H5('Graph of filtered datasets'),
        dcc.Graph(id="3Dgraph"),
    ],
        style={"width": '100%', "margin": '0.5%',
               'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
               'background-color': '#e0f0e0',
               "padding": 10, "borderRadius": 5}
    ),
    # html.Br(),
    # html.Br(),
    html.Div(id="dropdown_dataset",
             children=[
                 html.Label('Select Dataset'),
                 dcc.Dropdown(
                     id="chosen_dataset",
                     clearable=True,
                     searchable=True,
                 ),
                 # html.Br(),
                 # html.Br(),
                 html.H5('Dataset Selected'),
                 html.Div(id='dataset',
                          children=[],
                          # style={"width": '100%', "margin": '0.5%',
                          #        'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                          #        'background-color': '#e0f0e0',
                          #        "padding": 10, "borderRadius": 5}
                          ),
                 # dcc.Markdown(id='link')
                 # html.A(id='link', children=["For more information about the dataset: [Click here]"], target="_blank")
                 dcc.Link(
                     html.A("For more information about the dataset click here"),
                     href="",
                     # title="For more information about the dataset click here",
                     target="_blank",
                     style={'color': 'blue', 'text-decoration': 'none'},
                     id='link',
                 ),
                 html.Br(),
                 html.Br(),
                 html.Button('Submit', id='submitval', n_clicks=0,
                             style={'flush': 'right', 'padding': '10px', 'float': 'left'}, ),
             ],
             style={"width": '100%', "margin": '0.5%',
                    'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                    "padding": 10, "borderRadius": 5}
             ),
    # html.Br(),
    # html.Br(),
    # html.Br(),
    # html.Br(),
    # html.Br(),
    ## ====================================================================================
    ## ====================================================================================
    ## Mouna's part visualization
    ## Multi-dropdown of classes from selected dataset
    html.A(
        "Sample Representation",
        style={
            "text-decoration": "none",
            'font-size': 30,
        },
    ),
    html.Div([
        dcc.Markdown("Each label in the dataset has many samples inside.")
    ]),
    html.Div(
        id="time-series",
        children=[
            html.Div(
                children=[
                    html.H5('Dataset classes'),
                    dcc.Dropdown(
                        options=[],
                        id="multi-dropdown",
                        placeholder="Select classes",
                        multi=True,
                        value=[],
                        clearable=False,
                    ),
                    html.Br(),
                    html.Label('Number of samples to randomly plot from each class'),
                    dcc.Slider(1, 10, 1,
                               value=1,
                               id='nb-lines-slider'),

                    ## Button to select new instance of time serie
                    html.Button('Plot randomly', id='button', n_clicks=0),
                ], style={"width": '100%', "margin": '0.5%',
                          'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                          # 'background-color': '#e0f0e0',
                          "padding": 10, "borderRadius": 5},
            ),

            ## Time-series plot
            html.Div(
                children=[
                    dcc.Graph(id='sample')
                ],
                style={"width": '100%', "margin": '0.5%',
                       'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                       'background-color': '#e0f0e0',
                       "padding": 10, "borderRadius": 5},
            ),
            html.Br(),
            ## K-means centroids plot
            html.Div([
                html.H5('Centroids'),
                dcc.Markdown("Plotting centroids with K-Means sklearn")
            ]),
            html.Div(
                id="k-means",
                children=[
                    html.Label('Number of clusters'),
                    dcc.Slider(1, 10, 1,
                               value=2,
                               id='nb-clusters-slider'),
                    dcc.Graph(id='kmeans-graph')
                ],
                style={"width": '100%', "margin": '0.5%',
                       'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                       'background-color': '#e0f0e0',
                       "padding": 10, "borderRadius": 5},
            ),
        ]),

    html.Br(),

    # =============================== FREQUENCY DOMAIN ===============================
    ## Multi-dropdown of classes from selected dataset
    html.A(
        "Frequency domain representation",
        style={
            "text-decoration": "none",
            'font-size': 30,
        },
    ),
    html.Div([
        dcc.Markdown("Each data point $x_t$ of the time series is decomposed into $T$ frequency components",
                     mathjax=True),
    ]),
    html.Div(
        id="frequency domain",
        children=[
            html.H5('Dataset classes'),
            html.Div(
                children=[
                    dcc.Dropdown(
                        options=[],
                        id="multi-dropdown-fd",
                        placeholder="Select classes",
                        multi=True,
                        value=[],
                        clearable=False,
                    ),
                    html.Br(),
                    html.Label('Number of decomposed samples to randomly plot from each class'),
                    dcc.Slider(1, 10, 1,
                               value=1,
                               id='nb-lines-slider-fd'),
                    ## Button to select new instance of frequency domain
                    html.Button('Plot randomly', id='button-fd', n_clicks=0),
                ],
                style={"width": '100%', "margin": '0.5%',
                       'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                       # 'background-color': '#e0f0e0',
                       "padding": 10, "borderRadius": 5},
            ),
            ## Frequency-domain plots
            html.Div(
                children=[
                    dcc.Graph(id='amplitude-phase-plot'),
                ],
                style={"width": '100%', "margin": '0.5%',
                       'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                       'background-color': '#e0f0e0',
                       "padding": 10, "borderRadius": 5},
            ),
            html.Br(),
            ## K-means centroids plot
            html.Div([
                html.H5('Centroids'),
                dcc.Markdown("Plotting centroids with K-Means sklearn")
            ]),
            html.Div(
                id="k-means",
                children=[
                    html.Label('Choose number of clusters'),
                    dcc.Slider(1, 10, 1,
                               value=2,
                               id='nb-clusters-slider-fd'),
                    dcc.Graph(id='kmeans-graph-fd'),
                ],
                style={"width": '100%', "margin": '0.5%',
                       'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                       'background-color': '#e0f0e0',
                       "padding": 10, "borderRadius": 5},
            ),
        ]),
    ## ====================================================================================
    ## ====================================================================================
    # html.Br(),
    # html.Br(),
    html.Div(id='Performance_Description',
             children=[
                 html.H4("Approaches for time series classification"),
                 html.H5("k-Nearest Neighbor (kNN):"),
                 dcc.Markdown(
                     "The kNN classifier belongs to a baseline family of classifiers that strongly depend on a pair-wise time series distance measure,"
                     " such as the Euclidean distance, or more elastic measures such as dynamic time warping (DTW)."
                     " We use a 1-NN classifier with Euclidean distance."),
                 html.H5("Random Shapelet Forest (RSF):"),
                 dcc.Markdown(
                     "The RSF classifier belongs to the class of shapelet-based approaches for time series classification."
                     " Shapelets are time series subsequences that are generated to have strong similarities with specific classes of time series."
                     " The classifier efficiency then relies on how well the shapelets are constructed."
                     " The RSF classifier generates the shapelets with different lengths randomly from the time series dataset and constructs a dedicated random forest classifier."
                     " As we will see later, the classifications by the shapelet-based classifiers are explainable by relating to the relevant shapelets that the time series share high similarities with. "),
                 html.H5("BOSS, Rocket, RISE:"),
                 dcc.Markdown(
                     "These three classifiers belong to the class of transformation-based approaches for time series classification."
                     " In this class of approaches, several types of transformations exist:"
                     " Dictionary-based approaches map patterns in a time series to symbolic representations and subsequently build a time series representation composed of sequences of symbols."
                     " This technique is useful for time series similarity search through string matching and is utilized in the BOSS classifier."
                     " A recent line of research in transformation-based time series classification employs convolution kernels, which are used to transform the time series to a very large dimensional space in which efficient separation of the class is possible."
                     " The best performing classifier both in terms of training time as well as classification accuracy is ROCKET, which generates a large number of random convolution kernels to transform the time series, after which a linear classifier is applied."
                     " Other transformation-based approaches correspond to spectral transforms where the auto-correlation function and power spectrum are computed on time series intervals as in the RISE classifier."),
                 # dcc.Markdown('Classifiers performance'
                 #              'We have the following tasks: '),
                 # dcc.Markdown(
                 #     '1. Split the dataset into training and test data: we split by a slide from 1 to 10'),
                 # dcc.Markdown(
                 #     '2. Train a set of time series classifiers on the training data we have a multi select dropdown menu to select classifiers'),
                 # dcc.Markdown(
                 #     '3. DWT model we change the number of lower frequencies features using a sliding bar.'
                 #     'For this we have to change the maximum according to the selected dataset'),
                 # dcc.Markdown(
                 #     '4. DFT model we change the number of lower frequencies features using a sliding bar.'
                 #     'For this we have to change the maximum according to the selected dataset'),
                 # dcc.Markdown('5. Random Shapeless Forrest (RSF)'),
                 # dcc.Markdown('6. kNN L2 norm'),
                 # dcc.Markdown('7. Rocket '),
                 # dcc.Markdown('8. RISE'),
                 # dcc.Markdown('9. BOSS'),
             ],
             style={"width": '100%', "margin": '0.5%',
                    # 'border': 'thin lightgrey solid',
                    'display': 'inline-block', 'height': 'auto',
                    # 'background-color': '#e0f0e0',
                    "padding": 10, "borderRadius": 5},
             ),
    # html.Br(),
    # html.Br(),
    html.Div(children=[
        html.H4("Hyperparameter Interface"),
        html.Label("Number of splits: Times we randomly shuffle the data in our test and training dataset."),
        html.Br(),
        dcc.Slider(id='splits',
                   min=1,
                   max=10,
                   step=1,
                   value=1,
                   marks=None,
                   tooltip={"placement": "bottom", "always_visible": True},
                   ),
        html.Br(),
        html.Label("Split Size"),
        html.Br(),
        dcc.Slider(id='split_size',
                   min=0.1,
                   max=0.9,
                   step=0.1,
                   value=0.2,
                   # marks={'label': i, 'value': i} for i in np.arange(0.1, 0.9, 0.1),
                   tooltip={"placement": "bottom", "always_visible": True},
                   ),
        html.Br(),
        html.Label('Classifiers to choose from:'),
        dcc.Dropdown(
            id="classifiers",
            options=[
                {'label': i, 'value': i} for i in classifiers_lst
            ],
            clearable=True,
            multi=True,
            # value=["kNN", "RSF"],
            value=["DFT", "DWT"],
            # value=["kNN"],
        ),
        html.Br(),
        html.Div(children=[
            html.Label('DFT: Changing the number of lower frequencies features '),
            dcc.Slider(id='max_len_coef_DFT',
                       min=1,
                       step=1,
                       value=10,
                       marks=None,
                       tooltip={"placement": "bottom", "always_visible": True},
                       ),
        ], id='len_DFT',
            # style={'display': 'block'}
        ),
        html.Br(),
        html.Div(children=[
            html.Label('DWT: Changing the number of lower frequencies features '),
            dcc.Slider(id='max_len_coef_DWT',
                       min=1,
                       step=1,
                       value=10,
                       marks=None,
                       tooltip={"placement": "bottom", "always_visible": True},
                       ),
        ], id='len_DWT',
            # style={'display': 'block'}
        ),
        html.Div(
            children=[
                html.Label('Select kNN Metric to use for distance computation'),
                dcc.Dropdown(
                    id="knn_metric_dropdown",
                    options=[
                        {'label': i, 'value': i} for i in knn_metrics
                    ],
                    value=knn_metrics[2],
                ), ], id="knn_metric",
            # style={'display': 'block'}
        ),
        html.Br(),
        html.Br(),
        html.Div(id="rsf_metric", children=[
            html.Label('Select RSF Metric to use for distance computation'),
            dcc.Dropdown(
                id="rsf_metric_dropdown",
                options=[
                    {'label': i, 'value': i} for i in rsf_metrics
                ],
                value=rsf_metrics[1],
            ),
            html.Br(),
            html.Br(),
            html.Label('Number of estimators for RSF'),
            dcc.Slider(id='n_estimator',
                       min=10,
                       max=200,
                       step=10,
                       value=20,
                       # marks={'label': i, 'value': i} for i in np.arange(0.1, 0.9, 0.1),
                       tooltip={"placement": "bottom", "always_visible": True},
                       ),
            html.Br(),
            html.Br(),
            html.Label('Maximum Shapelet Size'),
            dcc.Slider(id='shapelet_size',
                       min=0.1,
                       max=0.9,
                       step=0.1,
                       value=0.4,
                       # marks={'label': i, 'value': i} for i in np.arange(0.1, 0.9, 0.1),
                       tooltip={"placement": "bottom", "always_visible": True},
                       ),
            html.Br(),
            html.Br(),
            html.Label('Maximum Depth of RSF'),
            dcc.Slider(id='max_depth',
                       min=1,
                       max=10,
                       step=1,
                       value=2,
                       # marks={'label': i, 'value': i} for i in np.arange(0.1, 0.9, 0.1),
                       tooltip={"placement": "bottom", "always_visible": True},
                       ),
        ],  # style={'display': 'block'}
                 ),
    ],
        style={"width": '100%', "margin": '0.5%',
               'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
               # 'background-color': '#e0f0e0',
               "padding": 10, "borderRadius": 5}
    ),
    ## --------
    html.Br(),
    html.Br(),
    # html.Label('Dataset Name: '),
    html.Div(id="Dataset_name", ),
    html.Br(),
    html.Br(),
    html.H4('Accuracy Table'),
    html.Div(id="acc_table", ),
    html.Br(),
    html.H4('Complexity Table'),
    html.Div(id="comp_table", ),
    html.Br(),
    html.Br(),

    html.Div(children=[
        html.H4("Expandability"),
        dcc.Markdown(
            "In this part, we try to explain different aspects of the model and the process of labeling the time series."),
        html.H5("Counterfactuals:"),
        dcc.Markdown(
            "Counterfactual explanations (CFEs) are an emerging technique for local, example-based post-hoc explanations methods."
            " Given a data point A and its prediction P from a model,"
            " a counterfactual is a datapoint close to A, such that the model predicts it to be in a different class Q (P ≠ Q). "
            "A close data point is considered a minimal change that needs to be made to our original data point to get a different prediction."
            " This is helpful in situations like the rejection of a loan or credit card request,"
            " where the applicant is willing to know about the smallest change in the feature set that can lead to acceptance of the request."),
        html.H5("Counterfactual Explanations for RSF and KNN"),
    ],
        style={"width": '100%', "margin": '0.5%',
               # 'border': 'thin lightgrey solid',
               'display': 'inline-block', 'height': 'auto',
               # 'background-color': '#e0f0e0',
               "padding": 10, "borderRadius": 5}
    ),
    html.Div(children=[
        html.H5('Select Example Class'),
        dcc.Dropdown(
            id='example_class',
            options=[
                {'label': i, 'value': i} for i in [1.0, 2.0]
            ],
            # clearable=False,
            value=1.0,
        ),
        html.Br(),
        html.Br(),
        html.H5('Select Desired Class'),
        dcc.Dropdown(
            id='desired_class',
            options=[
                {'label': i, 'value': i} for i in [1.0, 2.0]
            ],
            # clearable=False,
            value=2.0,
        ),
        # html.Br(),
        # html.Button('Submit Classes', id='btn_nclicks_1', n_clicks=0),
        html.Br(),
        html.Br(),
        html.H5('Select data point to use for distance computation'),
        dcc.Slider(id='data_slider',
                   min=1,
                   max=10,
                   step=1,
                   value=1,
                   # marks=None,
                   tooltip={"placement": "bottom", "always_visible": True},
                   ),
    ],
        style={"width": '100%', "margin": '0.5%',
               'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
               # 'background-color': '#e0f0e0',
               "padding": 10, "borderRadius": 5}
    ),
    html.Br(),
    html.Br(),
    html.Div(children=[
        html.Div(id="Dataset_name1", children=[]),
        dcc.Graph(id='counterfactuals'),
    ],
        style={"width": '100%', "margin": '0.5%',
               'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
               'background-color': '#e0f0e0',
               "padding": 10, "borderRadius": 5}
    ),
    html.Br(),
    html.Br(),
    html.Div(children=[
        html.H5("Number of Features in DFT graph"),
        dcc.Slider(id="n_features_DFT",
                   min=1,
                   max=10,
                   step=1,
                   value=10,
                   tooltip={"placement": "bottom", "always_visible": True},
                   ),
        html.H5("DFT Feature Importance"),
        dcc.Graph(id='feature_importance_DFT'),
    ],
    ),
    html.Div(children=[
        html.H5("Number of Features in DWT graph"),
        dcc.Slider(id="n_features_DWT",
                   min=1,
                   max=10,
                   step=1,
                   value=10,
                   tooltip={"placement": "bottom", "always_visible": True},
                   ),
        html.H5("DWT Feature Importance"),
        dcc.Graph(id='feature_importance_DWT'),
    ],
    ),

],
),


###=========================================================================================
###=========================================================================================
## We have the load dataset for rangeSliders callback and function here and we update the options in our dropdown menu and then we can choose a dataset from the options.

@app.callback(

    Output(component_id="3Dgraph", component_property="figure"),
    Output(component_id='datatable-filtered', component_property='children'),
    Output(component_id="chosen_dataset", component_property="options"),
    Output(component_id='dataset', component_property='children'),
    Output(component_id='chosen_dataset', component_property='value'),
    Output(component_id='link', component_property='href'),
    [Input(component_id='3Dgraph', component_property='clickData'),
     Input(component_id='sizes', component_property='value'),
     Input(component_id='classes', component_property='value'),
     Input(component_id='lengths', component_property='value'),
     Input(component_id='chosen_dataset', component_property='value'),
     ])
def update_image_src(clickData, sizes, classes, lengths, chosen_dataset):
    selected_datasets = filter_dataset(sizes, classes, lengths)

    table = dash_table.DataTable(
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True} for i in selected_datasets.columns
        ],
        style_cell={'textAlign': 'left'},
        data=selected_datasets.to_dict('records'),
        page_current=0,
        page_size=10,
    )

    fig = px.scatter_3d(selected_datasets, x='size', y='classes', z='length',
                        text='Dataset', opacity=0.7, log_x=True, log_y=True, log_z=True)

    fig.layout.update(title="3D-plot of datasets info",
                      #   text = selected_datasets.index,
                      showlegend=False,
                      scene=dict(
                          xaxis_title='size',
                          yaxis_title='classes',
                          zaxis_title='length',
                      ))

    if (chosen_dataset is None) and (clickData is not None):
        clickData = clickData['points'][0]['text']
        selected_dataset = selected_datasets.loc[selected_datasets['Dataset'] == clickData]
        clickData = None
    else:
        selected_dataset = selected_datasets.loc[selected_datasets['Dataset'] == chosen_dataset]
        chosen_dataset = None

    # selected_dataset.to_pickle('demoapps/Demo_page/data/selected_dataset.pkl')
    dataset_name = selected_dataset["Dataset"].to_list()

    if not dataset_name:  # list is empty
        # selected_datasets_name = selected_datasets["Dataset"].to_list()
        selected_dataset = selected_datasets.loc[selected_datasets['Dataset'] == "Chinatown"]
        dataset_name = selected_dataset["Dataset"].to_list()
        dataset_name = dataset_name[0]
    else:
        dataset_name = dataset_name[0]

    dataset_table = dash_table.DataTable(
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": False} for i in selected_dataset.columns
        ],
        style_cell={'textAlign': 'left'},
        data=selected_dataset.to_dict('records')
    )

    # with open('filename.pkl', 'wb') as handle:
    #     pickle.dump(dataset_name, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(dataset_name, open('demoapps/demo_page/data/dataset_name.pkl', 'wb'))

    link_out = f"https://www.timeseriesclassification.com/description.php?Dataset={dataset_name}"
    # link_out = f'For more information about the dataset: [Click here](https://www.timeseriesclassification.com/description.php?Dataset={dataset_name})'
    options = [{"label": i, "value": i} for i in selected_datasets['Dataset']]

    return fig, table, options, dataset_table, chosen_dataset, link_out


# f'For more information about the dataset: [Click here](https://www.timeseriesclassification.com/description.php?Dataset={dataset_name})'
###=========================================================================================
###=========================================================================================
# Mouna's part Visualization
@app.callback(
    Output(component_id='multi-dropdown', component_property='options'),
    # Output(component_id='multi-dropdown', component_property='value'),
    # Input('3d-plot', 'clickData'),
    # Input(component_id='dropdown', component_property='value')
    Input(component_id='submitval', component_property='n_clicks')
)
def update_multi_dropdown(submitval):
    selected_dataset = pickle.load(open('demoapps/demo_page/data/dataset_name.pkl', 'rb'))
    x_all, y_all = load_dataset(selected_dataset, repository='wildboar/ucr')
    pickle.dump(x_all, open('demoapps/demo_page/data/x_all.pkl', 'wb'))
    pickle.dump(y_all, open('demoapps/demo_page/data/y_all.pkl', 'wb'))
    pd.options.plotting.backend = "plotly"
    df = pd.DataFrame(x_all, y_all)
    values = df.index.unique().sort_values()
    return values


## Updating figure from selected dataset and class
@app.callback(
    Output(component_id='sample', component_property='figure'),
    # Output(component_id='dataset-link', component_property='children'),
    # Input(component_id='dropdown', component_property='value'),
    Input(component_id='multi-dropdown', component_property='value'),
    Input(component_id='button', component_property='n_clicks'),
    Input(component_id='nb-lines-slider', component_property='value')
)
def update_figure(
        # dataset,
        classes,
        clicks,
        nb_lines):
    # x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')
    # selected_dataset = pickle.load(open('demoapps/demo_page/data/dataset_name.pkl', 'rb'))
    x_all = pickle.load(open('demoapps/demo_page/data/x_all.pkl', 'rb'))
    y_all = pickle.load(open('demoapps/demo_page/data/y_all.pkl', 'rb'))
    x = np.arange(1, x_all.shape[1] + 1)
    rows = []

    if 'button' == ctx.triggered_id:
        fig = go.Figure()
        for j in range(0, len(classes)):
            current = classes[j]
            for i in range(0, len(x_all)):
                if y_all[i] == current:
                    rows.append(i)
            instance = np.random.choice(rows, nb_lines, replace=False)
            for line in range(nb_lines):
                fig.add_scatter(x=x,
                                y=x_all[instance[line]],
                                marker=
                                dict(
                                    color=colors[f"{int(current)}"]
                                ),
                                name=f"class {str(int(current))}",
                                )

    else:
        fig = go.Figure()
        for i in range(0, len(x_all)):
            if y_all[i] in classes:
                fig.add_scatter(x=x,
                                y=x_all[i],
                                marker=
                                dict(color=colors[f"{int(y_all[i])}"]
                                     ),
                                name=f"class {str(int(y_all[i]))}",
                                )

    fig.update_layout(
        title="Timeseries plot",
        xaxis=dict(title='time'),
        yaxis=dict(title='data'))
    return fig
    # , f'For more infomation about this dataset: [Click here](https://www.timeseriesclassification.com/description.php?Dataset={dataset})'


## K-MEANS
@app.callback(
    Output(component_id='kmeans-graph', component_property='figure'),
    # Input(component_id='dropdown', component_property='value'),
    Input(component_id='multi-dropdown', component_property='value'),
    Input(component_id='nb-clusters-slider', component_property='value'),

)
def clustering(
        # dataset,
        classes,
        nb_clusters):
    # x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')
    x_all = pickle.load(open('demoapps/demo_page/data/x_all.pkl', 'rb'))
    y_all = pickle.load(open('demoapps/demo_page/data/y_all.pkl', 'rb'))
    fig = go.Figure()
    x = np.arange(1, x_all.shape[1] + 1)
    data = []
    for j in range(0, len(classes)):
        current = classes[j]
        data.clear()
        for i in range(0, len(x_all)):
            if y_all[i] == current:
                data.append(x_all[i])

        km = KMeans(n_clusters=nb_clusters, random_state=0).fit(data)

        labels = km.labels_
        percentages = []
        total = len(labels)
        for i in range(nb_clusters):
            count = labels.tolist().count(i)
            percentages.append(round(((count / total) * 100)))

        centroids = km.cluster_centers_
        for i in range(len(centroids)):
            fig.add_scatter(x=x,
                            y=centroids[i],
                            mode='lines+markers',
                            marker=
                            dict(
                                symbol=markers[str(i + 1)],
                                size=6,
                                color=colors[str(current)],
                                line=dict(
                                    color=colors[str(current)],
                                    width=1
                                )
                            ),
                            name=f"class {str(int(current))} ({i + 1}) {percentages[i]}%",
                            )

    fig.update_layout(title="K-means Clustering Timeseries plot",
                      xaxis=dict(title='time'),
                      yaxis=dict(title='data'))
    return fig


###===============================        FREQUENCY DOMAIN           ========================================================
###==========================================================================================================================


## Updating dropdown for classes
@app.callback(
    Output(component_id='multi-dropdown-fd', component_property='options'),
    # Output(component_id='multi-dropdown-fd', component_property='value'),
    # Input('3d-plot', 'clickData'),
    # Input(component_id='dropdown', component_property='value'),
    Input(component_id='submitval', component_property='n_clicks'),
)
def update_multi_dropdown(submitval):
    # selected_dataset = pickle.load(open('demoapps/demo_page/data/dataset_name.pkl', 'rb'))
    x_all = pickle.load(open('demoapps/demo_page/data/x_all.pkl', 'rb'))
    y_all = pickle.load(open('demoapps/demo_page/data/y_all.pkl', 'rb'))
    # x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')
    pd.options.plotting.backend = "plotly"
    df = pd.DataFrame(x_all, y_all)
    values = df.index.unique().sort_values()
    return values


## Updating amplitude and angle figures
@app.callback(
    Output(component_id='amplitude-phase-plot', component_property='figure'),
    # Input(component_id='submitval', component_property='n_clicks'),
    # Input(component_id='dropdown', component_property='value'),
    Input(component_id='multi-dropdown-fd', component_property='value'),
    Input(component_id='button-fd', component_property='n_clicks'),
    Input(component_id='nb-lines-slider-fd', component_property='value')
)
def update_figure(
        # submitval,
        # dataset,
        classes, clicks, nb_lines
):
    # x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')
    # selected_dataset = pickle.load(open('demoapps/demo_page/data/dataset_name.pkl', 'rb'))
    x_all = pickle.load(open('demoapps/demo_page/data/x_all.pkl', 'rb'))
    y_all = pickle.load(open('demoapps/demo_page/data/y_all.pkl', 'rb'))
    btn_id = 'button-fd'
    freq_domains = np.fft.rfft(x_all)
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.07,
                        )
    # add amplitude trace
    amplitude = np.absolute(freq_domains)
    nb_row = 1
    fig = update_plot(amplitude, y_all, classes, btn_id, nb_lines, fig, nb_row)

    # add angle trace
    phase = np.angle(freq_domains)
    nb_row = 2
    fig = update_plot(phase, y_all, classes, btn_id, nb_lines, fig, nb_row)

    fig.update_layout(title=" Amplitude and angle subplots",
                      yaxis=dict(title='amplitude'),
                      xaxis2=dict(title='index'),
                      yaxis2=dict(title='phase'))
    return fig


## K-MEANS
@app.callback(
    Output(component_id='kmeans-graph-fd', component_property='figure'),
    # Input(component_id='dropdown', component_property='value'),
    Input(component_id='multi-dropdown-fd', component_property='value'),
    Input(component_id='nb-clusters-slider-fd', component_property='value'),
)
def clustering(
        # dataset,
        classes, nb_clusters):
    # selected_dataset = pickle.load(open('demoapps/demo_page/data/dataset_name.pkl', 'rb'))
    x_all = pickle.load(open('demoapps/demo_page/data/x_all.pkl', 'rb'))
    y_all = pickle.load(open('demoapps/demo_page/data/y_all.pkl', 'rb'))
    # x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')
    freq_domains = np.fft.rfft(x_all)
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.07, )

    amplitude = np.absolute(freq_domains)
    nb_row = 1
    fig = kmeans_clustering(amplitude, y_all, classes, nb_clusters, fig, nb_row)

    phase = np.angle(freq_domains)
    nb_row = 2
    fig = kmeans_clustering(phase, y_all, classes, nb_clusters, fig, nb_row)
    fig.update_layout(title="K-means amplitude and angle subplots",
                      yaxis=dict(title='amplitude'),
                      xaxis2=dict(title='index'),
                      yaxis2=dict(title='phase')),
    return fig


###=========================================================================================
###=========================================================================================
### In this section we have classifiers performance 

@app.callback(
    Output(component_id="max_len_coef_DFT", component_property="max"),
    Output(component_id="max_len_coef_DWT", component_property="max"),
    Output(component_id="len_DFT", component_property="style"),
    Output(component_id="len_DWT", component_property="style"),
    Output(component_id="knn_metric", component_property="style"),
    Output(component_id="rsf_metric", component_property="style"),
    Output(component_id="acc_table", component_property="children"),
    Output(component_id="comp_table", component_property="children"),
    Output(component_id="Dataset_name", component_property="children"),
    [Input(component_id='submitval', component_property='n_clicks'),
     Input(component_id='splits', component_property='value'),
     Input(component_id='classifiers', component_property='value'),
     Input(component_id='max_len_coef_DFT', component_property='value'),
     Input(component_id='max_len_coef_DWT', component_property='value'),
     Input(component_id='knn_metric_dropdown', component_property='value'),
     Input(component_id='rsf_metric_dropdown', component_property='value'),
     Input(component_id='split_size', component_property='value'),
     Input(component_id='shapelet_size', component_property='value'),
     Input(component_id='max_depth', component_property='value'),
     Input(component_id='n_estimator', component_property='value'),
     ]
)
def update_selected_datasets(submitval, splits, classifiers,
                             max_len_coef_DFT, max_len_coef_DWT,
                             knn_metric_dropdown, rsf_metric_dropdown,
                             split_size, shapelet_size,
                             max_depth, n_estimator):
    # with open('filename.pkl', 'rb') as handle:
    #     selected_dataset = pickle.load(handle)
    selected_dataset = pickle.load(open('demoapps/demo_page/data/dataset_name.pkl', 'rb'))
    # x_all, y_all = load_dataset(selected_dataset, repository='wildboar/ucr')
    x_all = pickle.load(open('demoapps/demo_page/data/x_all.pkl', 'rb'))
    y_all = pickle.load(open('demoapps/demo_page/data/y_all.pkl', 'rb'))
    # pickle.dump(x_all, open('demoapps/demo_page/data/x_all.pkl', 'wb'))
    # pickle.dump(y_all, open('demoapps/demo_page/data/y_all.pkl', 'wb'))
    # x = x_all[~np.isnan(x_all).any(axis=1)]

    # total_examples, ts_length = x.shape

    # max_DFT = int(ts_length/2 + 1) # maximum number of DFT coefficients
    # max_DWT = ts_length # maximum number of DWT coefficients
    # classifiers_performance(x_all, y_all, len_coef_DFT, len_coef_DWT, classifiers, splits, knn_metric,
    #                         rsf_metric, split_size, shapelet_size):

    acc, comp, max_DFT, max_DWT = classifiers_performance(x_all, y_all,
                                                          max_len_coef_DFT, max_len_coef_DWT,
                                                          classifiers,
                                                          splits,
                                                          knn_metric_dropdown, rsf_metric_dropdown,
                                                          split_size,
                                                          shapelet_size,
                                                          max_depth, n_estimator)
    acc = acc.round(decimals=3)
    comp = comp.round(decimals=3)

    t_acc = dash_table.DataTable(
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": False} for i in acc.columns
        ],
        style_cell={'textAlign': 'left'},
        data=acc.to_dict('records')
    )

    t_comp = dash_table.DataTable(
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": False} for i in comp.columns
        ],
        style_cell={'textAlign': 'left'},
        data=comp.to_dict('records')
    )

    if 'DFT' in classifiers:
        dft_style = {'display': 'block'}
    else:
        dft_style = {'display': 'none'}
    if 'DWT' in classifiers:
        dwt_style = {'display': 'block'}
    else:
        dwt_style = {'display': 'none'}
    if 'kNN' in classifiers:
        knn_metric = {'display': 'block'}
    else:
        knn_metric = {'display': 'none'}
    if 'RSF' in classifiers:
        rsf_metric = {'display': 'block'}
    else:
        rsf_metric = {'display': 'none'}

    return max_DFT, max_DWT, dft_style, dwt_style, knn_metric, rsf_metric, t_acc, t_comp, f'The Dataset that we run the models on is: \"{selected_dataset}\"'


###=========================================================================================
###=========================================================================================
## New Counterfactuals
@app.callback(
    Output(component_id="data_slider", component_property="max"),
    Output(component_id="example_class", component_property="options"),
    Output(component_id="desired_class", component_property="options"),
    Output(component_id="counterfactuals", component_property="figure"),
    Output(component_id="Dataset_name1", component_property="children"),
    [
        Input(component_id='knn_metric_dropdown', component_property='value'),
        Input(component_id='rsf_metric_dropdown', component_property='value'),
        Input(component_id='split_size', component_property='value'),
        Input(component_id='shapelet_size', component_property='value'),
        Input(component_id='max_depth', component_property='value'),
        Input(component_id='submitval', component_property='n_clicks'),
        # Input(component_id='counter_metric', component_property='value'),
        Input(component_id='example_class', component_property='value'),
        Input(component_id='desired_class', component_property='value'),
        Input(component_id='data_slider', component_property='value'),
        # Input(component_id='btn_nclicks_1', component_property='n_clicks'),
    ])
def counter(
        knn_metric_dropdown, rsf_metric_dropdown,
        split_size,
        shapelet_size, max_depth,
        # counter_metric,
        submitval,
        example_class, desired_class, data_slider,
        # btn_nclicks_1
):
    # with open('demoapps/demo_page/data/filename.pkl', 'rb') as handle:
    selected_dataset = pickle.load(open('demoapps/demo_page/data/dataset_name.pkl', 'rb'))

    # x_all, y_all = load_dataset(selected_dataset, repository='wildboar/ucr')

    x_all = pickle.load(open('demoapps/demo_page/data/x_all.pkl', 'rb'))
    y_all = pickle.load(open('demoapps/demo_page/data/y_all.pkl', 'rb'))
    # remove rows with missing values
    x = x_all[~np.isnan(x_all).any(axis=1)]
    y = y_all[~np.isnan(x_all).any(axis=1)]

    classes = np.unique(y)  # all class labels
    total_examples, ts_length = x.shape
    x_ind = np.arange(total_examples)
    x_train_ind, x_test_ind, y_train, y_test = train_test_split(x_ind, y, test_size=split_size, random_state=0,
                                                                shuffle=True,
                                                                stratify=None)
    x_train = x[x_train_ind, :]
    x_test = x[x_test_ind, :]

    y_train[y_train == example_class] = -2.0
    y_test[y_test == example_class] = -2.0
    example_class = -2.0

    y_train[(y_train == desired_class) & (y_train != example_class)] = -1.0
    y_test[(y_test == desired_class) & (y_test != example_class)] = -1.0

    x_train = np.delete(x_train, np.argwhere((y_train != -1.0) & (y_train != example_class)), axis=0)
    x_test = np.delete(x_test, np.argwhere((y_test != -1.0) & (y_test != example_class)), axis=0)

    y_train = np.delete(y_train, np.argwhere((y_train != -1.0) & (y_train != example_class)))
    y_test = np.delete(y_test, np.argwhere((y_test != -1.0) & (y_test != example_class)))

    y_train[y_train == example_class] = 1.0
    y_test[y_test == example_class] = 1.0

    clf_kNN = KNeighborsClassifier(metric=knn_metric_dropdown)
    clf_kNN.fit(x_train, y_train)

    # clf_RSF = ShapeletForestClassifier(n_estimators=50, metric='scaled_euclidean')
    clf_RSF = ShapeletForestClassifier(
        n_estimators=20,
        metric=rsf_metric_dropdown,
        max_depth=max_depth,
        max_shapelet_size=shapelet_size,  # INTERACTION: Make this as input from user
        random_state=1,
    )

    clf_RSF.n_features_in_ = x_train.shape[-1]
    clf_RSF.fit(x_train, y_train)

    clf_RFC = RandomForestClassifier()
    clf_RFC.fit(x_train, y_train)
    kwargs = {"background_x": x_train, "background_y": y_train}

    x_counterfactual_kNN, x_valid_kNN, x_score_kNN = counterfactuals(
        clf_kNN,
        x_test,
        -y_test,  # invert the classes, i.e., transform 1 -> -1 and -1 -> 1
        scoring="euclidean",
        valid_scoring=False,
    )
    x_counterfactual_RSF, x_valid_RSF, x_score_RSF = counterfactuals(
        clf_RSF,
        x_test,
        -y_test,  # invert the classes, i.e., transform 1 -> -1 and -1 -> 1
        scoring="euclidean",
        valid_scoring=False,
        random_state=2,
        epsilon=1,
    )

    x_counterfactual_RFC, x_valid_RFC, x_score_RFC = counterfactuals(
        clf_RFC,
        x_test,
        -y_test,  # invert the classes, i.e., transform 1 -> -1 and -1 -> 1
        scoring="euclidean",
        valid_scoring=False,
        random_state=2,
        **kwargs,
    )

    max_data_slider = x_counterfactual_RSF.shape[0]
    sel_instance_idx = data_slider
    x_counter_RSF = x_counterfactual_RSF[sel_instance_idx, :]
    x_counter_kNN = x_counterfactual_kNN[sel_instance_idx, :]
    x_counter_RFC = x_counterfactual_RFC[sel_instance_idx, :]
    x_counter_test = x_test[sel_instance_idx, :]
    y1 = x_counter_RSF
    y2 = x_counter_kNN
    y3 = x_counter_RFC
    y4 = x_counter_test

    df = pd.DataFrame(dict(
        length=np.arange(y1.shape[0]),
        RSF_CF=y1,
        kNN_CF=y2,
        Prototype_CF=y3,
        Data_Point=y4))

    fig = px.line(df, x=df["length"], y=df.columns)
    options = [{"label": i, "value": i} for i in classes]

    return max_data_slider, options, options, fig, f'The Dataset that we run the models on is: \"{selected_dataset}\"'


###=========================================================================================
###=========================================================================================
## Feature importance and interval importance
# clf_kNN = pickle.load(open('demoapps/demo_page/data/clf_kNN.pkl', 'rb'))
# clf_RSF = pickle.load(open('demoapps/demo_page/data/clf_RSF.pkl', 'rb'))

@app.callback(
    Output(component_id="feature_importance_DFT", component_property="figure"),
    Output(component_id="feature_importance_DWT", component_property="figure"),
    [
        Input(component_id='n_features_DFT', component_property='value'),
        Input(component_id='n_features_DWT', component_property='value'),
        #     Input(component_id='rsf_metric_dropdown', component_property='value'),
    ])
def feature_importance(n_features_DFT, n_features_DWT):
    xgbc_DFT = pickle.load(open('demoapps/demo_page/data/xgbc_DFT.pkl', 'rb'))
    xgbc_DWT = pickle.load(open('demoapps/demo_page/data/xgbc_DWT.pkl', 'rb'))
    X_F_train_DFT = pickle.load(open('demoapps/demo_page/data/X_F_train_DFT.pkl', 'rb'))
    X_F_train_DWT = pickle.load(open('demoapps/demo_page/data/X_F_train_DWT.pkl', 'rb'))
    ###------ DFT
    explainer = shap.explainers.Tree(xgbc_DFT, X_F_train_DFT)
    shap_values = explainer(X_F_train_DFT)
    # shap_values = shap_values
    # print(shap_values)
    # print(np.mean(np.absolute(shap_values.values - np.mean(shap_values.values, axis=0)), axis=0))
    values = np.mean(np.absolute(shap_values.values - np.mean(shap_values.values, axis=0)), axis=0)
    indexes = [f"Feature {i}" for i in range(len(values))]
    df = pd.DataFrame({"Shap_Value": values, "Feature": indexes})
    df.sort_values(by=["Shap_Value"], inplace=True, ascending=True)
    # df.reset_index(inplace=True)
    # print(df.tail())
    # ax = df[-10:].plot.barh(y='Shap_Value')
    fig_DFT = px.bar(df[-n_features_DFT:], x="Shap_Value", y="Feature", orientation='h')
    ###------ DWT
    explainer = shap.explainers.Tree(xgbc_DWT, X_F_train_DWT)
    shap_values = explainer(X_F_train_DWT)
    # shap_values = shap_values
    # print(shap_values)
    # print(np.mean(np.absolute(shap_values.values - np.mean(shap_values.values, axis=0)), axis=0))
    values = np.mean(np.absolute(shap_values.values - np.mean(shap_values.values, axis=0)), axis=0)
    indexes = [f"Feature {i}" for i in range(len(values))]
    df = pd.DataFrame({"Shap_Value": values, "Feature": indexes})
    df.sort_values(by=["Shap_Value"], inplace=True, ascending=True)
    # df.reset_index(inplace=True)
    # print(df.tail())
    # ax = df[-10:].plot.barh(y='Shap_Value')
    fig_DWT = px.bar(df[-n_features_DWT:], x="Shap_Value", y="Feature", orientation='h')

    return fig_DFT, fig_DWT


###=========================================================================================
###=========================================================================================
if __name__ == "__main__":
    app.run_server(debug=True)
