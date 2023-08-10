###=========================================================================================
# Import the libraries and then add the path.

import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import os
import plotly.graph_objects as go
from pathlib import Path
import sys

import numpy as np
import pandas as pd
#import random
import time

import matplotlib.pyplot as plt
plt.style.reload_library()

from sktime.classification.interval_based import RandomIntervalSpectralEnsemble
from sktime.classification.dictionary_based import ContractableBOSS, BOSSEnsemble 
from sktime.datatypes._panel._convert import from_2d_array_to_nested


from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV

from wildboar.datasets import list_datasets, load_dataset
from wildboar.ensemble import ShapeletForestClassifier

import pywt
from pywt import wavedec, waverec

from tqdm.notebook import tqdm

import pickle

from sktime.transformations.panel.rocket import Rocket

from extremum_config import app
THIS_FILEPATH = os.path.dirname(__file__)
sys.path.append(os.path.join(THIS_FILEPATH, "."))

###=========================================================================================
###=========================================================================================
### LOAD DATASETS and store them in a pickle file whenever you start the page

# UCR_datasets = list_datasets(repository='wildboar/ucr')

# dataset_info = pd.DataFrame(columns=['size', 'classes', 'length'], index = UCR_datasets, dtype=float)

# for dataset in tqdm(UCR_datasets):
#     x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')
    
#     # remove rows wirandomth missing values
#     x = x_all[~np.isnan(x_all).any(axis=1)]
#     y = y_all[~np.isnan(x_all).any(axis=1)]
    
#     classes = np.unique(y) # all class labels
#     total_examples, ts_length = x.shape
    
#     dataset_info.loc[dataset] = [total_examples, len(classes), ts_length]

# ## Drop rows that have a value of zero in a column
# dataset_info = dataset_info.loc[~(dataset_info==0).any(axis=1)]
# dataset_info = dataset_info.loc[~(dataset_info==1).any(axis=1)]

# dataset_info.to_pickle('demoapps/Demo_page/data/datasets_information.pkl')
# dataset_info = pd.read_pickle('demoapps/Demo_page/data/datasets_information.pkl')

# def load_datasets(path):
    # 'demoapps/Demo_page/data/datasets_information.pkl'
dataset_info = pd.read_pickle('demoapps/Demo_page/data/datasets_information.pkl')
dataset_info.reset_index(inplace=True)
dataset_info.rename(columns = {'index':'Dataset'}, inplace = True)
# selected_datasets = dataset_info.copy()
print("Read the dataSets and reset the indexes")
    # return dataset_info




###=========================================================================================
###=========================================================================================
## Select DATASETS in a function based on inputs size, length, classes 

def filter_dataset(sizes, classes, lengths):
    """
    The min and max of each variable (after loading and droping NAN and zero and one values) is:
    lb_size = 40
    up_size = 24000
    lb_length = 15
    up_length = 2844
    lb_class = 2
    up_class = 60
    """
    lb_size = sizes[0]
    up_size = sizes[1]
    lb_length = lengths[0]
    up_length = lengths[1]
    lb_class = classes[0]
    up_class = classes[1]
    # dataset_info = pd.read_pickle('demoapps/Demo_page/data/datasets_information.pkl')
    # 'demoapps/Demo_page/data/datasets_information.pkl'
    selected_datasets = dataset_info.loc[(dataset_info['size'] >= lb_size) & \
                           (dataset_info['size'] <= up_size) & \
                            (dataset_info['classes'] >= lb_class) & \
                            (dataset_info['classes'] <= up_class) & \
                           (dataset_info['length'] >= lb_length) & \
                           (dataset_info['length'] <= up_length)]
    return selected_datasets

###=========================================================================================
###=========================================================================================
## Classifiers performance (Arvin)

classifiers_lst = ["kNN","RSF","Rocket","RISE","BOSS","DFT","DWT"]
# print(classifiers_lst)
# print(type(classifiers_lst))
#classifiers = ["kNN","RSF","Rocket","RISE","BOSS","DFT","DWT"]
def classifiers_performance(x_all, y_all, len_coef_DFT, len_coef_DWT, classifiers, splits):
    # classifiers = ["kNN"]

    # selected_dataset = pd.read_pickle('demoapps/Demo_page/data/selected_dataset.pkl')
    # selected_dataset = selected_dataset["Dataset"].to_list()
    # # splits = 1
    # selected_dataset = selected_dataset[0]

    # iterables = [[selected_dataset], np.arange(splits)]
    # m_index = pd.MultiIndex.from_product(iterables, names=["dataset", "split"])
    m_index = np.arange(splits)
    # print("The indexes are printed: ", m_index)
    accuracies = pd.DataFrame(index = m_index, columns=classifiers, dtype=float)
    complexity = pd.DataFrame(index = m_index, columns=classifiers, dtype=float)
    # print("accuracies: ", accuracies)
    # print("complexity:", complexity)
    # x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')

    x_all, y_all = x_all, y_all
    
    x = x_all[~np.isnan(x_all).any(axis=1)]
    y = y_all[~np.isnan(x_all).any(axis=1)]

    classes = np.unique(y) # all class labels
    total_examples, ts_length = x.shape

    x_ind = np.arange(total_examples)
    

    max_len_coef_DFT = int(ts_length/2 + 1) # maximum number of DFT coefficients
    max_len_coef_DWT = ts_length # maximum number of DWT coefficients

    # for i in tqdm(range(splits), desc = selected_dataset, leave=False):
    for i in range(splits):
        # implement same split across all
        np.random.seed(i)
        
        x_train_ind, x_test_ind, y_train, y_test = train_test_split(x_ind, y, test_size=.30, 
                                                                    random_state=i, shuffle=True, stratify=None)
        x_train2_ind, x_val_ind, y_train2, y_val = train_test_split(x_train_ind, y_train, test_size=.20,
                                                                    random_state=i, shuffle=True, stratify=None)


        x_train = x[x_train_ind,:]
        x_test = x[x_test_ind,:]

        np.random.seed(i)

        ## kNN
        if 'kNN' in classifiers:

            knn_time_start = time.time()

            clf_kNN = KNeighborsClassifier(metric="euclidean")
            clf_kNN.fit(x_train, y_train)
            acc = clf_kNN.score(x_test,y_test)

            knn_time_end = time.time()

            accuracies.loc[i, "kNN"] = acc
            complexity.loc[i, "kNN"] = knn_time_end - knn_time_start
        # ----------


        ## RSF
        if 'RSF' in classifiers:
            rsf_time_start = time.time()

            clf_RSF = ShapeletForestClassifier(n_estimators=50, metric='scaled_euclidean')
            clf_RSF.n_features_in_ = x_train.shape[-1]
            clf_RSF.fit(x_train, y_train)
            acc = clf_RSF.score(x_test,y_test)

            rsf_time_end = time.time()
            accuracies.loc[i, "RSF"] = acc
            complexity.loc[i, "RSF"] = rsf_time_end - rsf_time_start

        # ----------


        ## ROCKET
        if 'Rocket' in classifiers:

            rocket_time_start = time.time()

            rocket = Rocket(num_kernels=20000, normalise=False, n_jobs=1, random_state=None)  # by default, ROCKET uses 10,000 kernels
            rocket.fit(np.expand_dims(x_train, axis=1))

            # transform training set and train classifier
            x_training_transform = rocket.transform(np.expand_dims(x_train, axis=1))

            classifierRocket = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas = np.logspace(-3, 3, 10)))


            classifierRocket.fit(x_training_transform, y_train)

            # transform test set and predict
            x_test_transform = rocket.transform(np.expand_dims(x_test, axis=1))
            # predictions = classifierRocket.predict(x_test_transform)
            acc = classifierRocket.score(x_test_transform,y_test)

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
            acc = classifierRISE.score(x_test_nested,y_test)
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
            acc = classifierBOSS.score(x_test_nested,y_test)
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
            X_F_train = np.empty((x_train2_ind.shape[0],2*max_len_coef_DFT))
            X_F_val = np.empty((x_val_ind.shape[0],2*max_len_coef_DFT))
            X_F_test = np.empty((x_test.shape[0],2*max_len_coef_DFT))

            X_F_train[:,0::2] = normalize(np.absolute(X_F[x_train2_ind,:]), axis=1)  # amplitude
            X_F_train[:,1::2] = normalize(np.angle(X_F[x_train2_ind,:]), axis=1)     # phase
            X_F_val[:,0::2] = normalize(np.absolute(X_F[x_val_ind,:]), axis=1)  # amplitude
            X_F_val[:,1::2] = normalize(np.angle(X_F[x_val_ind,:]), axis=1)     # phase
            X_F_test[:,0::2] = normalize(np.absolute(X_F[x_test_ind,:]), axis=1)  # amplitude
            X_F_test[:,1::2] = normalize(np.angle(X_F[x_test_ind,:]), axis=1)     # phase

            #fracs = [.1,.25,.5,.75,1]
            #acc_val = np.empty(len(fracs))
            #acc_test = np.empty(len(fracs))

#             for idx, L_DFT_frac in enumerate(fracs):

#                 L_DFT = int(L_DFT_frac*max_len_coef_DFT)
#                 clf = ExtraTreesClassifier(random_state=1)
#                 clf.fit(X_F_train[:,:2*L_DFT], y_train2)
#                 acc_val[idx] = clf.score(X_F_val[:,:2*L_DFT], y_val)
#                 acc_test[idx] = clf.score(X_F_test[:,:2*L_DFT], y_test)

            L_DFT = int(len_coef_DFT)
            clf = ExtraTreesClassifier(random_state=1)
            clf.fit(X_F_train[:,:2*L_DFT], y_train2)
            acc_val = clf.score(X_F_val[:,:2*L_DFT], y_val)
            acc_test = clf.score(X_F_test[:,:2*L_DFT], y_test)
            
            dft_time_end = time.time()

            # accuracies.loc[i, "DFT"] = acc_test[np.argmax(acc_val)]
            
            accuracies.loc[i, "DFT"] = acc_test
            complexity.loc[i, "DFT"] = dft_time_end - dft_time_start

        # --------------
        ## Changing the number of lower frequencies features using a sliding bar

        if 'DWT' in classifiers:

            dwt_time_start = time.time()

            # Time-Freq domain
            level=3 # no of DWT decomposition levels 
            waveletname='db2'

            X_TF = wavedec(x, waveletname, level=level) ## If using DWT
            X_TF_stacked = np.hstack(X_TF)

            X_TF_train = X_TF_stacked[x_train2_ind,:]
            X_TF_val = X_TF_stacked[x_val_ind,:]
            X_TF_test = X_TF_stacked[x_test_ind,:]

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
            clf = ExtraTreesClassifier(random_state=1)
            clf.fit(X_TF_train[:,:L_DWT], y_train2)
            acc_val = clf.score(X_TF_val[:,:L_DWT], y_val)
            acc_test = clf.score(X_TF_test[:,:L_DWT], y_test)   
            
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
            
    accuracies.reset_index(inplace=True)
    accuracies.rename(columns = {'index':'split'}, inplace = True)
    complexity.reset_index(inplace=True)
    complexity.rename(columns = {'index':'split'}, inplace = True)
    return accuracies, complexity, max_len_coef_DFT, max_len_coef_DWT

###=========================================================================================
###=========================================================================================
# We have the entire layout here 

layout = html.Div(children=[
    html.Div(
                            id="range_sliders",
                            children=[
                                dcc.Markdown('The size, classes, length'),
                                html.H5('Sizes Range'),
                                dcc.RangeSlider(id='sizes',
                                                min=40,
                                                max=24000,
                                                step=40,
                                                marks={
                                                    # 137: {"label": '0.1', "style": {"transform": "rotate(-90deg)"}},
                                                    # 200: {"label": '0.2', "style": {"transform": "rotate(-90deg)"}},
                                                    # 312: {"label": '0.3', "style": {"transform": "rotate(-90deg)"}},
                                                    # 462: {"label": '0.4', "style": {"transform": "rotate(-90deg)"}},
                                                    724: {"label": '0.5'},
                                                    # 888: {"label": '0.6', "style": {"transform": "rotate(-90deg)"}},
                                                    1272: {"label": '0.75'},
                                                    2250: {"label": '0.8'},
                                                    4362: {"label": '0.9'}
                                                   },
                                                value=[40,24000],
                                                tooltip={"placement": "bottom", "always_visible": True}, 
                                                allowCross=False
                                           ),
                                html.Br(),
                                html.Br(),
                                html.H5('Classes Range'),
                                dcc.RangeSlider(id='classes',
                                                min=2,
                                                max=60,
                                                step=2,
                                                marks={
                                                    2: '0.3',
                                                    3: '0.5',
                                                    5: '0.6',
                                                    7: '0.75',
                                                    10: '0.8',
                                                    16: '0.9'},
                                                value=[2,60],
                                                tooltip={"placement": "bottom", "always_visible": True},
                                                allowCross=False
                                           ),
                                html.Br(),
                                html.Br(),
                                html.H5('Lenghts Range'),
                                dcc.RangeSlider(id='lengths',
                                                min=15,
                                                max=2844,
                                                step=69,
                                                marks={
                                                    80: '0.1',
                                                    # 128: '0.2',
                                                    150: '0.3',
                                                    # 270: '0.4',
                                                    301: '0.5',
                                                    441: '0.6',
                                                    571: '0.7',
                                                    870: '0.8',
                                                    1476: '0.9'},
                                                value=[15,2844],
                                                tooltip={"placement": "bottom", "always_visible": True},
                                                allowCross=False
                                           ),
                                html.Br(),
                                html.Br(),

                            ],),
    # html.Div([
    #     dash_table.DataTable(dataset_info.to_dict('records'),[{"name": i, "id": i} for i in dataset_info.columns], id='tbl'),
    # ],
    # ),
    html.Div(id ='datatable-filtered'),
    html.Br(),
    html.Br(),
    html.Div(id="dropdown_dataset",
                            children=[
                                html.Label('Select Dataset'),
                                dcc.Dropdown(
                                    id="choosen_dataset",
                                    clearable=True,
                                    searchable=True,
                                        ),
                                # dcc.Graph(id ='dataset')
                                html.Div(id ='dataset'),
                            ],),
    html.Br(),
    html.Button('Submit', id='submitval', n_clicks=0, style={'flush': 'right', 'padding': '10px', 'float': 'left'},),
      
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div(id='Performance_Discription',
             children = [
                 dcc.Markdown('Classifires performance (Arvin)'
                             'We have the following tasks: '),
                 dcc.Markdown('1. Split the dataset into training and test data: we split by a slide from 1 to 10'),
                 dcc.Markdown('2. Train a set of time series classifiers on the training data we have a multi select dropdown menu to select classifires'),
                 dcc.Markdown('3. DWT model we change the number of lower frequencies features using a sliding bar.'
                             'For this we have to change the maximum according to the selected dataset'),
                 dcc.Markdown('4. DFT model we change the number of lower frequencies features using a sliding bar.'
                             'For this we have to change the maximum according to the selected dataset'),
                 dcc.Markdown('5. Random Shapeless Forrest (RSF)'),
                 dcc.Markdown('6. kNN L2 norm'),
                 dcc.Markdown('7. Rocket '),
                 dcc.Markdown('8. RISE'),
                 dcc.Markdown('9. BOSS'),
             ],                                
             style={"float": "left", "width": '100%', "margin": '.5%', 'border': '0 lightgrey solid',
                                        'display': 'inline-block', 'height': 'auto',
                                        "padding": 10, "borderRadius": 5, 'flex': 0},
            ),
    html.Br(),
    html.Br(),
    html.Div(children = [
        html.Label("1. Number of splits: Times we randomly shuffle the data in our test and training dataset."),
            html.Br(),
        dcc.Slider(id='splits',
                   min=1,
                   max=10,
                   step=1,
                   value=2,
                   tooltip={"placement": "bottom", "always_visible": True},
                   ),
        html.Br(),
        html.Label('2. classifiers to choose from'),
        dcc.Dropdown(
            # classifiers_lst = ["kNN","RSF","Rocket","RISE","BOSS","DFT","DWT"]
            id="classifiers",
            options=[
                {'label': i, 'value': i} for i in classifiers_lst
            ],
            clearable=True,
            multi=True,
            value=["DFT", "DWT"],
                ),
        html.Br(),
        html.Div(children=[
        html.Label('3. len_coef_DFT: Changing the number of lower frequencies features using a sliding bar'),
        dcc.Slider(id='max_len_coef_DFT',
           min=1,
           step=1,
           value=1,
           tooltip={"placement": "bottom", "always_visible": True},
           ),
        ], id='len_DFT', style= {'display': 'block'} ),
        html.Br(),
        html.Div(children=[
        html.Label('4. len_coef_DWT: Changing the number of lower frequencies features using a sliding bar'),
        dcc.Slider(id='max_len_coef_DWT',
           min=1,
           step=1,
           value=1,
           tooltip={"placement": "bottom", "always_visible": True},
           ),
            ], id='len_DWT', style= {'display': 'block'}),
            ]
),
    html.Br(),
    html.Br(),
    # html.Label('Dataset Name: '),
    html.Div(id= "Dataset_name",),
    html.Br(),
    html.Br(),
    html.Label('Accuracy Table'),
    html.Div(id= "acc_table",),
    html.Br(),
    html.Label('Complexity Table'),
    html.Div(id= "comp_table",),
    html.Br(),
    html.Br(),
],)

###=========================================================================================
###=========================================================================================
## We have the load dataset for rangeSliders callback and function here and we update the options in our dropdown menu and then we can choose a dataset from the options.

@app.callback(
    Output(component_id='datatable-filtered', component_property='children'),
    Output("choosen_dataset", "options"),
    Output(component_id='dataset', component_property='children'),
    [Input(component_id='sizes', component_property='value'),
     Input(component_id='classes', component_property='value'),
     Input(component_id='lengths', component_property='value'),
     Input(component_id='choosen_dataset', component_property='value')])

def update_image_src(sizes, classes, lengths, choosen_dataset):
    
    selected_datasets = filter_dataset(sizes, classes, lengths)
    
    # selected_datasets.to_pickle('demoapps/Demo_page/data/selected_datasets.pkl')
    
    table = dash_table.DataTable(
        # id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in selected_datasets.columns
        ],
        data=selected_datasets.to_dict('records'),
        page_current= 0,
        page_size= 10,
    )
    
    selected_dataset = selected_datasets.loc[selected_datasets['Dataset'] == choosen_dataset]
    
    selected_dataset.to_pickle('demoapps/Demo_page/data/selected_dataset.pkl') 
    
    dataset_table = dash_table.DataTable(
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": False} for i in selected_dataset.columns
        ],
        data=selected_dataset.to_dict('records')
    )
    slst = selected_datasets['Dataset'].to_list()
    
    return table, [{"label": i, "value": i} for i in selected_datasets['Dataset']], dataset_table

###=========================================================================================
###=========================================================================================
## In here we update the options in our dropdown menu and then we can choose a dataset from the options.

# @app.callback(
#     Output("choosen_dataset", "options"),
#     Output(component_id='dataset', component_property='children'),
#     [Input(component_id='choosen_dataset', component_property='value')]
#     # Input(component_id='load_interval', component_property='n_intervals')]
# )
# def update_selected_datasets(choosen_dataset):
    
#     selected_datasets = pd.read_pickle('demoapps/Demo_page/data/selected_datasets.pkl')
    
#     selected_dataset = selected_datasets.loc[selected_datasets['Dataset'] == choosen_dataset]
    
#     selected_dataset.to_pickle('demoapps/Demo_page/data/selected_dataset.pkl') 
#     dataset_table = dash_table.DataTable(
#         columns=[
#             {"name": i, "id": i, "deletable": False, "selectable": False} for i in selected_dataset.columns
#         ],
#         data=selected_dataset.to_dict('records')
#     )
#     slst = selected_datasets['Dataset'].to_list()
    
#     return [{"label": i, "value": i} for i in selected_datasets['Dataset']], dataset_table


# def update_options(n_intervals):
#     selected_datasets = pd.read_pickle('demoapps/Demo_page/data/selected_datasets.pkl')
#     return [{"label": i, "value": i} for i in selected_datasets['Dataset']]
    
# @app.callback(
#     Output(component_id='dataset', component_property='children'),
#     Input(component_id='choosen_dataset', component_property='value'),
# ) 
###=========================================================================================
###========================================================================================
### In this section we have classifiers performance 

@app.callback(
    Output(component_id="max_len_coef_DFT", component_property="max"),
    Output(component_id="len_DFT", component_property="style"),
    Output(component_id="max_len_coef_DWT", component_property="max"),
    Output(component_id="len_DWT", component_property="style"),
    Output(component_id="acc_table", component_property="children"),
    Output(component_id="comp_table", component_property="children"),
    Output(component_id="Dataset_name", component_property="children"),
    [Input(component_id='submitval', component_property='n_clicks'),
    Input(component_id='splits', component_property='value'),
    Input(component_id='classifiers', component_property='value'),
    Input(component_id='max_len_coef_DFT', component_property='value'),
    Input(component_id='max_len_coef_DWT', component_property='value'),]
)

def update_selected_datasets(submitval, splits, classifiers, max_len_coef_DFT, max_len_coef_DWT):
    
    selected_dataset = pd.read_pickle('demoapps/Demo_page/data/selected_dataset.pkl')

    selected_dataset = selected_dataset["Dataset"].to_list()
    
    print("why does the selected dataset has a problem?")
    print(selected_dataset)
    print(type(selected_dataset))
    
    if not selected_dataset: 
        selected_datasets = pd.read_pickle('demoapps/Demo_page/data/selected_datasets.pkl')
        print(selected_datasets["Dataset"])
        selected_datasets = selected_datasets["Dataset"].to_list()
        selected_dataset = selected_datasets[0]
        print(selected_dataset)
        print('The selected dataset was empty so we assigned the first dataset in the filtered list of datasets')
    else:
        print('The selected dataset was not empty.')
        selected_dataset = selected_dataset[0]
        print(selected_dataset)
        
    print("why does the selected dataset has a problem?")
    print(selected_dataset)
    print(type(selected_dataset))
    
    # splits = 1
    # iterables = [[selected_dataset], np.arange(splits)]
    # m_index = pd.MultiIndex.from_product(iterables, names=["dataset", "split"])
    
    x_all, y_all = load_dataset(selected_dataset, repository='wildboar/ucr')

    x = x_all[~np.isnan(x_all).any(axis=1)]

    total_examples, ts_length = x.shape

    max_DFT = int(ts_length/2 + 1) # maximum number of DFT coefficients
    max_DWT = ts_length # maximum number of DWT coefficients
    
    acc, comp, max_DFT, max_DWT = classifiers_performance(x_all, y_all, max_len_coef_DFT, max_len_coef_DWT, classifiers, splits)
    
    t_acc = dash_table.DataTable(
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": False} for i in acc.columns
        ],
        data=acc.to_dict('records')
    )
    
    t_comp = dash_table.DataTable(
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": False} for i in comp.columns
        ],
        data=comp.to_dict('records')
    )
    
    if 'DFT' in classifiers:
        DFT_style = {'display': 'block'}
    else:
        DFT_style = {'display': 'none'}
        
    if 'DWT' in classifiers:
        DWT_style = {'display': 'block'}
    else:
        DWT_style = {'display': 'none'}
        
    return max_DFT, DFT_style, max_DWT, DWT_style, t_acc, t_comp, f'The Dataset Name that we run the models on is: \"{selected_dataset}\"'


###=========================================================================================
###=========================================================================================
if __name__ == "__main__":
    app.run_server(debug=True)