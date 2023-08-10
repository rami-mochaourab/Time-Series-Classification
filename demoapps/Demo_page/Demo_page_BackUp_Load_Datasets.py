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

def filter_dataset(lb_size, up_size, lb_class, up_class, lb_length, up_length):
    """
    The min and max of each variable (after loading and droping NAN and zero and one values) is:
    lb_size = 40
    up_size = 24000
    lb_length = 15
    up_length = 2844
    lb_class = 2
    up_class = 60
    """
    # dataset_info = pd.read_pickle('demoapps/Demo_page/data/datasets_information.pkl')
    # 'demoapps/Demo_page/data/datasets_information.pkl'
    selected_datasets = dataset_info.loc[(dataset_info['size'] >= lb_size) & \
                           (dataset_info['size'] <= up_size) & \
                            (dataset_info['classes'] >= lb_class) & \
                            (dataset_info['classes'] <= up_class) & \
                           (dataset_info['length'] >= lb_length) & \
                           (dataset_info['length'] <= up_length)]
    selected_datasets.to_pickle('demoapps/Demo_page/data/selected_datasets.pkl')
    names = selected_datasets['Dataset'] 
    return selected_datasets

###=========================================================================================

###=========================================================================================
## Classifiers performance (Arvin)





###=========================================================================================
###=========================================================================================
# We have the entire layout here 

layout = html.Div(children=[
    html.Div(
                            id="demo",
                            children=[
                                dcc.Markdown('The size, classes, length'),
                                html.Label('Select lower bound and upper bound size of the datasets'),
                                dcc.RangeSlider(id='sizes',
                                           min=200,
                                           max=2200,
                                           step=100,
                                           value=[200,2200],
                                           tooltip={"placement": "bottom", "always_visible": True}, 
                                                allowCross=False
                                           ),
                                html.Br(),
                                html.Br(),
                                html.Label('Select lower bound and upper bound classes of the datasets'),
                                dcc.RangeSlider(id='classes',
                                           min=2,
                                           max=60,
                                           step=2,
                                           value=[2,60],
                                           tooltip={"placement": "bottom", "always_visible": True},
                                                allowCross=False
                                           ),
                                html.Br(),
                                html.Br(),
                                html.Label('Select lower bound and upper bound length of the datasets'),
                                dcc.RangeSlider(id='lengths',
                                           min=100,
                                           max=1000,
                                           step=50,
                                           value=[100,1000],
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
    html.Div(id ='datatable-interactivity'),
    html.Br(),
    html.Br(),
    html.Div(id="demo1",
                            children=[
                                html.Label('Select Dataset'),
                                dcc.Dropdown(
                                    id="sample",
                                    # options=[
                                    #     {'label': i, 'value': i} for i in names
                                    # ],
                                    clearable=True,
                                    searchable=True,
                                    # value = 
                                        ),
                            ],
     ),
    html.Div(id ='dataset'),   
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
],)

###=========================================================================================

###=========================================================================================
## We have the load dataset for rangeSliders callback and function here 

@app.callback(
    Output(component_id='datatable-interactivity', component_property='children'),
    [Input(component_id='sizes', component_property='value'),
     Input(component_id='classes', component_property='value'),
     Input(component_id='lengths', component_property='value')])

def update_image_src(sizes, classes, lengths):
    
    lb_size = sizes[0]
    up_size = sizes[1]
    lb_length = lengths[0]
    up_length = lengths[1]
    lb_class = classes[0]
    up_class = classes[1]

    selected_datasets = filter_dataset(lb_size, up_size, lb_class, up_class, lb_length, up_length)
    
    
    table = dash_table.DataTable(
        # id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in selected_datasets.columns
        ],
        data=selected_datasets.to_dict('records'),
        page_current= 0,
        page_size= 10,
    )
    return table

###=========================================================================================

###=========================================================================================
## In here we update the options in our dropdown menu and then we can choose a dataset from the options.

@app.callback(
    Output("sample", "options"),
    Output(component_id='dataset', component_property='children'),
    Input(component_id='sample', component_property='value'),
    # Input(component_id='load_interval', component_property='n_intervals')]
)
def update_selected_datasets(sample):
    selected_datasets = pd.read_pickle('demoapps/Demo_page/data/selected_datasets.pkl')
    selected_dataset = selected_datasets.loc[selected_datasets['Dataset'] == sample]
    dataset_table = dash_table.DataTable(
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": False} for i in selected_dataset.columns
        ],
        data=selected_dataset.to_dict('records')
    )
    return [{"label": i, "value": i} for i in selected_datasets['Dataset']], dataset_table


# def update_options(n_intervals):
#     selected_datasets = pd.read_pickle('demoapps/Demo_page/data/selected_datasets.pkl')
#     return [{"label": i, "value": i} for i in selected_datasets['Dataset']]
    
# @app.callback(
#     Output(component_id='dataset', component_property='children'),
#     Input(component_id='sample', component_property='value'),
# ) 


###=========================================================================================
if __name__ == "__main__":
    app.run_server(debug=True)