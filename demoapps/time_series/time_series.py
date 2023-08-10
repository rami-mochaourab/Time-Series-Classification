from locale import D_FMT
from turtle import width
from click import style
from dash.dependencies import Input, Output
from dash import ctx
from wildboar.datasets import list_datasets, load_dataset
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from tqdm import tqdm
from pathlib import Path
from plotly.subplots import make_subplots
from extremum_config import app
from os.path import dirname
from itertools import cycle
from plotly.validators.scatter.marker import SymbolValidator

import sys
import plotly.express as px
import numpy as np
import pandas as pd
import time
import os
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pickle
THIS_FILEPATH = os.path.dirname(__file__)
sys.path.append(os.path.join(THIS_FILEPATH, "."))
sys.path.append(dirname(__file__))
from colors import colors, markers

    
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

# dataset_info.to_pickle('./demoapps/time_series/data/datasets_information.pkl')
dataset_info = pd.read_pickle('./demoapps/time_series/data/datasets_information.pkl')


def update_plot(x_all, y_all, classes, btn_id, nb_lines, fig, nb_row):
    x = np.arange(1, x_all.shape[1]+1)
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
                    x = x, 
                    y = x_all[instance[line]],
                    marker=
                        dict(
                            color=colors[f"{int(current)}"]
                            ),
                        name= f"class {str(int(current))}"),
                    row=nb_row, col=1)
    else:      
        for i in range(0, len(x_all)):   
            if y_all[i] in classes: 
                fig.add_trace(go.Scatter(
                    x = x, 
                    y = x_all[i],
                    marker=
                        dict(color=colors[f"{int(y_all[i])}"]
                            ),
                        name= f"class {str(int(y_all[i]))}"),
                    row=nb_row, col=1)
    return fig


def kmeans_clustering(x_all, y_all, classes, nb_clusters, fig, nb_row):
    x = np.arange(1, x_all.shape[1]+1)
    data = []
    for i in range(0, len(classes)):
        current = classes[i]
        data.clear()
        for j in range(0, len(x_all)):   
            if y_all[j] == current:       
                data.append(x_all[j])
                
        km =  KMeans(n_clusters=nb_clusters, random_state=0).fit(data)
        labels = km.labels_
        percentages = []
        total = len(labels)
        for l in range(nb_clusters):
            count = labels.tolist().count(l)
            percentages.append(round(((count/total)*100)))
    
        centroids = km.cluster_centers_
        for k in range(len(centroids)):
            fig.add_trace(go.Scatter(
                x=x, 
                y=centroids[k], 
                mode = 'lines+markers',
                marker=
                    dict(
                        symbol=markers[str(k+1)], 
                        size=6,
                        color=colors[f"{current}"]),
                        line=dict(
                            color=colors[str(current)],
                            width=1
                        ),
                    name= f"class {str(int(current))} ({k+1}) {percentages[k]}%",
            ), row=nb_row, col=1)
    return fig

    
    
layout = html.Div(
    children = [
        html.Div(
        html.A(
            "Timeseries",
            style={
                "text-decoration": "none",
                "color": "red",
                'font-size': 50,
            },
        )
        ),
        html.Br(),
        
        dcc.Markdown('In this page you will find 128 timeseries datasets with various numbers of classes. You can plot the timeseries data based on options that you will discover through the page.'),
        
          ## Rangeslider for Size Selection
        html.Div(
            id='size',
            children=[
                html.Label('Choose range of size'),
                dcc.RangeSlider(
                    min=min(dataset_info['size']),
                    max=max(dataset_info['size']),
                    step=2000, 
                    value=[40, 24000], 
                    id='my-size-range-slider'),
                html.Div(id='output-container-size-range-slider')
            ]
        ),
        
         ## Rangeslider for Class Selection
         html.Div(
            id='classes',
            children=[
                html.Label('Choose range of classes'),
                dcc.RangeSlider(
                    min=min(dataset_info['classes']),
                    max=max(dataset_info['classes']),
                    step=5, 
                    value=[2, 60], 
                    id='my-classes-range-slider'),
                html.Div(id='output-container-classes-range-slider')
            ]
        ),
        
          ## Rangeslider for Length Selection
        html.Div(
            id='length',
            children=[
                html.Label('Choose range of length'),
                dcc.RangeSlider(
                    min=min(dataset_info['length']),
                    max=max(dataset_info['length']),
                    step=300, 
                    value=[15, 3000], 
                    id='my-length-range-slider'),
                
                html.Br(),
            ]
        ),
        
        html.H5('Datasets'),
          ## Dropdown of selected datasets
        html.Div(
            id="datasets",
            children=[
                # html.Label('Select a dataset'),
                dcc.Dropdown(
                    options=[],
                    id="dropdown",
                    placeholder="Select a dataset",
                    # value="BME",
                    persistence=True,
                    clearable=False
                )
            ]
        ),
        
        dcc.Markdown(id='dataset-link'),
        
        ## 3D-plot
        html.Div(
            children = [
                dcc.Graph(id='3d-plot'),
                 dbc.Card(id='text-message')  
                ], 
                 style={"width": '100%', "margin": '0.5%',
                        'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                        'background-color': '#e0f0e0',
                        "padding": 10, "borderRadius": 5},  
                                   
            ),
        html.Br(),
        
        ## Multi-dropdown of classes from selected dataset
        html.H5('Classes of selected dataset'),
        html.Div(
            id="time-series",
            children=[
                html.Div(
                children=[
                    dcc.Dropdown(
                        options=[],
                        id="multi-dropdown",
                        placeholder="Select classes",
                        multi = True,
                        value=[],
                        clearable=False,
                    ),
                    html.Br(),
                    html.Label('Choose number of timeseries to plot from the slider below.'),
                    dcc.Slider(1, 10, 1, 
                               value=1, 
                               id='nb-lines-slider'),
    
                     ## Button to select new instance of time serie
                    html.Button('Plot randomly', id='button', n_clicks=0),
                ],  style={"width": '100%', "margin": '0.5%',
                        'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                        'background-color': '#e0f0e0',
                        "padding": 10, "borderRadius": 5},    
            ),
                
         
        
          ## Time-series plot
            html.Div(
                children = [
                    dcc.Graph(id='sample')
                ], 
                 style={"width": '100%', "margin": '0.5%',
                        'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                        'background-color': '#e0f0e0',
                        "padding": 10, "borderRadius": 5},                       
            ),
            html.Br(),
            
            ## K-means centroids plot
            html.H5('K-Means with sklearn'),
            html.Div(
                id="k-means",
                children = [
                    html.Label('Choose number of clusters from the slider below.'),
                    dcc.Slider(1, 10, 1, 
                               value=3, 
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
        
        #=============================== FREQUENCY DOMAIN ===============================
        ## Multi-dropdown of classes from selected dataset
        html.A(
            "Frequency domain",
            style={
                "text-decoration": "none",
                "color": "red",
                'font-size': 30,
            },
        ),
        
        html.H5('Classes of selected dataset'),
        html.Div(
            id="frequency domain",
            children=[
                html.Div(
                children=[
                    dcc.Dropdown(
                        options=[],
                        id="multi-dropdown-fd",
                        placeholder="Select classes",
                        multi = True,
                        value=[],
                        clearable=False,
                    ),
                    html.Br(),
                    html.Label('Choose number of lines'),
                    dcc.Slider(1, 10, 1, 
                            value=1, 
                            id='nb-lines-slider-fd'),
    
                    ## Button to select new instance of frequency domain
                    html.Button('Plot randomly', id='button-fd', n_clicks=0),
                ],  style={"width": '100%', "margin": '0.5%',
                        'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                        'background-color': '#e0f0e0',
                        "padding": 10, "borderRadius": 5},    
            ),
                
        
            ## Frequency-domain plots
            html.Div(
                children = [
                    dcc.Graph(id='amplitude-phase-plot'),
                ], 
                style={"width": '100%', "margin": '0.5%',
                        'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                        'background-color': '#e0f0e0',
                        "padding": 10, "borderRadius": 5},                       
            ),
            
            html.Br(),
            
            ## K-means centroids plot
            html.H5('K-Means with sklearn'),
            html.Div(
                id="k-means",
                children = [
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
    ]
)

###===============================        CALLBACKS           ========================================================
###=========================================================================================
## Loading the corresponding datasets from range-Sliders selection


@app.callback(
    Output('3d-plot', 'figure'),
    Output('dropdown', 'options'),
    Output('dropdown', 'value'),
    Input('my-size-range-slider', 'value'),
    Input('my-classes-range-slider', 'value'),
    Input('my-length-range-slider', 'value'),
    Input('3d-plot', 'clickData')
)

def update_output(value_size, value_classes, value_length, clickData):
    lb_size = value_size[0]
    up_size = value_size[1]
    lb_length = value_length[0]
    up_length = value_length[1]
    lb_class = value_classes[0]
    up_class = value_classes[1]
    
    dataset_info = pd.read_pickle('./demoapps/time_series/data/datasets_information.pkl')

    selected_datasets = dataset_info.loc[
                            (dataset_info['size'] >= lb_size) & \
                            (dataset_info['size'] <= up_size) & \
                            (dataset_info['classes'] >= lb_class) & \
                            (dataset_info['classes'] <= up_class) & \
                            (dataset_info['length'] >= lb_length) & \
                            (dataset_info['length'] <= up_length )
                            ]
    
    #3d plot
    x = selected_datasets['size']
    y = selected_datasets['classes']
    z = selected_datasets['length']
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                       text = [format(x) for x in selected_datasets.index ],
                                        marker=dict(
                                        size=5,
                                        color=z,
                                        showscale=True,
                                        colorbar=dict(title='length'),
                                        ))]
                                        )
    
    fig.update_layout(title = "3D-plot of datasets info",
                    scene = dict(
                        xaxis_title='size',
                        yaxis_title='classes',
                        zaxis_title='length'))
    
    if clickData:
        dataset = np.array([clickData['points'][0]['text']])[0]
    else:
        dataset = None
    
    return fig, selected_datasets.index.to_list(), dataset

    

## Updating dropdown for classes
@app.callback(
    Output(component_id='multi-dropdown', component_property='options'),
    Input('3d-plot', 'clickData'),
    Input(component_id='dropdown', component_property='value')
)
def update_multi_dropdown(clickData, dataset):
    x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')
    pd.options.plotting.backend = "plotly"
    df = pd.DataFrame(x_all, y_all)
    return df.index.unique().sort_values()



## Updating figure from selected dataset and class
@app.callback(
    Output(component_id='sample', component_property='figure'),
    Output(component_id='dataset-link', component_property='children'),
    Input(component_id='dropdown', component_property='value'),
    Input(component_id='multi-dropdown', component_property='value'),
    Input(component_id='button', component_property='n_clicks'),
    Input(component_id='nb-lines-slider', component_property='value')
)
def update_figure(dataset, classes, clicks, nb_lines):
    x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')
    x = np.arange(1, x_all.shape[1]+1)
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
                    name= f"class {str(int(current))}", 
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
                    name= f"class {str(int(y_all[i]))}", 
                )
                
    fig.update_layout(
        title = "Timeseries plot", 
        xaxis=dict(title='time'), 
        yaxis=dict(title='data'))
    return fig, f'For more infomation about this dataset: [Click here](https://www.timeseriesclassification.com/description.php?Dataset={dataset})'

## K-MEANS
@app.callback(
    Output('kmeans-graph', 'figure'),
    Input(component_id='dropdown', component_property='value'),
    Input(component_id='multi-dropdown', component_property='value'),
    Input(component_id='nb-clusters-slider', component_property='value'),
    
)
 
def clustering(dataset, classes, nb_clusters):
    x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')
    
    fig = go.Figure() 
    x = np.arange(1, x_all.shape[1]+1)
    data = []
    for j in range(0, len(classes)):
        current = classes[j]
        data.clear()
        for i in range(0, len(x_all)):   
            if y_all[i] == current:       
                data.append(x_all[i])
                
        km =  KMeans(n_clusters=nb_clusters, random_state=0).fit(data)
        
        labels = km.labels_
        percentages = []
        total = len(labels)
        for i in range(nb_clusters):
            count = labels.tolist().count(i)
            percentages.append(round(((count/total)*100)))
    
        centroids = km.cluster_centers_
        for i in range(len(centroids)):
            fig.add_scatter(x=x, 
                y=centroids[i], 
                mode = 'lines+markers',
                marker=
                    dict(
                        symbol=markers[str(i+1)], 
                        size=6,
                        color=colors[str(current)],
                        line=dict(
                            color=colors[str(current)],
                            width=1
                        )
                    ),
                    name= f"class {str(int(current))} ({i+1}) {percentages[i]}%",
                )
            
    
    fig.update_layout(title = "K-means Clustering Timeseries plot",
        xaxis=dict(title='time'), 
        yaxis=dict(title='data'))
    return fig


###===============================        FREQUENCY DOMAIN           ========================================================
###==========================================================================================================================


## Updating dropdown for classes
@app.callback(
    Output(component_id='multi-dropdown-fd', component_property='options'),
    Input('3d-plot', 'clickData'),
    Input(component_id='dropdown', component_property='value')
)
def update_multi_dropdown(clickData, dataset):
    x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')
    pd.options.plotting.backend = "plotly"
    df = pd.DataFrame(x_all, y_all)
    return df.index.unique().sort_values()


## Updating amplitude and angle figures
@app.callback(
    Output(component_id='amplitude-phase-plot', component_property='figure'),
    Input(component_id='dropdown', component_property='value'),
    Input(component_id='multi-dropdown-fd', component_property='value'),
    Input(component_id='button-fd', component_property='n_clicks'),
    Input(component_id='nb-lines-slider-fd', component_property='value')
)
def update_figure(dataset, classes, clicks, nb_lines):
    x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')
    btn_id = 'button-fd'
    freq_domains = np.fft.rfft(x_all)
    fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.07,
                    )
    # add amplitude trace
    amplitude = np.absolute(freq_domains)
    nb_row=1
    fig = update_plot(amplitude, y_all, classes, btn_id, nb_lines, fig, nb_row)
    
    # add angle trace        
    phase = np.angle(freq_domains)
    nb_row=2
    fig = update_plot(phase, y_all, classes, btn_id, nb_lines, fig, nb_row)
                
    fig.update_layout(title = " Amplitude and angle subplots",
        yaxis=dict(title='amplitude'),
        xaxis2=dict(title='index'), 
        yaxis2=dict(title='phase'))
    return fig
    

## K-MEANS 
@app.callback(
    Output('kmeans-graph-fd', 'figure'),
    Input(component_id='dropdown', component_property='value'),
    Input(component_id='multi-dropdown-fd', component_property='value'),
    Input(component_id='nb-clusters-slider-fd', component_property='value'),
)
def clustering(dataset, classes, nb_clusters):
    x_all, y_all = load_dataset(dataset, repository='wildboar/ucr')
    freq_domains = np.fft.rfft(x_all)
    fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.07,)
    
    
    amplitude = np.absolute(freq_domains)
    nb_row=1
    fig = kmeans_clustering(amplitude, y_all, classes, nb_clusters, fig, nb_row)
    
    phase = np.angle(freq_domains)
    nb_row=2
    fig = kmeans_clustering(phase, y_all, classes, nb_clusters, fig, nb_row)
    fig.update_layout(title = "K-means amplitude and angle subplots",
        yaxis=dict(title='amplitude'),
        xaxis2=dict(title='index'), 
        yaxis2=dict(title='phase')),
    return fig


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
    
    