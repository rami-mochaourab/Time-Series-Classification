import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import os
import plotly.graph_objects as go
from pathlib import Path
import sys

from .utils.functions import *
from extremum_config import app
THIS_FILEPATH = os.path.dirname(__file__)
sys.path.append(os.path.join(THIS_FILEPATH, "."))

def load_data():
    X_train = np.load(Path(__file__).resolve().parent / 'utils/X_train.sav.npy')
    y_train = np.load(Path(__file__).resolve().parent / 'utils/y_train.sav.npy')
    return X_train, y_train

def show_sample(id, X_train, y_train):
    fig = go.Figure()
    example = X_train[id]
    fig.add_scatter(name="Positive class", x=X_train[y_train == 1, 0], y=X_train[y_train == 1, 1], mode='markers')
    fig.add_scatter(name="Negative class", x=X_train[y_train == -1, 0], y=X_train[y_train == -1, 1], mode='markers')
    fig.add_scatter(name="Instance", x=np.array(example[0]), y=np.array(example[1]), mode='markers',
                    marker=dict(symbol="circle", color="black", size=8))
    fig.update_layout(  # legend=dict(yanchor="top", xanchor="left", x=.1, y=0.000001),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
    return fig

xtrain, ytrain = load_data()
xtrain_len = len(xtrain)

layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Tutorial for platform development",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        html.A(
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    children=[
                        html.Div(
                            children=[
                                html.H5('Tutorial - Basics'),
                                dcc.Markdown('The development of this platform is done one card app at a time. '
                                             'Each card has its own directory in the "ROOT_DIR/extreme/EXTREMUM/demoapps" directory. '
                                             'All the pickled files and supplemental scripts pertaining to an app '
                                             'should go inside the "utils" directory within the card app directory. '
                                             'Another file that you might have to modify in order to add the card of an app '
                                             'to the dashboard is the dashboards_config.py located in "ROOT_DIR/EXTREMUM/ '
                                             'directory.'),
                                html.Br(),
                                dcc.Markdown('To get familiar with the platform development, please try to complete '
                                             'the following tasks:'),
                                dcc.Markdown('1. Clone the git repo.'),
                                dcc.Markdown('2. Create a virtualenv, activate it and install the dependencies by issuing '
                                             '"pip install -r requirements.txt" command.'),
                                dcc.Markdown('3. Run the demo by executing "python ROOT_DIR/extreme/EXTREMUM/app.py".'),
                                dcc.Markdown('4. Open the project base in your favorite IDE and locate the tutorial '
                                             'directory present inside the demoapps directory. Open the tutorial.py file'),
                                dcc.Markdown('5. Change the single div element below to split horizontally so that '
                                             'Dropdown menu is in the left div element and the plot on the right div element.'),
                                dcc.Markdown('6. On the left div element, change the drop down menu to radio buttons to '
                                             'select one of the two data classes.'),
                                dcc.Markdown('7. Create a function named add_scatter_plot to return a plotly graph figure '
                                             'for the data points of the selected class.'),
                                dcc.Markdown('8. On the right div element, add the scatter plot.'),
                                dcc.Markdown('9. Modify the card description on the landing page to reflect the dataset'
                                             ' name you are using.'),
                                dcc.Markdown('10. Finally, remove all the text that you see on this page.')
                                ],
                                style={"float": "left", "width": '100%', "margin": '.5%', 'border': '0 lightgrey solid',
                                        'display': 'inline-block', 'height': 'auto',
                                        "padding": 10, "borderRadius": 5, 'flex': 0}),
                                ],
                            ),
                        html.Div(
                            id="demo1",
                            children=[
                                html.Label('Select sample point'),
                                dcc.Dropdown(
                                    id="sample",
                                    options=[
                                        {'label': i, 'value': i} for i in range(xtrain_len)
                                    ],
                                    clearable=False,
                                    searchable=False,
                                    value=1,
                                        ),
                                dcc.Graph(id='sample-plot')
                                    ],
                                ),

                        ],
                    ),
                ]
            )

@app.callback(
    Output(component_id='sample-plot', component_property='figure'),
    [Input(component_id='sample', component_property='value'),
    ])
def update_image_src(sample):
    xtrain, ytrain = load_data()
    fig = show_sample(sample, xtrain, ytrain)
    return fig
# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8055)
