#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luis-eduardo@dsv.su.se
# Created Date: 2021/01/25
# =============================================================================
"""
About webpage
"""
# =============================================================================
# Imports
# =============================================================================

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import extremum_config

# =============================================================================
# Main
# =============================================================================

def generate_page(fluid = True):
    """
    Generates the html webpage for the index.
        - fluid: Use all the available space?
    """
    
    header = dbc.Jumbotron([
            # html.H1(extremum_config.TITLE, className="display-3"),
            # html.Hr(className="my-2"),
            # html.P("About", className="lead"),
            # html.Div(['More information about the project: ', 
            #     html.A('https://datascience.dsv.su.se/projects/extremum.html', href='https://datascience.dsv.su.se/projects/extremum.html')
            # ]),

            # dcc.Markdown(extremum_config.read_markdown(extremum_config.get_full_path(["_mkdown","about_text.md"]))),

            html.Div(
            [
                html.H1("About EXTREMUM"),

                html.H3("Explainable Machine Learning"),

                html.P("This project intends to build a novel data management and analytics framework, focusing on three pillars: (1) data integration and federated learning, (2) explainable machine learning, and  (3) legal and ethical integrity of predictive models."),

                html.P("The final product will be a set of methods and tools for integrating massive and heterogeneous medical data sources in a federated manner, a set of predictive models for learning from these data sources, with emphasis on interpretability and explainability of the models rationale for the predictions, while focusing on maintaining ethical integrity and fairness in the underlying decision making mechanisms that govern machine learning. The project will focus on two critical application areas: adverse drug event detection and heart failure treatment."),

                html.P("The project is a collaborative effort between four research institutions: the department of Computer and Systems Sciences at Stockholm University, the Department of Law at Stockholm University, RISE Research Institute Sweden, and KTH."),

                html.P(["Find more information on: ", html.A("https://datascience.dsv.su.se/projects/extremum.html", href="https://datascience.dsv.su.se/projects/extremum.html")]),
            ] ),

            html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(src=extremum_config.app.get_asset_url( extremum_config.join_paths(["images","logos", logo] )),
                                     style={'width':'50%'}),
                            width=4,  className="center")
                            for logo in extremum_config.logos
                    ]
                ),
            ]),
            
    ], style=dict(marginTop=5) )

    # Final layout to return
    layout = dbc.Container([
        dbc.Row([dbc.Col([header])]),
    ], fluid=fluid, className="p-5")

    return layout
