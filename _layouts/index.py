#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luis-eduardo@dsv.su.se
# Created Date: 2021/01/25
# =============================================================================
"""
Index webpage
"""
# =============================================================================
# Imports
# =============================================================================

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go

import extremum_config
import dashboards_config

import pandas as pd

# =============================================================================
# Main
# =============================================================================

def generate_page(dashboards = [], 
                        n_dashboard_cols = 3,
                        fluid = True):
    """
    Generates the html webpage for the index.
        - dashboards: List of dashboards.Dashboard to generate the cards
        - n_dashboard_cols: How many cols per row
        - fluid: Use all the available space?
    """
    def dashboard_decks(dashboards, n_cols):
        """
        Generates the cards with the available dashboards
        """
        full_rows = int(len(dashboards)/ n_cols)
        n_last_row = len(dashboards) % n_cols

        card_decks = []

        next_dashboard = 0
        for i in range(0, full_rows+1):
            card_row = []
            for _ in range(0,n_cols):
                
                # Load next dashboard
                dashboard = dashboards[next_dashboard] if next_dashboard < len(dashboards) else None
                if dashboard is None:
                    break
                
                card_content = dbc.Card([
                            dbc.CardHeader([
                                html.H5(dashboard.title, className='card-title'),
                            ]),
                            dbc.CardBody([ dbc.CardImg(src= extremum_config.app.get_asset_url( extremum_config.join_paths([dashboard.imagepath]) ), top=True) if (dashboard.imagepath is not None) else None,
                                html.H6(dashboard.description),
                            ]),
                            dbc.CardFooter([
                                dbc.CardLink("Go to dashboard", 
                                            href=dashboards_config.get_url_from_dashboard(dashboard),
                                            external_link=True),
                            ])])

                card_row.append( dbc.Col(card_content, width = int(12/n_cols) ) )
                next_dashboard = next_dashboard + 1

            card_decks.append( dbc.Row(card_row.copy(), style=dict(marginBottom=30)) )

        # for i in range(0, full_rows*n_cols, n_cols):
        #     card_decks.append(
        #         [dbc.Card([
        #                 dbc.CardHeader([
        #                     html.H5(dashboard.title, className='card-title'),
        #                 ]),
        #                 dbc.CardBody([ dbc.CardImg(src= extremum_config.app.get_asset_url( extremum_config.join_paths([dashboard.imagepath]) ), top=True) if (dashboard.imagepath is not None) else None,
        #                     html.H6(dashboard.description),
        #                 ]),
        #                 dbc.CardFooter([
        #                     dbc.CardLink("Go to dashboard", 
        #                                 href=dashboards_config.get_url_from_dashboard(dashboard),
        #                                 external_link=True),   
        #                 ])]) for dashboard in dashboards[i:i+n_cols]
        #         ]
        #     )
        """
        if n_last_row > 0:
            last_row = [
                dbc.Card([
                    dbc.CardHeader([
                        html.H3(dashboard.title, className='card-title'),
                    ]),
                    dbc.CardBody([ dbc.CardImg(src=f"{extremum_config.ROOT_URL}{dashboard.imagepath}", top=True) if (dashboard.imagepath is not None) else None,
                        html.H6(dashboard.description),
                    ]),
                    dbc.CardFooter([
                        dbc.CardLink("Go to dashboard", 
                                    href=f"{extremum_config.ROOT_URL}{dashboard.url}", 
                                    external_link=True),   
                    ])
                ]) for dashboard in dashboards[full_rows*n_cols:full_rows*n_cols+n_last_row]]
            for i in range(len(last_row), n_cols):
                last_row.append(dbc.Card([], style=dict(border="none")))
            card_decks.append(last_row)
        """
        return card_decks

    header = dbc.Jumbotron([
            html.H1(extremum_config.TITLE, className="display-2"),
            html.H2(extremum_config.SUBTITLE, className="display-5"),
            html.Hr(className="my-2"),

            # dcc.Markdown(extremum_config.read_markdown(extremum_config.get_full_path(["_mkdown","home_text.md"]))),

            html.P("This project intends to build a novel data management and analytics framework, focusing on three pillars: (1) data integration and federated learning, (2) explainable machine learning, and (3) legal and ethical integrity of predictive models."),

            html.P("The project is a collaborative effort between four research institutions: the Department of Computer and Systems Sciences at Stockholm University, the Department of Law at Stockholm University, RISE Research Institute Sweden, and KTH."),

            # Logos
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
    ])

    dashboard_rows = html.Div(
        # [
            # dbc.Row([dbc.CardDeck(deck)], style=dict(marginBottom=30)) 
            #     for deck in dashboard_decks(dashboards, n_dashboard_cols)
            dashboard_decks(dashboards, n_dashboard_cols)
        # ]
        )

    """
    # assume you have a "long-form" data frame
    # see https://plotly.com/python/px-arguments/ for more options
    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig = go.Figure(data=[go.Bar(x=df.Fruit, y=df.Amount)])

    table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
    """

    # Final layout to return
    layout = dbc.Container([
        dbc.Row([dbc.Col([header])]),
        # logos,
        #dbc.Row([dbc.Col([html.H2("Dashboards:")])]),
        dashboard_rows,
        
        # dbc.Alert("Example other objects!", color="success"),
        # dcc.Markdown(extremum_config.read_markdown(extremum_config.get_full_path(["_mkdown","home_text.md"]))),
        # dcc.Graph(
                    # id='example-graph',
                    # figure=fig
                    # ),
        # table

    ], fluid=fluid, className="p-5")

    return layout
