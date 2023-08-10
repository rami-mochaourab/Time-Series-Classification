#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luis-eduardo@dsv.su.se
# Created Date: 2021/01/25
# =============================================================================
"""
Navigation bar
"""
# =============================================================================
# Imports
# =============================================================================

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go

import extremum_config
import dashboards_config

import pandas as pd

# =============================================================================
# Main
# =============================================================================

def generate_header():
    """
    Generates the html webpage for the header with navigation bar.
    """
    layout = html.Div([
        # represents the URL bar, doesn't render anything
        dcc.Location(id='url', refresh=False),

        dbc.NavbarSimple(
            children=[
                # Logo of the project
                # html.A(
                #         # Use row and col to control vertical alignment of logo / brand
                #         dbc.Row(
                #             [
                #                 dbc.Col(html.Img(src= extremum_config.app.get_asset_url( extremum_config.join_paths(["images","logos", extremum_config.LOGO_FILENAME]) ), 
                #                                 height="30px")),
                #             ],
                #             align="center",
                #         ),
                #         href=extremum_config.ROOT_URL,
                #     ),
                dbc.NavItem(dbc.NavLink("Home", active=False, href=str(dashboards_config.MainURL.Index) )),
                dbc.DropdownMenu(
                    # [dbc.DropdownMenuItem("Available apps", header=True)] + # Header of the dropdown
                    [
                        dbc.DropdownMenuItem(
                            f'"{url}"', href=url 
                        ) 
                        for url in dashboards_config.dashboards_info.keys()
                    ],
                    label="Dashboards",
                    in_navbar=True,
                    nav=True
                    ),
                dbc.NavItem(dbc.NavLink("About", active=False, href=str(dashboards_config.MainURL.About) )), #disabled=True
            ],
            brand=extremum_config.TITLE,
            brand_href=extremum_config.ROOT_URL,
            color="light",
            dark=False,
        ),

        # html.Div([ dbc.Row([ dbc.Col(
        #     dbc.Nav(
        #         [
        #             html.A(
        #                 # Use row and col to control vertical alignment of logo / brand
        #                 dbc.Row(
        #                     [
        #                         dbc.Col(html.Img(src= extremum_config.app.get_asset_url( extremum_config.join_paths(["images","logos", extremum_config.LOGO_FILENAME]) ), 
        #                                         height="30px")),
        #                         dbc.Col(dbc.NavbarBrand(extremum_config.TITLE, className="ml-2")),
        #                     ],
        #                     align="center",
        #                     no_gutters=True,
        #                 ),
        #                 href=extremum_config.ROOT_URL,
        #             ),
        #             dbc.NavItem(dbc.NavLink("Home", active=True, href=str(dashboards_config.MainURL.Index) )),
        #             dbc.DropdownMenu(
        #                 [
        #                     dbc.DropdownMenuItem(
        #                         f'"{url}"', href=url 
        #                     ) 
        #                     for url in dashboards_config.dashboards_info.keys()
        #                 ],
        #                 label="Dashboards",
        #                 nav=True
        #             ),
        #             dbc.NavItem(dbc.NavLink("About", href=str(dashboards_config.MainURL.About) )), #disabled=True
        #         ],
        #         fill=True,
        #     ),
        #     width=6)], justify="center"),  # Close COLUMN, ROW FROM NAVIGATION BAR 
        # ],
        # style = {"background-color": "lightgray"},
        # ), # div

        # content will be rendered in this element
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col( html.Div(id='page-content'), width=8),
                    ],
                    justify="center",
                    ),
            ]
        )
    ])

    return layout
