#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luis-eduardo@dsv.su.se
# Created Date: 2021/01/25
# =============================================================================
"""
Entry point for the dashboard of EXTREMUM project.
Hosted on extremum.dsv.su.se/app

For dev purposes, run this app with `python app.py` and
visit http://127.0.0.1:8050/ in your web browser.
"""
# =============================================================================
# Imports
# =============================================================================

# Recognize all folders from one level above, to be used in relative paths
from pathlib import Path
import sys,os
THIS_FILEPATH = str(Path().absolute())
print("File Path:", THIS_FILEPATH)
sys.path.append(os.path.join(Path().absolute(), "."))

import extremum_config
import dashboards_config

from extremum_config import app, server

import pandas as pd

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from _layouts import navigation

# =============================================================================
# Main
# =============================================================================

extremum_config.MAIN_FILEPATH = THIS_FILEPATH # Set the variable accessible to all other scripts

# Header
app.layout = navigation.generate_header()
server = server # This is called by extremum.wsgi in the Apache server


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    """
    Main layout of the webpage, the children layout changes depending
    on the URL
    """
    layout = dashboards_config.get_layout_from_url(pathname)
    ret = layout if layout is not None else '404'
    return ret

if __name__ == '__main__':
    print("WARNING: If this execution does not work when trying locally, remember to check the `experiment_config.py` and comment `requests_pathname_prefix`")
    app.run_server( debug=True, 
                    use_reloader=True, port=8055)  # Turn off reloader if inside Jupyter

    
