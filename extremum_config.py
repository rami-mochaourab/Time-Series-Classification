#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luis-eduardo@dsv.su.se
# Created Date: 2021/01/25
# =============================================================================
"""
Setup functions and variables specific to the dashboard of EXTREMUM project.
Main application is in `tutorial.py`
"""
# =============================================================================
# Imports
# =============================================================================

import dash
import dash_bootstrap_components as dbc

import os

# =============================================================================
# Main
# =============================================================================

# GLOBAL VARIABLES

MAIN_FILEPATH = ""      # This is changed from script app.py to know the path
                        # from which the application is executed

# =============================================================================
# Custom variables
# =============================================================================

ROOT_URL = "/app/" # "http://extremum.dsv.su.se/app"

TITLE = "EXTREMUM"
SUBTITLE = "Explainable Machine Learning"

# Filenames of the logos located in assets/images/logos/*
logos = ["su.png", "kth.png", "rise.png"]
LOGO_FILENAME = "extremum.png"


# =============================================================================
# Common Functions for file processing
# =============================================================================

def read_markdown(path):
    """
    Use to import a markdown as an element in the webpage.
    """
    text = None
    try:
        with open(path, 'r') as file:
            text = file.read()
        return text
    except Exception as e:
        print(e)
        return None


def get_full_path(relative_path):
    """
    Converts a filepath that is relative from `app.py`, to a full path
    """
    if (type(relative_path) is str):
        result = os.path.join(MAIN_FILEPATH, relative_path)
    elif (type(relative_path) is list):
        result = os.path.join(MAIN_FILEPATH, *relative_path)
    return result


def join_paths(list_paths):
    """
    Returns a joint sequence of paths.
    """
    result = os.path.join(*list_paths)
    return result

# =============================================================================
# Server Setup
# =============================================================================

# Add compatibility with LaTeX formulas with MathJax
external_stylesheets = [dbc.themes.BOOTSTRAP] #['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_scripts = ["https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML",
                    join_paths([ROOT_URL,"assets","js","updatemathjax.js"]) ]

app = dash.Dash(__name__, title = TITLE, update_title = "Loading...", 
		        #requests_pathname_prefix=ROOT_URL, ### COMMENT THIS LINE WHEN DEPLOYING LOCALLY (UNCOMMENT IN APACHE SERVER)
                suppress_callback_exceptions=True,
		        serve_locally=True,
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts)

server = app.server


