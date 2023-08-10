#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luis-eduardo@dsv.su.se
# Created Date: 2021/01/25
# =============================================================================
"""
Configure access to the independent applications that are part of the whole 
project (i.e. each of the dashboards).
Main application is in `tutorial.py`
"""
# =============================================================================
# Imports
# =============================================================================

import enum
import extremum_config
#from demoapps.time_svm import time_svm
from demoapps.privatesvm import privatesvm
from demoapps.tutorial import tutorial
from demoapps.Demo_page import Demo_page
from _layouts import index, about
# =============================================================================
# Functions
# =============================================================================

class DashboardExtremum():
    def __init__(self,
                    layout,         # Compatible object that draws the HTML layout 
                    title:str="Dashboard", 
                    url:str="dashboard1",
                    description:str="Description",
                    imagepath:str=None,
                    ):

        self.title = title
        self.url = url
        self.description = description
        self.imagepath = imagepath
        self.layout = layout

def get_layout_from_url(url):
    """
    Dictionary to control the layout of specific urls
    """
    if(url == str(MainURL.Index)):
        return index.generate_page( dashboards = dashboards,
                                    n_dashboard_cols = 4)
    elif(url == str(MainURL.About)):
        return about.generate_page()     # To be created in the _layouts
    
    # Check from the dict containing the apps
    elif url in dashboards_info:
        return dashboards_info[url]
    else:
        return None


def get_url_from_dashboard(dashboard:DashboardExtremum):
    """
    Returns string with URL for a specific Dashboard
    """
    return f"{str(MainURL.Dashboards)}/{dashboard.url}"

# =============================================================================
# Main
# =============================================================================

class MainURL(enum.Enum):
    """
    Avoid hardcoding the paths of the website. Every new app should refer
    to the enum.
    """
    Index = ""
    Dashboards = "dashboard"
    About = "about"
    def __str__(self):
        return str(extremum_config.ROOT_URL + self.value)

class DashboardsURL(enum.Enum):
    # Apps
    #TimeSVM="timesvm"
    PrivateSVM = "privatesvm"
    Tutorial = "tutorial"
    Demo_Page = "Demo_Page"

    def __str__(self):
        return str(self.value)

# Create an object for each of the dashboards

#time_svm = DashboardExtremum(time_svm.layout,
#                            title="SVM for timeseries data",
#                            url=DashboardsURL.TimeSVM,
#                            description="""
#                                        SVM demo for timeseries data from plotly
#                                        """,
#                            imagepath=None,
#                            )

privatesvm = DashboardExtremum(privatesvm.layout,
                            title="Robust Counterfactual Explanations for Private SVMs",
                            url=DashboardsURL.PrivateSVM,
                            description="""
                                        
                                        """,
                            imagepath="/images/private_SVM_overview_nontechnical.png",
                            )

tutorial = DashboardExtremum(tutorial.layout,
                               title="Tutorial",
                               url=DashboardsURL.Tutorial,
                               description="""

                                        """,
                               imagepath="/images/private_SVM_overview_nontechnical.png",
                               )

Demo_page = DashboardExtremum(Demo_page.layout,
                               title="MNIST Page",
                               url=DashboardsURL.Demo_Page,
                               description="""

                                        """,
                               imagepath="/images/private_SVM_overview_nontechnical.png",
                               )

# Contains all the dashboards that exist in the application
dashboards = [privatesvm, tutorial, Demo_page] #time_svm, demo_dami_analytics, app1, app2, breastcancer

# Used to get the respective layout when the url changes.
dashboards_info = { get_url_from_dashboard(dboard) : dboard.layout for dboard in dashboards}

