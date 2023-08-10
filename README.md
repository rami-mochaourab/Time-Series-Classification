
# EXTREMUM

## AIM:
Dashboard to interact with different machine learning and interpretability algorithms

## DEV-SETUP:

### Setting up environment
Clone the repository from:
<https://extremum.dsv.su.se/gitea/luva3178/EXTREMUM.git>

Create and activate python virtual environment from CMD terminal:
```console
...\> python -m venv env

...\> env\Scripts\activate.bat (for Windows)
...\> source env/bin/activate (for Mac/Linux)

...\> pip install -r requirements.txt
```

### Run the Dash/Flask server locally.

- Go at the end of the file `extremum_config.py` and comment the lines `serve_locally` and `requests_pathname_prefix`.
- Run the server locally with `python app.py`


## LIBRARIES SPECIFICATION

> The server runs Debian, therefore try to use Python packages with versions that are stable in the [Debian pkg repo](https://packages.debian.org/stable/python/).

Flask's built-in server that used with Dash-plotly is [**NOT** suitable for production](https://flask.palletsprojects.com/en/1.1.x/deploying/). Thus, it needs to be deployed in the self-hosted option using [mod_wsgi](https://flask.palletsprojects.com/en/1.1.x/deploying/mod_wsgi/) (*to reuse the setup from the previous Django app*)

Main libraries and versions can be read from [requirements.txt](./requirements.txt)

```
numpy==1.16.2
pandas==0.23.3
plotly==4.14.3
dash==1.19.0
scipy==1.1.0
scikit-learn==0.20.2
dash-bootstrap-components==0.11.1
```

 - *No version was found on debian packages*: `dash`, `dash-bootstrap-components`.
 - *Packages with differnet version than Debian*: `plotly` from 3.6.1 to 4.14.3

## UI

Three sources to create UI:
- https://dash-bootstrap-components.opensource.faculty.ai/docs/components/
- https://dash.plotly.com/dash-html-components
- https://dash.plotly.com/dash-core-components

## Demo examples

Taken from:
- https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-tsne
- https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-aix360-heart
- https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-clinical-analytics
- https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-svm

## Resources

- Awesome dash: https://github.com/ucg8j/awesome-dash
- Dash Gallery: https://dash-gallery.plotly.host/Portal/

--- 
## Update:

On Jan 2021, it was decided to start the final project using Dash-plotly libraries to reduce development time related to front-end. Therefore, the django prototype was not used anymore and the info below is no longer applicable.

**Note that the required version of Django is 1.11.29 to be compatible with the Debian server and to avoid possible vulnerabilities.**

### Run the django platform

```console
...\> python webplatform/manage.py runserver
```

- `Django` web-framework in Python used as backend.
- `Jinja2` templating language for Python to render dynamic HTML, it is used instead of the default Django templates.

