# About <extremum.dsv.su.se>

## SERVER:

Brand new Apache server running Debian 4 and configured with Git, Python and Django.

The whole server is hosted on extremum.dsv.su.se
Decommission date: January 1st, 2027

## GIT:

Version control system embedded in extremum.

- Accessed through: extremum.dsv.su.se/gitea
- Only the users: ppapa, zhwa9764, luva3178 are full **Administrators** of the git server.
- First time you login, you need to set forgot password, enter you dsv-email and set a new password just for gitea.
- Administrators can add more users (even external like RISE, KTH) from the menu in the top-right corner ***Site Administrator***. Every added user will receive an email (most likely to spam folder) where they have to setup their own password.
- The main repository of the EXTREMUM project is in:
    https://extremum.dsv.su.se/gitea/luva3178/EXTREMUM.git

## DEPLOY WEBPLATFORM:


To keep in mind when deploying the django platform from git to the production server.

- Webplatform accessed through: extremum.dsv.su.se/app
- Connect through SSH. The server only accepts SSH connections from DSV networks, you can setup the VPN from this link: https://serviceportalen.su.se/en-us/article/1319269
- So far just Panos, Zhendong, and Luis have privileges to access the server through membership in the extremum group. To add more people to the group please email dmc@dsv.su.se
- **Updating the code in the server:**
  ```
  ssh extremum.dsv.su.se
  cd /var/www/extremum
  git pull https://extremum.dsv.su.se/gitea/luva3178/EXTREMUM.git
  ```
- It will ask for your username and password **from gitea account** (notice that this password could be different than DSV password)
- Git has a tendency to be sloppy with permissions in a way that can break updates when several users are pulling in the same directory (say Zhendong and Panos both pull new changes at different times). To fix this there is a script that can be called with `sudo fix-web-permissions`, which should clear up any such problems. Use it whenever git complains about permissions when trying to pull new updates.
- The Django server is NOT attached to a python virtual environment, therefore all the **Python packages** need to be installed in the main Python of the server through `sudo pip3 [package name]`. This can become messy since many packages will be required for the webplatform to run.
To reload changes in the server and see the platform running type `sudo service apache2 reload`
- Versions of packages:
  - Python 3.7.6
  - ~~Django 2.2.7~~ **Django was replaced by Dash-plotly!!**
- Apache configuration tries to find the django site in the path `/extremum/webplatform/webplatform/wsgi.py` (this needs to be preserved even if we start a project from scratch)
The **apache configuration file** is found in `/etc/apache2/sites-available/extremum.conf`. To change this configuration to a new django site please email dmc@dsv.su.se
- The **homepage website** for the extremum project is changed from the folder `/var/html/`
- The apache logs are located in the folder `/var/log/apache2/`


## HOW TO ADD MORE DASHBOARDS

- Develop your own dash locally, when finished comment the lines that define `app` and `server` in your `app.py`
- Import the app from the general server as: `from extremum_config import app`
- In `dashboards_config.py`, add a new line to the enum `DashboardsURL` with the address that the dashboard should have. e.g. `CustomApp = "customURL"`
- Import your app from the corresponding folder. E.g. `from apps import MyCustomApp`
- At the end of the same document, create a new object as an instance of `DashboardExtremum`, containing the layout, 
    ``` python
    svm = DashboardExtremum(MyCustomApp.layout, 
                        title="Custom Dashboard Extremum",
                        url=DashboardsURL.CustomApp,
                        description="""
                                    Yet another demo for Extremum
                                    """,
                        imagepath="assets/images/cover_customapp.png", # None
                        )
                        ```
