import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import os
from pathlib import Path
import plotly.graph_objects as go
import random
from sklearn import preprocessing
import sys
from .utils.svm import LinearSupportVectorMachine_noOffset
from .utils.RandFourier import RandomFourier
from .utils.functions import *
from extremum_config import app
import extremum_config
THIS_FILEPATH = os.path.dirname(__file__)
sys.path.append(os.path.join(THIS_FILEPATH, "."))
neg_samples = np.arange(170)

def load_data():
    X_train = np.load(Path(__file__).resolve().parent / 'utils/X_train.sav.npy')
    X_test = np.load(Path(__file__).resolve().parent / 'utils/X_test.sav.npy')
    y_train = np.load(Path(__file__).resolve().parent / 'utils/y_train.sav.npy')
    y_test = np.load(Path(__file__).resolve().parent / 'utils/y_test.sav.npy')
    K = np.outer(X_train, X_train)
    kappa = np.sqrt(K.max())
    return X_train, X_test, y_train, y_test, K, kappa

def load_data_bc():
    X_train_bc = np.load(Path(__file__).resolve().parent / 'utils_bc/X_train.sav.npy')
    X_test_bc = np.load(Path(__file__).resolve().parent / 'utils_bc/X_test.sav.npy')
    y_train_bc = np.load(Path(__file__).resolve().parent / 'utils_bc/y_train.sav.npy')
    y_test_bc = np.load(Path(__file__).resolve().parent / 'utils_bc/y_test.sav.npy')
    K_bc = np.outer(X_train_bc, X_train_bc)
    kappa_bc = np.sqrt(K_bc.max())
    return X_train_bc, X_test_bc, y_train_bc, y_test_bc, K_bc, kappa_bc

def find_proto(y_train):
    X_train, _, _, _, _, _ = load_data()
    prototype = [np.mean(X_train[y_train == 1], axis=0), np.mean(X_train[y_train == -1], axis=0)]
    return prototype

def find_proto_bc(y_train):
    X_train_bc, _, _, _, _, _ = load_data_bc()
    prototype_bc = [np.mean(X_train_bc[y_train == 1], axis=0), np.mean(X_train_bc[y_train == -1], axis=0)]
    return prototype_bc

def load_svm():
    SVM = np.load(Path(__file__).resolve().parent / 'utils/SVM.pkl', allow_pickle=True)
    return SVM

def get_freq_graph_bc(sample_id, beta, proba, xaxis_column, yaxis_column, seed = 10):
    X_train, X_test, y_train, y_test, K, kappa = load_data_bc()
    n = X_train.shape[0]
    F = int(2*50)
    C = np.sqrt(n)
    SVM = LinearSupportVectorMachine_noOffset(C=C)
    # calculate noise scale: lambda
    kappa = 1
    noise_lambda = 4 * C * np.sqrt(F) / (beta * n)
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    FM_transform = RandomFourier(n_components=F, random_state=1).fit(X_train_scaled)
    X_train_FM = FM_transform.transform(X_train_scaled)
    X_test_FM = FM_transform.transform(X_test_scaled)
    X_train_b = add_bias(X_train_scaled)
    lagr_multipliers, idx_support_vectors, support_vectors, support_vector_labels = SVM.fit(X_train_FM, y_train)
    # weight vector
    w = np.dot(lagr_multipliers * support_vector_labels, support_vectors)
    np.random.seed(seed=seed)
    mu = np.random.laplace(loc=0.0, scale=noise_lambda, size=(1, F))
    idx_selected = sample_id
    prototype = find_proto_bc(y_train)
    instance = [X_test[idx_selected]]
    instance_transformed = FM_transform.transform(np.array(instance).reshape(1, -1))
    prediction_instance = np.sign(SVM.predict(instance_transformed, noise=0).flatten())
    prediction_instance_robust = np.sign(SVM.predict(instance_transformed, noise=mu).flatten())
    # choose prototype of different class than instance
    selected_prototype = prototype[np.where(prediction_instance != [1, -1])[0][0]]
    selected_prototype_transformed = FM_transform.transform(selected_prototype.reshape(1, -1))
    explanation, plot_convergence = bisection_chance(instance, prediction_instance, selected_prototype, SVM,
                                                     FM_transform, mu, noise_lambda, p=proba, acc=0.0001)
    counterfactual_explanation = unscale(explanation, scaler)
    prediction_counterfactual_explanation = np.sign(SVM.predict(FM_transform.transform(np.array(explanation).reshape(1, -1)), noise=0).flatten())
    prediction_counterfactual_explanation_private = np.sign(
        SVM.predict(FM_transform.transform(np.array(explanation).reshape(1, -1)), noise=mu).flatten())
    featurevaluesfig = go.Figure()
    featurevaluesfig.add_trace(go.Bar(x=features_labels, y=X_train[idx_selected], name='selected instance', marker_color='rgb(225,202,158)'))
    featurevaluesfig.add_trace(go.Bar(x=features_labels, y=counterfactual_explanation.flatten(), name='robust counterfactual explanation', marker_color='rgb(202,158,225)'))
    featurevaluesfig.update_xaxes(title_text="Feature")
    featurevaluesfig.update_yaxes(title_text="Value")
    featurevaluesfig.update_yaxes(type="log")
    featurevaluesfig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis = dict(gridcolor="#ddd"),
        font=dict(size=8),
        autosize=False,
        height=200,
        margin=dict(
            l=50,
            r=20,
            b=70,
            t=80,
            pad=0
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor='#ffffff'
    )
    c1 = xaxis_column
    c2 = yaxis_column

    X_train_unscaled = scaler.inverse_transform(X_train_scaled)
    X_test_unscaled = scaler.inverse_transform(X_test_scaled)

    y_pred = np.sign(SVM.predict(X_train_FM).flatten())
    y_pred_priv = np.sign(SVM.predict(X_train_FM, mu).flatten())

    P = y_train == 1
    N = y_train == -1

    FN = (y_pred == -1) & (y_train == 1)
    FP = (y_pred == 1) & (y_train == -1)

    FP_p = (y_pred_priv == 1) & (y_train == -1)
    FN_p = (y_pred_priv == -1) & (y_train == 1)

    cffig = go.Figure()
    cffig.add_scatter(name="Positive Class", x=X_train_unscaled[P, int(xaxis_column)],
                     y=X_train_unscaled[P, int(yaxis_column)], mode='markers',
                      marker=dict(color="dodgerblue"))
    cffig.add_scatter(name="Negative Class", x=X_train_unscaled[N, int(xaxis_column)],
                     y=X_train_unscaled[N, int(yaxis_column)], mode='markers', marker=dict(color="orange"))
    cffig.add_scatter(name="False Positive", x=X_train_unscaled[FP, int(xaxis_column)],
                     y=X_train_unscaled[FP, int(yaxis_column)], mode='markers',
                      marker=dict(color='dodgerblue',line=dict(color='cyan',width=2)
                    ))
    cffig.add_scatter(name="False Negative", x=X_train_unscaled[FN, int(xaxis_column)],
                     y=X_train_unscaled[FN, int(yaxis_column)], mode='markers',
                      marker=dict(color='orange',line=dict(color='darkgreen',width=2)
                    ))
    cffig.add_scatter(name="False Positive (Private SVM)", x=X_train_unscaled[FP_p, int(xaxis_column)],
                     y=X_train_unscaled[FP_p, int(yaxis_column)], mode='markers',
                      marker=dict(color='dodgerblue',line=dict(color='crimson',width=2)
                    ))
    cffig.add_scatter(name="False Negative (Private SVM)", x=X_train_unscaled[FN_p, int(xaxis_column)],
                     y=X_train_unscaled[FN_p, int(yaxis_column)], mode='markers',
                     marker = dict(color='orange', line=dict(color='chocolate', width=2)
                    ))

    cffig.add_scatter(name="Instance", x=np.array(X_test_unscaled[idx_selected][int(xaxis_column)]),
                     y=np.array(X_test_unscaled[idx_selected][int(yaxis_column)]), mode='markers',
                     marker=dict(symbol="circle", color="black", size=8))
    cffig.add_scatter(name="Robust Explanation", x=np.array(explanation[0][int(xaxis_column)]),
                     y=np.array(explanation[0][int(yaxis_column)]),
                     mode='markers', marker=dict(symbol="star", size=8, color="green"))

    cffig.update_xaxes(title_text=features_labels[int(xaxis_column)])
    cffig.update_yaxes(title_text=features_labels[int(yaxis_column)])
    cffig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
    cffig.update_layout(
        autosize=False,
        margin=dict(
            l=50,
            r=20,
            b=70,
            t=80,
            pad=4
        ),
        paper_bgcolor="#e0f0e0",
        plot_bgcolor='#e0f0e0'
    )
    return featurevaluesfig, cffig, y_test[idx_selected], prediction_instance.squeeze(), \
           prediction_instance_robust.squeeze(), prediction_counterfactual_explanation.squeeze()

def compute_explanation_maths(instance, beta, seed, proba):
    SVM = load_svm()
    X_train, X_test, y_train, y_test, K, kappa = load_data()
    SS = 250
    n = 2 * SS
    F = 2
    C = np.sqrt(n)
    # calculate noise scale: lambda
    noise_lambda = 4 * C * kappa * np.sqrt(F + 1) / (beta * n)
    np.random.seed(seed=seed)
    mu = np.random.laplace(loc=0.0, scale=noise_lambda, size=(1, F + 1))

    X_train_b = add_bias(X_train)
    lagr_multipliers, idx_support_vectors, support_vectors, support_vector_labels = SVM.fit(X_train_b, y_train)

    w = np.dot(lagr_multipliers * support_vector_labels, support_vectors)
    w_tilde = w + mu.flatten()
    prototype = find_proto(y_train)
    instance_b = add_bias(instance)
    prediction_instance = np.sign(SVM.predict(instance_b, noise=0).flatten())
    # choose prototype of different class than instance
    selected_prototype = prototype[np.where(prediction_instance != [1, -1])[0][0]]
    b_robust = prediction_instance * w_tilde / (noise_lambda * np.sqrt(2) * np.log(2 * (1 - proba)))
    b_non_robust = prediction_instance * w_tilde
    b_opt = prediction_instance * w
    explanation_robust = socp_opt(instance, F, b_robust)
    explanation_non_robust = counterfactual_explanation_linear(instance, F, b_non_robust)
    explanation_opt = counterfactual_explanation_linear(instance, F, b_opt)
    x_min, x_max, y_min, y_max = -1, 2, -1, 2
    res = 500j
    h = 0.2
    XX, YY = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_grid = np.c_[XX.ravel(), YY.ravel()]

    X_grid_b = add_bias(X_grid)
    Z = SVM.predict(X_grid_b)
    Z = Z.reshape(XX.shape)

    Z_p = SVM.predict(X_grid_b, mu)
    Z_p = Z_p.reshape(XX.shape)

    y_ = np.arange(y_min, y_max, h)
    fig = go.Figure()
    fig.add_scatter(name="Positive class", x=X_train[y_train == 1, 0], y=X_train[y_train == 1, 1], mode='markers')
    fig.add_scatter(name="Negative class", x=X_train[y_train == -1, 0], y=X_train[y_train == -1, 1], mode='markers')
    fig.add_scatter(name="Instance", x=np.array(instance[0][0]), y=np.array(instance[0][1]), mode='markers',
                    marker=dict(symbol="circle", color="black", size=8))
    fig.add_scatter(name="Robust Explanation", x=np.array(explanation_robust[0]), y=np.array(explanation_robust[1]),
                    mode='markers', marker=dict(symbol="star", size=8, color="green"))
    fig.add_scatter(name="Non-robust explanation", x=np.array(explanation_non_robust[0]),
                    y=np.array(explanation_non_robust[1]), mode='markers',
                    marker=dict(symbol="diamond-tall", color="white",
                                line=dict(width=2,
                                          color='black'), size=8))
    fig.add_scatter(name="Optimal explanation", x=np.array(explanation_opt[0]), y=np.array(explanation_opt[1]),
                    mode='markers', marker=dict(symbol="square", color="white", line=dict(width=2,
                                                                                          color='black'), size=8))
    fig.add_contour(x=XX[0], y=y_, z=Z, contours_coloring='lines', colorscale='peach',
                    contours=dict(start=0, end=0, showlabels=False), showscale=False, showlegend=False)
    fig.add_contour(line_dash='dash', x=XX[0], y=y_, z=Z_p, colorscale='magenta',
                    contours=dict(coloring='lines', showlabels=False, start=0, end=0), showscale=False, showlegend=False)
    fig.update_xaxes(title_text="$x_1$")
    fig.update_yaxes(title_text="$x_2$")
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
    fig.update_layout(
        autosize=False,
        #width=650,
        #height=650,
        margin=dict(
            l=50,
            r=20,
            b=70,
            t=80,
            pad=4
        ),
        paper_bgcolor="#e0f0e0",
        plot_bgcolor='#e0f0e0'
    )
    return fig

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
                                    "Robust Counterfactual Explanations for Differentially Private SVM",
                                    # href="https://github.com/plotly/dash-svm",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        html.A(),
                    ],
                    style={"text-align": "center", "padding": "20px 0px"}
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            children=[
                                html.H5('Motivation'),
                                dcc.Markdown('Despite their efficiency in solving complex problems, machine learning (ML) '
                                       'algorithms and models are seldom value-neutral to the extent that they include '
                                       'social and ethical values. Even when such values are integrated into the models '
                                       'they may be mandated by regulatory frameworks, such as traditional laws '
                                       'or policy documents. This work aims to illustrate the relational nexus between '
                                       'social and ethical values in a technical context, by focusing on three values '
                                       'advocated by the General Data Protection Regulation (GDPR), namely, '
                                       '*explainability*, *privacy*, and *accuracy*.')
                                ],
                                style={"float": "left", "width": '100%', "margin": '.5%', 'border': '0 lightgrey solid',
                                        'display': 'inline-block', 'height': 'auto',
                                        "padding": 10, "borderRadius": 5, 'flex': 0}),
                        html.Div(
                            children=[
                            html.Img(width="50%",
                                     src=extremum_config.app.get_asset_url(extremum_config.join_paths(
                                         ["images", "private_SVM_overview_nontechnical.png"])),
                                     style={"display": "block", "margin": "auto"}),
                            ],
                            style={"float": "center", "width": '100%', "margin": '.5%', 'border': '0px lightgrey solid',
                                        'display': 'inline-block', 'height': 'auto',
                                        "padding": 10, "borderRadius": 5, 'flex': 1}),
                        html.Div(
                            children=[
                                html.P("As shown in the figure above, consider a dataset containing sensitive data such as patients’ historical "
                                   "health records and respective diagnosis, represented by $\mathcal{D}$. It is assumed "
                                   "that the dataset is securely stored within the confines of the hospital’s technical infrastructure "
                                   "without public access to its entries. Within the secure confines of the hospital, the dataset is "
                                   "employed to train a machine learning classifier, in this case a Support Vector Machine (SVM), "
                                   "that would predict the diagnosis of future patients based on the available historical data. The "
                                   "objective of SVM learning is to achieve the highest possible prediction accuracy and accordingly "
                                   "perform correct diagnosis for most future patients, i.e. as many patients as is technically possible. "
                                   "Hence, the value of accuracy is the prime goal."),
                                html.P("The functionality of the SVM classifier depends on a set of parameters which are"
                                       " determined using the patients’ health records in the dataset. Hence, any public "
                                       "accessibility to the trained SVM classifier parameters may lead to privacy breaches "
                                       "if an adversary manages to reconstruct the patients’ dataset using the classifier parameters. "
                                       "Therefore, we need to ensure that the privacy of the persons in the dataset is preserved before "
                                       "publicly releasing the classifier. The privacy mechanism we will use here guarantees differential privacy [DR14], "
                                       "which is a privacy notion that incorporates a tuneable degree of uncertainty about "
                                       "the actual presence of any entry in the dataset. This uncertainty is achieved in our case through "
                                       "random perturbation of the SVM classifier parameters [RBHT12]. "
                                       "Consequently, the private version of the SVM classifier can be made publicly available with "
                                       "potential utilization in various contexts, e.g., in many different hospitals."),
                                html.P("Clearly, the benefit in guaranteeing privacy comes at the cost of reduced classifier accuracy. "
                                       "In addition, providing valid counterfactual explanations for the privacy preserving SVM classifier is a "
                                       "challenging task due to the introduced perturbations to the optimal classifier parameters. "
                                       "Counterfactual explanations provide us with the necessary changes to the patient’s data that lead "
                                       "to a different predicted diagnosis according to the classifier. When employing the "
                                       "private SVM classifier, the counterfactual explanations may still give the same original diagnosis according to the optimal SVM "
                                       "classifier. Addressing this issue requires studying robust "
                                       "counterfactual explanations that consider the extent of perturbations required by the privacy "
                                       "mechanism. Generating larger perturbations to achieve larger levels of differential privacy would "
                                       "essentially require larger changes in the patient’s data for counterfactual explanations. "),
                                       ],
                            id='element-to-show1',
                            style={"float": "left", "width": '100%', "margin": '.5%', 'border': '1 lightgrey solid',
                                   'display': 'inline-block', 'height': 'auto',
                                   "padding": 10, "borderRadius": 5, 'flex': 1, "clear": "both"}),
                        html.Div(
                            children=[
                                html.H5('SVM Classification'),
                                dcc.Markdown('Dataset ' 
                                            r'$\mathcal{D} = \{(\boldsymbol{x}_1,y_1), \ldots, (\boldsymbol{x}_n,y_n)\}$'),
                                dcc.Markdown(r'$\quad -$ ' '*features vector*: ' r'$\boldsymbol{x}_i~\in~\mathbb{R}^L$'),
                                dcc.Markdown(r'$\quad -$ ' '*class label*: ' r'$y_i \in \\{-1,1\\}$'),
                                dcc.Markdown('SVM classifier with hinge loss and parameter ' r'$C \geq 0$:'
                                             r'$$\qquad\qquad \boldsymbol{w}^* = \text{argmin}_{\boldsymbol{w}\in\mathbb{R}^F} \frac{1}{2}\Vert \boldsymbol{w} \Vert ^2 '
                                             r'+ \frac{C}{n}'
                                             r'\sum\_{i=1}^n'
                                             r'\max \\{0,1-y_i f\_\phi ( \boldsymbol{x}\_i, \boldsymbol{w}) \\}$$ '),
                                dcc.Markdown(r'$\quad -$ ' '*classifier* function: ' r'$f_\phi(\boldsymbol{x},\boldsymbol{w}) := \phi(\boldsymbol{x})^T\boldsymbol{w}$'),
                                dcc.Markdown(r'$\quad -$ ' '*feature mapping*: ' r'$\phi: \mathbb{R}^L \rightarrow \mathbb{R}^{F}$'),
                                dcc.Markdown(r'$\quad -$ ' '*binary classification*: of a data point ' r'$\boldsymbol{x}$' ' is the sign of ' r'$f_\phi(\boldsymbol{x},\boldsymbol{w}^*)$.'),
                                ],
                            id='element-to-hide1',
                            style={"float": "left", "width": '100%', "margin": '.5%', 'border': 'thin lightgrey solid',
                                    'display': 'inline-block', 'height': 'auto',
                                    "padding": 10, "borderRadius": 5, 'flex': 1, "clear": "both"}),
                        html.Div(
                            children=[
                                html.H5('Private SVM'),
                                dcc.Markdown('From Theorem 10 in \[RBHT12\], the perturbed SVM weight vector '
                                    r'$$\qquad \tilde{\boldsymbol{w}} := \boldsymbol{w}^* + \boldsymbol{\mu}$$'
                                    ' guarantees ' r'$\beta-$' 'differential privacy \[DR14\] for iid Laplace random noise '
                                    r'$$\qquad \mu_i \sim \textrm{Lap}(0,\lambda), \quad  i = 1,\ldots,F,$$' ' where '
                                    r'$\lambda \geq 4 C \kappa \sqrt{F}/ (\beta n),$' ' and ' 
                                    r'$\kappa$' ' satisfies ' r'$\phi(\boldsymbol{x})^T \phi(\boldsymbol{x}) \leq \kappa^2$' ' for all ' r'$\boldsymbol{x}\in\mathbb{R}^L.$'
                                    )
                                ],
                            id='element-to-hide2',
                            style={"float": "left", "width": '100%', "margin": '.5%', 'border': 'thin lightgrey solid',
                                    'display': 'inline-block', 'height': 'auto',
                                    "padding": 10, "borderRadius": 5, 'flex': 1}),
                        html.Div(
                            children=[
                                html.H5('Robust Counterfactual Explanation'),
                                dcc.Markdown(
                                    'The counterfactual explanation problem \[WMR18\] is finding the least necessary changes to a data instance ' 
                                    r'$\boldsymbol{x}$' ' to change its classification ' r'$y$:'
                                    r'$$\qquad\qquad\text{minimize}_{\boldsymbol{x}^{ex} \in \mathbb{R}^L} \quad '
                                    r'd(\boldsymbol{x}^{ex},\boldsymbol{x}) \quad s.t. \quad y'
                                    r'f\_\phi(\boldsymbol{x}^{ex},\boldsymbol{w})'
                                    r'\leq 0,$$'
                                ),
                                dcc.Markdown(
                                    'where ' r'$d(\boldsymbol{x}^{ex},\boldsymbol{x})$' ' is a distance between ' r'$\boldsymbol{x}^{ex}$ and $\boldsymbol{x}$. '
                                    'Since private SVM uses a noisy version of the optimal SVM weight vector ' 
                                             r'$\boldsymbol{w}^{*}$' 
                                             ', the uncertainty about the correctness of its classification must be taken into account for counterfactual explanation.'
                                    'Therefore, it is necessary to consider the following **robust counterfactual explanation problem**:'
                                    r'$$\qquad\qquad\text{minimize}_{\boldsymbol{x}^{ex} \in \mathbb{R}^L} \quad '
                                    r'd(\boldsymbol{x}^{ex},\boldsymbol{x}) \quad s.t. \quad'
                                    r'\mathrm{Pr} \left\[ yf\_\phi(\boldsymbol{x}^{ex},\boldsymbol{\xi})'
                                    r'\leq 0 \right\] \geq p,$$'
                                ),
                                dcc.Markdown('where ' r'$p \in \[0.5,1\]$ ' 'and the random vector ' 
                                             r'$\boldsymbol{\xi} = \tilde{\boldsymbol{w}} - \boldsymbol{\nu}$,' 
                                            ' with ' r'$\nu_i \sim \textrm{Lap}(0,\lambda), i = 1,\ldots,F,$'
                                            ' models the uncertainty about the unknown optimal weight vector ' r'$\boldsymbol{w}^*$.'),
                                dcc.Markdown('**Proposition**: The deterministic equivalent of the probabilistic constraint above, with ' r'$p\in \[0.5,1\]$' ', is'),
                                dcc.Markdown(r'$$\qquad\qquad \underbrace{y f\_\phi(\boldsymbol{x}^{ex},\tilde{\boldsymbol{w}}) - \lambda \sqrt{2} \ln(2(1-p)) ||{\phi(\boldsymbol{x}^{ex})}||}_{g(\boldsymbol{x}^{ex})} \leq 0.$$'),
                                dcc.Markdown('- For linear SVM, i.e., ' r'$\phi(\boldsymbol{x}) = \boldsymbol{x}$' ', the constraint can be reformulated as a convex second order cone constraint.'),
                                dcc.Markdown('- For kernel SVM, a suboptimal solution is a *root* for the function ' r'$g$' ' which can be found efficiently using the *bisection method*.'),
                            ],
                        id='element-to-hide3',
                        style={"float": "left", "width": '100%', "margin": '0.5%', 'border': 'thin lightgrey solid',
                               'display': 'inline-block', 'background-color': '#FFF',
                               "padding": 10, "borderRadius": 5, 'flex': 1}),
                        dcc.Checklist(
                            id='check_tech',
                            options=[
                                {'label': ' Technical Content', 'value': 'on'},
                            ],
                            value=[],
                            style={'flush': 'right', 'padding': '5px', 'float': 'right'},
                        ),
                        html.Div(
                            id="demo1",
                            children=[
                                dcc.Markdown('The figure below shows the optimal linear SVM (red line) and its '
                                             'private version (dashed line). Counterfactual explanations are the closest '
                                             'points to the selected instance ' r'($\bullet$)' ' that lie on the decision '
                                             'boundaries. Non-robust explanation ' r'($\diamond$)' ' may have the same class as the instance wrt optimal SVM. '
                                             'Hence, non-robust explanations are not credible and therefore we find robust '
                                             'explanations ' r'($\star$)' ' that provide confidence in explanation credibility.'),
                                html.Label(r'Select privacy parameter value ($\beta$)'),
                                dcc.Slider(id='beta',
                                           min=0.25,
                                           max=50,
                                           step=.25,
                                           value=5,
                                            marks={i: '{:.0f}'.format(i) for i in np.linspace(0, 50, 20)}
                                           ),
                                html.Br(),
                                html.Br(),
                                html.Label('Select explanation confidence level ($p$)'),
                                dcc.Slider(id='proba',
                                           min=.51,
                                           max=.99,
                                           step=.01,
                                           value=.9,
                                           marks={i: '{:.1f}'.format(i) for i in np.linspace(0.51, .99, 5)},
                                           ),
                                html.Br(),
                                html.Br(),
                                html.Button('generate new noise instance', id='submit-val', n_clicks=0, style={
                                    "background-color": "whitesmoke",
                                    "borderRadius": 5,
                                    "border": 'thin lightgrey solid',
                                    "padding": "0px 5px",
                                    "margin": "0px 0px",
                                    "text-align": "center",
                                    "display": "inline-block"}),
                                #    ],
                                # )
                            ],
                            style={"float": "left", "width": '100%', "margin": '0.5%',
                                   'border': 'thin lightgrey solid',
                                   'display': 'inline-block', 'height': 'auto',
                                   "padding": 10, "borderRadius": 5, 'flex': 1, "clear": "both"}),
                        html.Div(
                            id="right-column",
                            children=[
                                dcc.Graph(id='privatesvmplotly'),
                            ],
                            style={"width": '100%', "margin": '0.5%',
                                   'border': 'thin lightgrey solid', 'display': 'inline-block', 'height': 'auto',
                                   'background-color': '#e0f0e0',
                                   "padding": 10, "borderRadius": 5}),
                        html.Div(
                            id="demo2",
                            children=[
                                html.H5('Application on Breast Cancer Wisconsin (Diagnostic) Data Set'),
                                dcc.Markdown('The publicly available UCI Breast Cancer Wisconsin (Diagnostic) dataset \[DG17\]'
                                             ' includes 569 instances, each with 30 features and the binary diagnosis: '
                                             'benign (class -1) or malignant (class 1).'),
                                dcc.Markdown('We randomly split the dataset once into a training (70% of total) and a '
                                             'test set (30% of total). The results were qualitatively similar for '
                                             'different random splits with same splitting ratio. Moreover, we normalize '
                                             'the training data to have zero mean and unit variance, and the calculated '
                                             'normalization parameters are applied to the test data. Next, a feature '
                                             'mapping $\phi$ is generated using the Radial Basis Function (RBF) kernel '
                                             'approximation in \[RR07\] with dimensions F=100.')
                                ],
                                style={"float": "left", "width": '100%', "margin": '.5%', 'border': '0 lightgrey solid',
                                        'display': 'inline-block', 'height': 'auto',
                                        "padding": 10, "borderRadius": 5, 'flex': 0}),
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Label('Select Sample'),
                                        dcc.Dropdown(
                                            id="testsample_bc",
                                            options=[
                                                    {'label': i, 'value': i} for i in neg_samples
                                                    ],
                                            clearable=False,
                                            searchable=False,
                                            value=neg_samples[0],
                                        ),
                                        html.Br(),
                                        html.Label(r'Select privacy parameter value ($\beta$)'),
                                        dcc.Slider(id='beta_bc',
                                                   min=0.25,
                                                   max=50,
                                                   step=.25,
                                                   value=5,
                                                   marks={i: '{:.0f}'.format(i) for i in np.linspace(0, 50, 20)}
                                                   ),

                                        html.Br(),
                                        html.Label('Select explanation confidence level ($p$)'),
                                        dcc.Slider(id='proba_bc',
                                               min=.51,
                                               max=.99,
                                               step=.01,
                                               value=.9,
                                               marks={i: '{:.1f}'.format(i) for i in np.linspace(0.51, .99, 5)}
                                               ),
                                        html.Br(),
                                        ],
                                    style={"width": '50%', "margin": '0', "float": "left", 'border': '0 lightgrey solid',
                                       'display': 'inline-block', 'height': 'auto',
                                       "padding": 0, "borderRadius": 5, 'flex': 0}),
                                html.Div(
                                    children=[
                                        html.Label('Classification:'),
                                        html.Div(children=[html.Span('True label of selected instance: '), html.Span(id='true')]),
                                        html.Div(children=[html.Span('Selected instance classification with non-private SVM: '), html.Span(id='nonprivsvm')]),
                                        html.Div(children=[html.Span('Selected instance classification with private SVM: '),
                                                           html.Span(id='privsvm')]),
                                        html.Div(children=[html.Span('Robust CE classification with non-private SVM: '),
                                                           html.Span(id='cesvm')]),],
                                    style={"width": '49%', "margin": '0', "float": "right", 'border': '0 lightgrey solid',
                                       'display': 'inline-block', 'height': 'auto', 'background-color': '#f0e0f0',
                                       "padding": 10, "borderRadius": 5, 'flex': 0}),
                            ],
                            style={"width": '100%', "margin": '0.5%', 'border': 'thin lightgrey solid',
                                   'display': 'inline-block', 'height': 'auto',
                                   "padding": 10, "borderRadius": 5, 'flex': 0}
                        ),
                        html.Div(
                            id="full-column",
                            children=[
                                html.Label('Select two features'),
                                html.Br(),
                                html.Div(
                                    dcc.Dropdown(
                                        id='xaxis_column',
                                        options=[{'label': features_labels[i], 'value': i} for i in
                                                 range(len(features_labels))],
                                        value='0'
                                    ),
                                    style={"width": '49.5%', "float": "left", "margin": '0%',
                                           'border': '0px solid black', 'display': 'inline-block'}
                                ),

                                html.Div(
                                    dcc.Dropdown(
                                        id='yaxis_column',
                                        options=[{'label': features_labels[i], 'value': i} for i in
                                                 range(len(features_labels))],
                                        value='1'
                                    ),
                                    style={"float": "right", "width": '49.5%', "margin": '0%',
                                           'border': '0px solid black', 'display': 'inline-block'}
                                ),
                            ],
                            style={"width": '100%', "margin": '0.5%', 'border': 'thin lightgrey solid',
                                   'display': 'inline-block', 'height': 'auto', 'background-color': '#e0f0e0',
                                   "padding": 10, "borderRadius": 5, 'flex': 0}
                        ),

                        html.Div(
                            id="full-column",
                            children=[
                                dcc.Graph(id='cfplot')
                            ],
                            style={"width": '100%', "margin": '0.5%', 'border': 'thin lightgrey solid',
                                   'display': 'inline-block', 'height': 'auto', 'background-color': '#e0f0e0',
                                   "padding": 10, "borderRadius": 5, 'flex': 0}
                        ),
                        html.Div(
                            id="right-column",
                            children=[
                                html.H5('Feature Values of Selected Instance and Counterfactual Explanation', style={}),
                                dcc.Graph(id='featurevaluesplot')
                            ],
                            style={"width": '100%', "margin": '0.5%', 'border': 'thin lightgrey solid',
                                   'display': 'inline-block', 'height': 'auto',
                                   "padding": 10, "borderRadius": 5, 'flex': 0}
                        ),

                        html.Div(
                            children=[
                                html.H5('Publications'),
                                dcc.Markdown(
                                    'R. Mochaourab, S. Sinha, S. Greenstein, and P. Papapetrou, “Robust Counterfactual Explanations for Privacy-Preserving SVMs,” *International Conference on Machine Learning (ICML 2021), Workshop on Socially Responsible Machine Learning* ([link](https://icmlsrml2021.github.io/paper.html)), Jul. 2021. ([arXiv](https://arxiv.org/abs/2102.03785))'),
                                dcc.Markdown(
                                    'S. Greenstein, P. Papapetrou, R. Mochaourab, “Embedding Human Values into Artificial Intelligence,” in De Vries, Katja (ed.), *De Lege*, Uppsala University, 2022.'),
                            ],
                            style={"float": "left", "width": '100%', "margin": '0.5%', 'border': 'thin lightgrey solid',
                                   'display': 'inline-block', 'background-color': '#FFF',
                                   "padding": 10, "borderRadius": 5, 'flex': 1}),

                        html.Div(
                            children=[
                                html.H5('References'),
                                html.Table([
                                    html.Tr([
                                        html.Td('[DR14]', style={"vertical-align": "top", "padding-right": "5px"}),
                                        html.Td('Cynthia Dwork and Aaron Roth. "The algorithmic foundations of differential privacy." Found. Trends Theor. Comput. Sci., 9(3–4):211–407, August 2014.'),
                                        ]),
                                    html.Tr([
                                        html.Td('[RBHT12]', style={"vertical-align": "top", "padding-right": "5px"}),
                                        html.Td('Benjamin I. P. Rubinstein, Peter L. Bartlett, Ling Huang, and Nina Taft. "Learning in a large function space: Privacy-preserving mechanisms for SVM learning." Journal of Privacy and Confidentiality, 4(1), July 2012.'),
                                        ]),
                                    html.Tr([
                                        html.Td('[WMR18]', style={"vertical-align": "top", "padding-right": "5px"}),
                                        html.Td('Sandra Wachter, Brent Mittelstadt, and Chris Russell. "Counterfactual explanations without opening the black box: Automated decisions and the GDPR." Harvard Journal of Law & Technology, 31(2), 2018.'),
                                    ]),
                                    html.Tr([
                                        html.Td('[DG17]', style={"vertical-align": "top", "padding-right": "5px"}),
                                        html.Td('D. Dua and C. Graff. "UCI machine learning repository." University of California, Irvine, School of Information and Computer Sciences, 2017. URL: http: //archive.ics.uci.edu/ml.'),
                                    ]),
                                    html.Tr([
                                        html.Td('[RR07]', style={"vertical-align": "top", "padding-right": "5px"}),
                                        html.Td('A. Rahimi, and B. Recht. "Random features for large-scale kernel machines." In Proceedings of the 20th International Conference on Neural Information Processing Systems, NIPS’07, pp. 1177–1184.'),
                                    ]),
                                ]),
                                ],
                                style={"float": "left", "width": '99%', "margin": '0.5%', 'border': '0 lightgrey solid',
                                        'display': 'inline-block',
                                        "padding": 10, "borderRadius": 5, 'flex': 1})
                    ],
                    style={'border': '0px solid white'}),
            ],
        )
    ],
),

@app.callback(
    Output(component_id='element-to-hide1', component_property='style'),
    Output(component_id='element-to-hide2', component_property='style'),
    Output(component_id='element-to-hide3', component_property='style'),
    Output(component_id='element-to-show1', component_property='style'),
    [Input(component_id='check_tech', component_property='value')])
def show_hide_element(visibility):
    if visibility:
        return {"float": "left", "width": '100%', "margin": '.5%', 'border': 'thin lightgrey solid',
                 'display': 'inline-block','height': 'auto', "padding": 10, "borderRadius": 5, 'flex': 1,
                 "clear": "both"}, \
               {"float": "left", "width": '100%', "margin": '.5%', 'border': 'thin lightgrey solid',
                'display': 'inline-block','height': 'auto', "padding": 10, "borderRadius": 5, 'flex': 1, "clear": "both"},\
                {"float": "left", "width": '100%', "margin": '.5%', 'border': 'thin lightgrey solid', 'display': 'inline-block',
                 'height': 'auto', "padding": 10, "borderRadius": 5, 'flex': 1, "clear": "both"},\
                {"float": "left", "width": '100%', "margin": '.5%', 'border': '0 lightgrey solid', 'display': 'inline-block',
                 'height': 'auto', "padding": 10, "borderRadius": 5, 'flex': 1, "clear": "both"},
    else:
        return {"float": "left", "width": '100%', "margin": '.5%', 'border': 'thin lightgrey solid',
                 'display': 'none','height': 'auto', "padding": 10, "borderRadius": 5, 'flex': 1,
                 "clear": "both"}, \
               {"float": "left", "width": '100%', "margin": '.5%', 'border': 'thin lightgrey solid',
                'display': 'none','height': 'auto', "padding": 10, "borderRadius": 5, 'flex': 1, "clear": "both"},\
                {"float": "left", "width": '100%', "margin": '.5%', 'border': 'thin lightgrey solid', 'display': 'none',
                 'height': 'auto', "padding": 10, "borderRadius": 5, 'flex': 1, "clear": "both"},\
                {"float": "left", "width": '100%', "margin": '.5%', 'border': '0 lightgrey solid', 'display': 'inline-block',
                 'height': 'auto', "padding": 10, "borderRadius": 5, 'flex': 1, "clear": "both"},

@app.callback(
    Output(component_id='privatesvmplotly', component_property='figure'),
    [Input('privatesvmplotly', 'clickData'),
     Input('submit-val', 'n_clicks'),
     Input(component_id='beta', component_property='value'),
     Input(component_id='proba', component_property='value')])
def update_image_src(clickData, btn1, beta, proba):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    seed = 10
    if clickData is None:
        datapoint = np.array([0, 0])
    else:
        datapoint = np.array([clickData['points'][0]['x'], clickData['points'][0]['y']])
    # print('clickData')
    # print(clickData)
    if 'submit-val' in changed_id:
        seed = random.randint(1, 99)
    else:
        seed = 10
    fig = compute_explanation_maths(datapoint.reshape(1, -1), beta, seed, proba)
    return fig


@app.callback(
    [Output(component_id='featurevaluesplot', component_property='figure'),
     Output(component_id='cfplot', component_property='figure'),
     Output(component_id='true', component_property='children'),
     Output(component_id='nonprivsvm', component_property='children'),
     Output(component_id='privsvm', component_property='children'),
     Output(component_id='cesvm', component_property='children')
     ],
    [Input(component_id='testsample_bc', component_property='value'),
     Input(component_id='beta_bc', component_property='value'),
     Input(component_id='proba_bc', component_property='value'),
     Input(component_id='xaxis_column', component_property='value'),
     Input(component_id='yaxis_column', component_property='value')])
def update_image_src(testsample, beta_bc, proba_bc, xaxis_column, yaxis_column):
    return get_freq_graph_bc(testsample, beta_bc, proba_bc, xaxis_column, yaxis_column)

# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
