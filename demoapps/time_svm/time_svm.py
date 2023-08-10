import sys,os
THIS_FILEPATH = os.path.dirname(__file__) #str(Path().absolute())
sys.path.append(os.path.join(THIS_FILEPATH, "."))
import dash_html_components as html
import base64
#from alibi.explainers import CounterFactualProto
from collections import defaultdict
import dill
import plotly.graph_objects as go
import sys, os
import dash
import dash_core_components as dcc
import dash_html_components as html
import joblib
import numpy as np
from dash.dependencies import Input, Output, State
import pandas as pd
from pathlib import Path
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import utils.dash_reusable_components as drc
from .utils.gs import growingspheres
from extremum_config import app

pos_image_filename = Path(__file__).resolve().parent / 'utils/pos_class.png'
pos_encoded_image = base64.b64encode(open(pos_image_filename, 'rb').read())
neg_image_filename = Path(__file__).resolve().parent / 'utils/neg_class.png'
neg_encoded_image = base64.b64encode(open(neg_image_filename, 'rb').read())

def get_data():
    '''
    Returns X_train, X_test, X_train_freq_amp, X_train_freq_ang, y_train, X_test_freq_amp, X_test_freq_ang, y_test
    '''
    data_train, meta = arff.loadarff(Path(__file__).resolve().parent / 'utils/ECG200_TRAIN.arff')
    data_test, meta = arff.loadarff(Path(__file__).resolve().parent / 'utils/ECG200_TEST.arff')
    number_train_samples = len(data_train)
    number_test_samples = len(data_test)
    length = len(data_train[1]) - 1  # time sequence length

    X_train = np.empty((number_train_samples, length))
    X_test = np.empty((number_test_samples, length))
    y_train = np.empty((number_train_samples))
    y_test = np.empty((number_test_samples))

    for index, sample in enumerate(data_train):
        sample_list = list(sample)
        sample_list = sample_list[:-1]
        X_train[index] = sample_list
        y_train[index] = sample[-1]

    for index, sample in enumerate(data_test):
        sample_list = list(sample)
        sample_list = sample_list[:-1]
        X_test[index] = sample_list
        y_test[index] = sample[-1]

    maximum_number_coefficients = int(length / 2 + 1)

    X_train_freq_real = np.empty((number_train_samples, maximum_number_coefficients))
    X_train_freq_imag = np.empty((number_train_samples, maximum_number_coefficients))
    X_train_freq_amp = np.empty((number_train_samples, maximum_number_coefficients))
    X_train_freq_ang = np.empty((number_train_samples, maximum_number_coefficients))

    X_test_freq_real = np.empty((number_test_samples, maximum_number_coefficients))
    X_test_freq_imag = np.empty((number_test_samples, maximum_number_coefficients))
    X_test_freq_amp = np.empty((number_test_samples, maximum_number_coefficients))
    X_test_freq_ang = np.empty((number_test_samples, maximum_number_coefficients))

    for idx in range(number_train_samples):
        X_train_freq_real[idx] = np.real(np.fft.rfft(X_train[idx]))
        X_train_freq_imag[idx] = np.imag(np.fft.rfft(X_train[idx]))
        X_train_freq_amp[idx] = np.absolute(np.fft.rfft(X_train[idx]))
        X_train_freq_ang[idx] = np.angle(np.fft.rfft(X_train[idx]))

    for idx in range(number_test_samples):
        X_test_freq_real[idx] = np.real(np.fft.rfft(X_test[idx]))
        X_test_freq_imag[idx] = np.imag(np.fft.rfft(X_test[idx]))
        X_test_freq_amp[idx] = np.absolute(np.fft.rfft(X_test[idx]))
        X_test_freq_ang[idx] = np.angle(np.fft.rfft(X_test[idx]))
    L = 20  # number of coefficients selected as features < maximum_number_coefficients

    X_train_first_features = np.empty((number_train_samples, 2 * L))
    X_test_first_features = np.empty((number_test_samples, 2 * L))

    X_train_first_features[:, 0::2] = X_train_freq_real[:, :L]
    X_train_first_features[:, 1::2] = X_train_freq_imag[:, :L]
    X_test_first_features[:, 0::2] = X_test_freq_real[:, :L]
    X_test_first_features[:, 1::2] = X_test_freq_imag[:, :L]
    #return x_train_first_features, x_freq_angle_train, y_train, x_test_first_features, x_freq_angle_test, y_test
    return X_train, X_test, X_train_first_features, X_train_freq_real, X_train_freq_imag, y_train,\
           X_test_first_features, X_test_freq_real, X_test_freq_imag, y_test

def get_dataframe():
    '''
    Returns df_train, df_test
    '''
    data_train, meta = arff.loadarff(Path(__file__).resolve().parent / 'utils/ECG200_TRAIN.arff')
    data_test, meta = arff.loadarff(Path(__file__).resolve().parent / 'utils/ECG200_TEST.arff')
    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)
    return df_train, df_test

def create_data_combined():
    data_train, meta = arff.loadarff(Path(__file__).resolve().parent / 'utils/ECG200_TRAIN.arff')
    data_test, meta_test = arff.loadarff(Path(__file__).resolve().parent / 'utils/ECG200_TEST.arff')
    # combine training and test data, to be split arbitrarily later
    data_com = np.concatenate((data_train, data_test), axis=0)
    data_df = pd.DataFrame(data_com)
    data_numpy_array = data_df.to_numpy()
    df_numpy_save_loc = Path(__file__).resolve().parent / 'utils/ecg_data_array.npy'
    np.save(df_numpy_save_loc, data_numpy_array, allow_pickle=True)

def get_data_combined_and_train_test():
    df_numpy_save_loc = Path(__file__).resolve().parent / 'utils/ecg_data_array.npy'
    data_df = np.load(df_numpy_save_loc, allow_pickle=True)
    data_df = pd.DataFrame(data=data_df, columns=['att' + str(i) for i in range(1, data_df.shape[1] + 1)])
    data_df = data_df.rename(columns={"att97": "target"})
    df = data_df.sample(n=len(data_df), random_state=13).reset_index(drop=True)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=13)
    df_test = df_test.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    return  df, df_train, df_test

def new_convert_to_DFT_train_test():
    _, df_train, df_test = get_data_combined_and_train_test()
    x_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values.astype(int)
    length = df_train.shape[1]-1
    x_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values.astype(int)
    maximum_number_coefficients = int(length/2+1)
    x_freq_train = np.empty((x_train.shape[0], maximum_number_coefficients), dtype=np.float64) #dtype=np.complex128
    x_freq_angle_train = np.empty((x_train.shape[0], maximum_number_coefficients), dtype=np.float64) #dtype=np.complex128
    x_freq_test = np.empty((x_test.shape[0], maximum_number_coefficients), dtype=np.float64) #dtype=np.complex128
    x_freq_angle_test = np.empty((x_test.shape[0], maximum_number_coefficients), dtype=np.float64) #dtype=np.complex128
    for idx in range(x_train.shape[0]):
        x_freq_train[idx] = np.absolute(np.fft.rfft(x_train[idx]))
        x_freq_angle_train[idx] = np.angle(np.fft.rfft(x_train[idx]))
    for idx in range(x_test.shape[0]):
        x_freq_test[idx] = np.absolute(np.fft.rfft(x_test[idx]))
        x_freq_angle_test[idx] = np.angle(np.fft.rfft(x_test[idx]))
    L = 50  # number of coefficients selected as features < maximum_number_coefficients
    x_train_first_features = x_freq_train[:, :L]
    x_test_first_features = x_freq_test[:, :L]
    return x_train_first_features, x_freq_angle_train, y_train, x_test_first_features, x_freq_angle_test, y_test

def svmexplain(cf, sample_id, clf_svm, x_train_freq, x_train_angle, y_train_freq, x_test_freq, x_test_angle, y_test_freq):
    print('Computing SVM explain explanation ...')
    _, df_train, df_test = get_data_combined_and_train_test()
    x_train = df_train.iloc[:, :-1].values
    x_test = df_test.iloc[:, :-1].values
    idx = sample_id
    #print(idx, clf_svm.predict([x_test_freq[7]]))
    sample_from = x_test_freq[idx]
    sample_from_pred = clf_svm.predict(sample_from.reshape(1, -1))[0]
    print("predicted class: ", sample_from_pred)
    sample_from_time_domain = x_test[idx]
    EN = np.linalg.norm(x_train_freq - sample_from, ord=2, axis=1)
    EN[clf_svm.predict(x_train_freq) == sample_from_pred] = np.inf
    idx_sample_proto = np.argmin(EN)
    sample_proto = x_train_freq[idx_sample_proto]
    print(clf_svm.predict(sample_proto.reshape(1, -1))[0])
    sample_proto_pred = y_train_freq[idx_sample_proto]  # not predicted but true label
    sample_proto_time_domain = x_train[idx_sample_proto]
    ub = sample_from.copy()
    lb = sample_proto.copy()
    length = df_train.shape[1] - 1
    maximum_number_coefficients = int(length / 2 + 1)
    L = 50
    while np.linalg.norm(ub - lb) > 0.0001:
        sample_to = (ub + lb) / 2
        label = clf_svm.predict(sample_to.reshape(1, -1))

        if label == sample_from_pred:
            ub = sample_to.copy()
        else:
            lb = sample_to.copy()
    sample_to_freq_amp = np.concatenate((sample_to, np.zeros(maximum_number_coefficients - L+1)))
    sample_to_freq_ang = x_train_angle[idx, :]  # choose the same angles as sample_from
    sample_to_time_domain = np.fft.irfft(sample_to_freq_amp * np.exp(1j * sample_to_freq_ang))
    fig = go.Figure()
    fig.add_scatter(name="Original timeseries", y=sample_from_time_domain)
    if 'svmex' in cf:
        fig.add_scatter(name="Counterfactual using SVM explain", y=sample_to_time_domain)
    if 'tt' in cf:
        fig.add_scatter(name="knn timeseries tweaking", y=sample_proto_time_domain)
    return fig


def transform_to_time_domain(freq_domain_signal, N, source_noise_real=None, source_noise_imag=None, **kwargs):
    real = freq_domain_signal[0::2]
    imag = freq_domain_signal[1::2]
    L = len(real)
    if source_noise_real is None:
        hf = np.zeros(N - L)
    else:
        hf = source_noise_real + 1j * source_noise_imag
    X_lp = real + 1j * imag
    X = np.concatenate((X_lp, hf[L - N:]))
    return np.fft.irfft(X)

def new_svmexplain(pred, cf, sample_id, cf_thresh, clf, X_train, X_test, X_train_first_features, X_train_freq_real,
                   X_train_freq_imag, y_train, X_test_first_features, X_test_freq_real, X_test_freq_imag, y_test):
    print('Computing SVM explain explanation ...')
    idx_selected = sample_id
    df_train, df_test = get_dataframe()
    length = df_train.shape[1] - 1
    maximum_number_coefficients = int(length / 2 + 1)
    L = 20
    df_train, df_test = get_dataframe()
    #print(idx, clf_svm.predict([x_test_freq[7]]))
    sample_from = X_test_first_features[idx_selected]
    sample_from_pred = clf.predict(sample_from.reshape(1, -1))[0]
    print("predicted class: ", sample_from_pred, "true class: ", y_test[idx_selected])

    sample_from_time_domain = X_test[idx_selected]
    sample_from_time_domain_rm_noise = transform_to_time_domain(sample_from, maximum_number_coefficients)

    EN = np.linalg.norm(X_train_first_features - sample_from, ord=2, axis=1)
    train_predict = clf.predict(X_train_first_features)
    EN[train_predict == sample_from_pred] = np.inf
    idx_sample_proto = np.argmin(EN)
    print("predicted class: ", train_predict[idx_sample_proto], "true class: ", y_train[idx_sample_proto])

    sample_proto = X_train_first_features[idx_sample_proto]
    sample_proto_pred = train_predict[idx_sample_proto]  # not predicted but true label
    sample_proto_time_domain = X_train[idx_sample_proto]

    sample_proto_time_domain_rm_noise = transform_to_time_domain(sample_proto, maximum_number_coefficients)

    # Bisection on all coefficients
    ub = sample_from.copy()
    lb = sample_proto.copy()

    while np.linalg.norm(ub - lb) > 0.0001:
        sample_to = (ub + lb) / 2
        label = clf.predict(sample_to.reshape(1, -1))

        if label == sample_from_pred:
            ub = sample_to.copy()
        else:
            lb = sample_to.copy()
        print("Label of sample_to:", label, ", Norm: ", np.linalg.norm(ub - lb))

    sample_to_time_domain_rm_noise = transform_to_time_domain(sample_to, maximum_number_coefficients)
    sample_to_time_domain_w_noise = transform_to_time_domain(sample_to, maximum_number_coefficients,
                                                             X_test_freq_real[idx_selected],
                                                             X_test_freq_imag[idx_selected])


    o=0.2
    base_line_o = 1
    svm_ex_color = "orangered"
    knn_color = 'orange'
    fig = go.Figure()
    fig.add_scatter(name="Original timeseries", y=sample_from_time_domain, mode="lines", marker=dict(color="slateblue"))
    fig.update_traces(showlegend=True)
    fig.update_layout(#legend=dict(yanchor="top", xanchor="left", x=.1, y=0.000001),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
                      xaxis=dict(title='time'),
                      yaxis=dict(title='amplitude',
                                #titlefont=dict(
                                #family='Courier New, monospace',
                                #size=18,
                                #color='#7f7f7f'
                                #)
                                ))

    if 'svmex' in cf:
        if cf_thresh == 0:
            base_line_o = 1
        else:
            base_line_o = 0.4
        fig.add_scatter(name="Counterfactual using SVM explain", y=sample_to_time_domain_w_noise, opacity = base_line_o, mode='lines',
                        marker=dict(color=svm_ex_color))
        for i in range(len(sample_from_time_domain)):
            if abs(sample_from_time_domain[i] - sample_to_time_domain_w_noise[i]) > cf_thresh and i < 95:
                # print(abs(sample_from_time_domain[i]-sample_to_time_domain[i]))
                fig.add_scatter(mode='lines', x=np.array([i - 1, i, i + 1]), y=np.array(
                    [sample_to_time_domain_w_noise[i - 1], sample_to_time_domain_w_noise[i], sample_to_time_domain_w_noise[i + 1]]),
                                marker=dict(color=svm_ex_color), showlegend=False)

    if 'proto' in cf:
        #fig.add_scatter(name="knn timeseries tweaking", y=sample_proto_time_domain)
        if cf_thresh == 0:
            base_line_o = 1
        else:
            base_line_o = 0.4
        fig.add_scatter(name="knn timeseries tweaking", y=sample_proto_time_domain, opacity = base_line_o, mode='lines',
                        marker=dict(color=knn_color))
        for i in range(len(sample_from_time_domain)):
            if abs(sample_from_time_domain[i] - sample_proto_time_domain[i]) > cf_thresh and i < 95:
                # print(abs(sample_from_time_domain[i]-sample_to_time_domain[i]))
                fig.add_scatter(mode='lines', x=np.array([i - 1, i, i + 1]), y=np.array(
                    [sample_proto_time_domain[i - 1], sample_proto_time_domain[i], sample_proto_time_domain[i + 1]]),
                                marker=dict(color=knn_color), showlegend=False)
    return fig


def load_model_svm():
    model_path = Path(__file__).resolve().parent / 'utils/svm_model.pkl'
    model = joblib.load(model_path)
    return model

def get_myocardial_samples_test():
    clf_svm = load_model_svm()
    svm_x_index = []
    neg_samples_index = {}
    #_, df_train, df_test = get_data_combined_and_train_test()
    df_train, df_test = get_dataframe()
    #print(df_test.index.values)
    neg_samples = df_test.loc[df_test['target']==b'-1'].index.values
    neg_samples_index = df_test.index[df_test['target']==b'-1'].tolist()
    print(df_test.index.values)
    #neg_samples = df_test.index.values
    print(neg_samples)
    #return list(neg_samples)
    return df_test.index.values

def get_lime_explainer():
    model_path = Path(__file__).resolve().parent / 'utils/lime_explainer.pkl'
    with open(model_path, "rb") as dill_file:
        lime_explainer=dill.load(dill_file)
    return lime_explainer

def get_lime_explanation_test_svm(sample_id, clf, x_train_freq, x_train_angle, y_train_freq, x_test_freq, x_test_angle, y_test_freq):
    print('loading lime model ...')
    ex = get_lime_explainer()
    print('Model loaded. Computing lime explanation...')
    xp = ex.explain_instance(x_test_freq[sample_id], clf.predict_proba)
    print("Lime explanation computed.")
    obj = html.Iframe(
        # Javascript is disabled from running in an Iframe for security reasons
        # Static HTML only!!!
        srcDoc=xp.as_html(),
        width='100%',
        height='400px',
        style={'border': '2px #d3d3d3 solid'},
    )

    return obj

def get_growingspheres_cf(pred, svmexplain_fig, cfsel, sample_id, cf_thresh, clf, X_train, X_test, X_train_first_features,
                          X_train_freq_real, X_train_freq_imag, y_train, X_test_first_features, X_test_freq_real,
                          X_test_freq_imag, y_test):
    if 'gs' in cfsel:
        # print('Computing Growing spheres counterfactual ...')
        # df_train, df_test = get_dataframe()
        # idx_selected = sample_id
        # sample_from_time_domain = X_test[idx_selected]
        # #x_train = df_train.iloc[:, :-1].values
        # #x_test = df_test.iloc[:, :-1].values
        #
        # length = df_train.shape[1] - 1
        # maximum_number_coefficients = int(length / 2 + 1)
        # L = 20
        # predict_fn = lambda x: clf.predict(x)
        # a = growingspheres.GrowingSpheres(np.array(X_test_first_features[idx_selected]),
        #                                   predict_fn)  # n_in_layer=2000,first_radius=0.1,dicrease_radius=10,
        #
        # c_gs = a.find_counterfactual()
        # #sample_to_freq_amp = np.concatenate((c_gs, np.zeros(maximum_number_coefficients - L + 1)))
        # #sample_to_freq_ang = x_train_angle[sample_id, :]  # choose the same angles as sample_from
        # sample_to_time_domain = transform_to_time_domain(c_gs, maximum_number_coefficients, X_test_freq_real[idx_selected], X_test_freq_imag[idx_selected])

        #svmexplain_fig.add_scatter(name="Growingspheres", y=sample_to_time_domain)
        idx_selected=sample_id
        file_path = Path(__file__).resolve().parent / 'utils/gs-cf.npy'
        sample_from_time_domain = X_test[idx_selected]
        cgs_cf=np.load(file_path, allow_pickle=True)
        sample_to_time_domain = cgs_cf[idx_selected]
        print(len(sample_to_time_domain), len(sample_from_time_domain))
        o = 0.2
        base_line_o = 1
        c="goldenrod"
        if cf_thresh == 0:
            base_line_o = 1
        else:
            base_line_o = 0.4
        svmexplain_fig.add_scatter(name="Growing spheres", y=sample_to_time_domain, opacity = base_line_o, mode='lines',
                        marker=dict(color=c))
        for i in range(len(sample_from_time_domain)):
            if abs(sample_from_time_domain[i] - sample_to_time_domain[i]) > cf_thresh and i < 95:
                # print(abs(sample_from_time_domain[i]-sample_to_time_domain[i]))
                svmexplain_fig.add_scatter(mode='lines', x=np.array([i - 1, i, i + 1]), y=np.array(
                    [sample_to_time_domain[i - 1], sample_to_time_domain[i], sample_to_time_domain[i + 1]]),
                                marker=dict(color=c), showlegend=False)

        return svmexplain_fig
    else:
        return svmexplain_fig

def get_cf_guided_proto(pred, svmexplain_fig, cfsel, sample_id, cf_thresh, clf, X_train, X_test, X_train_first_features,
                          X_train_freq_real, X_train_freq_imag, y_train, X_test_first_features, X_test_freq_real,
                          X_test_freq_imag, y_test):
    if 'cfp' in cfsel:
        print('Computing cf explanation guided by prototype...')
    #     _, df_train, df_test = get_data_combined_and_train_test()
    #     eg = x_test_freq[sample_id]
    #     eg_X = eg.reshape((1,) + eg.shape)
    #     shape = eg_X.shape
    #     eps = (.05, .05)
    #     predict_fn = lambda x: clf.predict_proba(x)
    #     print("computing now")
    #     cf = CounterFactualProto(predict_fn, shape, use_kdtree=True, theta=50., max_iterations=1,
    #                              feature_range=(x_train_freq.min(axis=0), x_train_freq.max(axis=0)),
    #                              c_init=1., c_steps=1, eps=eps)
    #     print("initialized#########")
    #     cf.fit(x_train_freq)
    #     explanation = cf.explain(eg_X)
    #     sample_to_freq_ang = x_train_angle[sample_id, :]
    #     cf_tdomain = np.fft.irfft(explanation['cf']['X'][0] * np.exp(1j * sample_to_freq_ang))
    #     print("CF guided by prototype computed.")
    #     svmexplain_fig.add_scatter(name="Growingspheres", y=cf_tdomain)
    #     return svmexplain_fig
    # else:
    #     return svmexplain_fig
        idx_selected = sample_id
        file_path = Path(__file__).resolve().parent / 'utils/cfp-cf.npy'
        sample_from_time_domain = X_test[idx_selected]
        cgs_cf = np.load(file_path, allow_pickle=True)
        sample_to_time_domain = cgs_cf[idx_selected]
        print(len(sample_to_time_domain), len(sample_from_time_domain))
        o = 0.2
        base_line_o = 1
        c = "saddlebrown"
        if cf_thresh == 0:
            base_line_o = 1
        else:
            base_line_o = 0.4
        svmexplain_fig.add_scatter(name="Growing spheres", y=sample_to_time_domain, opacity=base_line_o, mode='lines',
                                   marker=dict(color=c))
        for i in range(len(sample_from_time_domain)):
            if abs(sample_from_time_domain[i] - sample_to_time_domain[i]) > cf_thresh and i < 95:
                # print(abs(sample_from_time_domain[i]-sample_to_time_domain[i]))
                svmexplain_fig.add_scatter(mode='lines', x=np.array([i - 1, i, i + 1]), y=np.array(
                    [sample_to_time_domain[i - 1], sample_to_time_domain[i], sample_to_time_domain[i + 1]]),
                                           marker=dict(color=c), showlegend=False)

        return svmexplain_fig
    else:
        return svmexplain_fig

def get_knn_cf(pred, fig, cfsel, sample_id, cf_thresh, X_test, X_test_first_features, X_test_freq_real,
                          X_test_freq_imag):
    if 'tt' in cfsel:

        idx_selected=sample_id
        df_train, df_test = get_dataframe()
        length = df_train.shape[1] - 1
        maximum_number_coefficients = int(length / 2 + 1)
        L = 20
        sample_from_time_domain = X_test[idx_selected]
        file_path = Path(__file__).resolve().parent / 'utils/knn-cf.npy'
        x_counterfactual = np.load(file_path)
        sample_to_time_domain = transform_to_time_domain(x_counterfactual[idx_selected], maximum_number_coefficients,
                                          X_test_freq_real[idx_selected], X_test_freq_imag[idx_selected])
        o = 0.2
        base_line_o = 1
        c='darkolivegreen'
        if cf_thresh == 0:
            base_line_o = 1
        else:
            base_line_o = 0.4
        fig.add_scatter(name="Nearest neighbor counterfactual", y=sample_to_time_domain, opacity=base_line_o,
                                   mode='lines',
                                   marker=dict(color=c))
        for i in range(len(sample_from_time_domain)):
            if abs(sample_from_time_domain[i] - sample_to_time_domain[i]) > cf_thresh and i < 95:
                # print(abs(sample_from_time_domain[i]-sample_to_time_domain[i]))
                fig.add_scatter(mode='lines', x=np.array([i - 1, i, i + 1]), y=np.array(
                    [sample_to_time_domain[i - 1], sample_to_time_domain[i], sample_to_time_domain[i + 1]]),
                                           marker=dict(color=c), showlegend=False)

        return fig

    else:
        return fig

def get_summary(svmexplain_fig, datasummary):
    df_train, _ = get_dataframe()
    o = 0.7
    df_train_neg = df_train[df_train['target'] == b'-1']
    df_train_pos = df_train[df_train['target'] == b'1']
    print(datasummary)
    if 'neg_mean' in datasummary:
        svmexplain_fig.add_scatter(name="Myocardial Infraction samples mean", y=df_train_neg.mean()[:-1], opacity=o,
                                   mode="lines", marker=dict(color="deeppink"))
    if 'neg_percentiles' in datasummary:
        #svmexplain_fig.add_scatter(name="Myocardial infraction 10 percentile", y=df_train_neg.quantile(.1)[:-1], opacity=o)
        #svmexplain_fig.add_scatter(name="Myocardial infraction 90 percentile", y=df_train_neg.quantile(.9)[:-1], opacity=o)
        svmexplain_fig.add_trace(go.Scatter(
            x=np.concatenate([np.arange(96),np.arange(95,-1,-1)]),
            y=pd.concat([df_train_neg.quantile(.9)[:-1], df_train_neg.quantile(.1)[:-1][::-1]]),
            fill='toself', name='10 & 90 percentile range Myocardial Infraction', opacity=o, mode="lines", marker=dict(color="darksalmon")))
    if 'pos_mean' in datasummary:
        svmexplain_fig.add_scatter(name="Normal heartbeat samples mean", y=df_train_pos.mean()[:-1], opacity=o, mode="lines", marker=dict(color="teal"))
    if 'pos_percentiles' in datasummary:
        #svmexplain_fig.add_scatter(name="Normal heartbeat 10 percentile", y=df_train_pos.quantile(.1)[:-1], opacity=o)
        #svmexplain_fig.add_scatter(name="Normal heartbeat 90 percentile", y=df_train_pos.quantile(.9)[:-1], opacity=o)
        svmexplain_fig.add_trace(go.Scatter(
            x=np.concatenate([np.arange(96), np.arange(95, -1, -1)]),
            y=pd.concat([df_train_pos.quantile(.9)[:-1], df_train_pos.quantile(.1)[:-1][::-1]]),
            fill='toself', name='10 & 90 percentile range Normal Heartbeat', opacity=o, mode="lines", marker=dict(color="aquamarine")))
    if len(datasummary) == 0:
        pass
    return svmexplain_fig

neg_samples = get_myocardial_samples_test()

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
                                    "Counterfactual Explanations for time series playground",
                                    #href="https://github.com/plotly/dash-svm",
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
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        html.Br(),
                                        html.H5("Explore training data summary"),
                                        dcc.Checklist(
                                            id="datasummary",
                                            options=[{'label': 'Myocardial infraction samples mean', 'value': 'neg_mean'},
                                                 {'label': '10 & 90 percentile band for Myocardial infraction samples',
                                                  'value': 'neg_percentiles'},
                                                 {'label': 'Normal heartbeat samples mean',
                                                  'value': 'pos_mean'},
                                                 {
                                                     'label': '10 & 90 percentile band for Normal heartbeat samples',
                                                     'value': 'pos_percentiles'}
                                                 ],
                                            value=[],#'neg_mean',
                                            labelStyle={'display': 'inline-block'}),
                                        html.Br(),
                                        html.H5("Select explanation method"),
                                        dcc.Checklist(
                                            id="cfsel",
                                            options=[
                                                {'label': 'SVM explain ' , 'value': 'svmex'},
                                                {'label': 'KNN timeseries tweaking ', 'value': 'tt'},
                                                {'label': 'Growing Spheres ', 'value': 'gs'},
                                                {'label': 'Counterfactual guided by prototype', 'value': 'cfp'},
                                                ],
                                            value=['svmex'],
                                            labelStyle={'display': "block"},
                                            style={}
                                            ),
                                        html.Br(),
                                        html.H5('Select Sample:', style={}),#'font-size':'9pt'
                                        dcc.Dropdown(
                                            id="samplesel",
                                            options=[
                                                    {'label': i, 'value': i} for i in neg_samples

                                            ],
                                            clearable=False,
                                            searchable=False,
                                            value=neg_samples[0],
                                        ),
                                        html.H5('Counterfactual threshold:', style={}),
                                        dcc.Slider(id='cfthreshold',
                                                    min=0,
                                                    max=3,
                                                    step=.1,
                                                    value=0,
                                                    marks={int(i) if i % 1 == 0 else i: '{:.1f}'.format(i) for i in np.arange(0, 5, 0.2)}
                                                    ),
                                    ],

                                )], style={"width":'50%', "margin": 0, 'border': '0px solid white', 'display': 'inline-block'}),
                        html.Div(
                            # className="three columns",
                            id="right-column",
                            children=[
                                drc.Card(
                                    id="svmexplain-card",
                                    children=[
                                        html.Br(),
                                        html.Div(id='pred'),
                                        #html.Img(id='image-svmexplain'),
                                        dcc.Graph(id='graph-plotly'),

                                        #html.Div(id='lime-plotly')
                                        #html.Img(id='image-lime'),
                                    ],

                                )],
                            style={"width":'50%',  "margin": "0", 'border': '0px solid white', 'display': 'inline-block'})],

                    style={'border':'0px solid white'}),

                        html.Div([
                                html.Br(),
                                drc.Card(
                                    id="theory",
                                    children=[
                                        html.H3("Dataset"),
                                        html.Strong("ECG 200 Dataset"),
                                        html.Div("This dataset was formatted by R. Olszewski as part of"
                                                 " his thesis 'Generalized feature extraction for structural pattern "
                                                 "recognition in time-series data' at Carnegie Mellon University, 2001."
                                                 " Each series traces the electrical activity recorded during one "
                                                 "heartbeat. The two classes are a normal heartbeat and a Myocardial "
                                                 "Infarction."),
                                    ],
                                ),
                                html.Br(),
                                drc.Card(
                                    id="motivation",
                                    children=[
                                        html.H3("DFT coefficients as features"),
                                        html.Div("Finding similarities between time series can be efficiently "
                                                 "done in the frequency domain by exploiting the discrete Fourier"
                                                 " transform. By assuming that the time series of length T is "
                                                 "periodic, each data point of the time series is decomposed into"
                                                 " T frequency components which are weighted with the Fourier "
                                                 "coefficients. Each Fourier coefficient can be considered as an "
                                                 "independent description of a sub-component of the whole time "
                                                 "series. This representation is often powerful enough to extract"
                                                 " the discriminant features of the time series efficiently.  "),
                                        html.Br(),
                                        html.Strong("Feature Attribution using Local Interpretable Model-Agnostic Explanations"
                                                    " (LIME)"),
                                        html.Div("LIME explains the prediction of any classifier or regressor in a "
                                                 "faithful way, by approximating it locally with an interpretable"
                                                 " model. It modifies a single data sample by tweaking the feature"
                                                 " values and observes the resulting impact on the output. It performs "
                                                 "the role of an 'explainer' to explain predictions from each data"
                                                 " sample. The output of LIME is a set of explanations representing the"
                                                 " contribution of each feature to a prediction for a single sample,"
                                                 " which is a form of local interpretability."),
                                        html.Br(),
                                        html.Div("Below are explanations generated using LIME, for a sample positively "
                                                 "classified and negatively classified by the classifier, "
                                                 "when DFT coefficients as features were used on ECG200 Dataset."),
                                        html.Caption("LIME explanation for sample classified positive by the classfier"),
                                        html.Img(src='data:image/png;base64,{}'.format(pos_encoded_image.decode()), style={'height':'60%', 'width':'60%'}),
                                        html.Caption("LIME explanation for sample classified negative by the classfier"),
                                        html.Img(src='data:image/png;base64,{}'.format(neg_encoded_image.decode()), style={'height':'60%', 'width':'60%'}),
                                    ],
                                ),
                                html.Br(),
                                    drc.Card(
                                    id="classifier",
                                    children=[
                                        html.H3("Classifier - Support Vector Machines"),
                                        html.Div("The support vector machine classifier learns to separate the two "
                                                 "classes of data points by finding the separating hyperplane which"
                                                 " minimizes the number of misclassified data points and ensures "
                                                 "sufficient robustness through a regularization term. We use a kernel"
                                                 " method (radial basis function) in order to suitable map the data"
                                                 " points to a large space in which the points can be better separated"
                                                 " by the linear classifier. "),
                                    ],
                                ),

                                    html.Br(),
                                    drc.Card(
                                    id="cftheory",
                                    children=[
                                        html.H3("Counterfactual Explanations"),
                                        html.Strong("Bisection Method (Mochaourab 2021)"),
                                        html.Div("The bisection method is a root finding algorithm which we use to "
                                                 "find a root for the SVM decision function. Those roots correspond"
                                                 " to points on the decision boundary which are hence candidates for "
                                                 "counterfactual explanations. "),
                                        html.Br(),
                                        html.Strong("Growing Spheres (Laugel 2019)"),
                                        html.Div(" Growing spheres is a post-hoc, model-agnostic interpretability method"
                                                 "that does not require any information about the classifier"
                                                 "or the data used to train the classifier. It uses a generative"
                                                 "approach that locally explores the input space of classifier "
                                                 "to find its decision boundary, eventually finding the "
                                                 "minimal change needed to alter the associated prediction"
                                                 "of the given sample"),
                                        html.Br(),
                                        html.Strong("KNN (Karlsson 2019)"),
                                        html.Div("KNN classifiers for time series rely on the used distance measure "
                                                 "between time series. We use the Euclidian distance in this demo. "
                                                 "The counterfactual explanation corresponds to closest time series "
                                                 "used as prototypes for KNN classification. "),
                                        html.Br(),
                                        # html.Strong("Shapelet tweaking (Karlsson 2019)"),
                                        # html.Div("Uses the generalized shapelet forest classifier. Through the"
                                        #          " partitioning of the time series within the shapelet trees, it is "
                                        #          "possible to determine shapelet-based time series counterfactual "
                                        #          "explanations by examining the suitability of the prediction paths of "
                                        #          "the classifier."),

                                    ],
                                ),
                                html.Br(),
                                drc.Card(
                                    id="references",
                                    children=[
                                        html.H3("References"),
                                        html.Div('R. Mochaourab, S. Sinha, S. Greenstein, P. Papapetrou, '
                                                 '"Robust Explanations for Private Support Vector Machines," '
                                                 'Preprint available on arXiv: https://arxiv.org/abs/2102.03785), Feb. 2021. '),
                                        html.Br(),
                                        html.Div('Isak Karlsson, Jonathan Rebane, Panagiotis Papapetrou, Aristides Gionis,'
                                                 ' “Locally and globally explainable time series tweaking,” Knowledge '
                                                 'and Information Systems, vol. 62, no. 5, pp. 1671– 1700, Aug. 2019. '),
                                        html.Br(),
                                        html.Div('T. Laugel, M. J. Lesot, C. Marsala, X. Renard, and M. Detyniecki. '
                                                 '"Inverse classification for comparison-based interpretability in '
                                                 'machine learning." arXiv preprint arXiv:1712.08443, 2017.'),
                                        html.Br(),
                                    ],
                                ),

                            ],
                        ),
                    ],
                )
            ],
        ),
#    ]
#)

@app.callback(
    Output(component_id='pred', component_property='children'),
   # Output(component_id='image-svmexplain', component_property='src'),
    Output(component_id='graph-plotly', component_property='figure'),
    #Output('lime-plotly', 'children'),
    [Input(component_id='cfsel', component_property='value'),
     Input(component_id='datasummary', component_property='value'),
     Input(component_id='samplesel', component_property='value'),
     Input(component_id='cfthreshold', component_property='value')])
def update_image_src(cfsel, datasummary, samplesel, cfthreshold):
    print()
    clf = load_model_svm()
    #x_train_freq, x_train_angle, y_train_freq, x_test_freq, x_test_angle, y_test_freq = new_convert_to_DFT_train_test()
    X_train, X_test, X_train_first_features, X_train_freq_real, X_train_freq_imag, y_train, \
    X_test_first_features, X_test_freq_real, X_test_freq_imag, y_test = get_data()
    pred = clf.predict(X_test_first_features[samplesel].reshape(1, -1))[0]
    print(cfsel)
    if pred == -1:
        fig = new_svmexplain(pred, cfsel, int(samplesel), cfthreshold, clf, X_train, X_test,
                                        X_train_first_features, X_train_freq_real, X_train_freq_imag, y_train,
                                        X_test_first_features, X_test_freq_real, X_test_freq_imag, y_test)
        fig = get_knn_cf(pred, fig, cfsel, int(samplesel), cfthreshold, X_test, X_test_first_features, X_test_freq_real, X_test_freq_imag)
        fig = get_growingspheres_cf(pred, fig, cfsel, int(samplesel), cfthreshold, clf,
                                                   X_train, X_test, X_train_first_features, X_train_freq_real,
                                                   X_train_freq_imag, y_train,
                                                   X_test_first_features, X_test_freq_real, X_test_freq_imag, y_test)
        fig = get_cf_guided_proto(pred, fig, cfsel, int(samplesel), cfthreshold, clf,
                                                   X_train, X_test, X_train_first_features, X_train_freq_real,
                                                   X_train_freq_imag, y_train,
                                                   X_test_first_features, X_test_freq_real, X_test_freq_imag, y_test)
        # lime_fig = get_lime_explanation_test_svm(int(samplesel), clf, x_train_freq, x_train_angle, y_train_freq, x_test_freq, x_test_angle, y_test_freq)
        # lime_filename = Path(__file__).resolve().parent / 'utils/lime_exp.png'
        # encoded_lime_image = base64.b64encode(open(lime_filename, 'rb').read())
        # cf_guided_proto_fig = get_cf_guided_proto(growingspheres_fig, cfsel, int(samplesel), clf, x_train_freq, x_train_angle,
        #                                            y_train_freq, x_test_freq, x_test_angle, y_test_freq)
        fig = get_summary(fig, datasummary)
        print("outputting files...")
        #'data:image/png;base64,{}'.format(encoded_lime_image.decode()),
        #return  'Classifier prediction: {}'.format(str(pred)), 'data:image/png;base64,{}'.format(encoded_svmexplain_image.decode()), 'data:image/png;base64,{}'.format(encoded_lime_image.decode())
        return 'Classifier prediction: {}'.format(str(pred)), fig#cf_guided_proto_fig, lime_fig
    else:
        sample_from_time_domain = X_test[samplesel]
        fig = go.Figure()
        fig.add_scatter(name="Original timeseries", y=sample_from_time_domain, mode="lines", marker=dict(color="slateblue"))
        fig.update_traces(showlegend=True)
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title='time'), yaxis=dict(title='amplitude'))
        fig = get_summary(fig, datasummary)
        return 'Classifier prediction: {}. No counterfactual needed!'.format(str(pred)), fig
# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)