import pandas as pd
import numpy as np
import cvxpy as cp

features_labels = ['radius_mean',
    'texture_mean' ,
    'perimeter_mean',
    'area_mean'     ,
    'smoothness_mean',
    'compactness_mean',
    'concavity_mean'   ,
    'concave points_mean',
    'symmetry_mean'       ,
    'fractal_dimension_mean',
    'radius_se'              ,
    'texture_se'              ,
    'perimeter_se'             ,
    'area_se'                   ,
    'smoothness_se'               ,
    'compactness_se'               ,
    'concavity_se'                 ,
    'concave points_se'            ,
    'symmetry_se'                  ,
    'fractal_dimension_se'         ,
    'radius_worst'                 ,
    'texture_worst'                ,
    'perimeter_worst'              ,
    'area_worst'                   ,
    'smoothness_worst'             ,
    'compactness_worst'            ,
    'concavity_worst'              ,
    'concave points_worst'         ,
    'symmetry_worst'               ,
    'fractal_dimension_worst'      ]

def load_wdbc():
    
    features_labels = ['radius_mean',                  
    'texture_mean' ,                 
    'perimeter_mean',               
    'area_mean'     ,                  
    'smoothness_mean',              
    'compactness_mean',             
    'concavity_mean'   ,            
    'concave points_mean',          
    'symmetry_mean'       ,         
    'fractal_dimension_mean',       
    'radius_se'              ,      
    'texture_se'              ,     
    'perimeter_se'             ,    
    'area_se'                   ,   
    'smoothness_se'               , 
    'compactness_se'               ,
    'concavity_se'                 ,
    'concave points_se'            , 
    'symmetry_se'                  ,
    'fractal_dimension_se'         ,
    'radius_worst'                 ,
    'texture_worst'                ,
    'perimeter_worst'              ,
    'area_worst'                   ,
    'smoothness_worst'             ,
    'compactness_worst'            ,
    'concavity_worst'              ,
    'concave points_worst'         ,
    'symmetry_worst'               ,
    'fractal_dimension_worst'      ]

    col_names = ['id',                           
    'diagnosis',                    
    'radius_mean',                  
    'texture_mean' ,                 
    'perimeter_mean',               
    'area_mean'     ,                  
    'smoothness_mean',              
    'compactness_mean',             
    'concavity_mean'   ,            
    'concave points_mean',          
    'symmetry_mean'       ,         
    'fractal_dimension_mean',       
    'radius_se'              ,      
    'texture_se'              ,     
    'perimeter_se'             ,    
    'area_se'                   ,   
    'smoothness_se'               , 
    'compactness_se'               ,
    'concavity_se'                 ,
    'concave points_se'            , 
    'symmetry_se'                  ,
    'fractal_dimension_se'         ,
    'radius_worst'                 ,
    'texture_worst'                ,
    'perimeter_worst'              ,
    'area_worst'                   ,
    'smoothness_worst'             ,
    'compactness_worst'            ,
    'concavity_worst'              ,
    'concave points_worst'         ,
    'symmetry_worst'               ,
    'fractal_dimension_worst'      ]
    
    
    dataset = pd.read_csv('datasets/UCI Breast Cancer Wisconsin/wdbc.data', names=col_names)
    
    Y_df = dataset['diagnosis']
    Y_all = pd.DataFrame(Y_df)

    # process and change classes from M, B to 1, -1, respectively
    Y_all[Y_df == 'M'] = 1
    Y_all[Y_df == 'B'] = -1
    y_num = pd.to_numeric(Y_all['diagnosis'], errors='coerce')

    # X: features vector , Y: labels
    Y = y_num.to_numpy()
    X = dataset.drop(columns=['id','diagnosis']).to_numpy()

    return X, Y, features_labels

def add_bias(X):
    n_samples, n_features = np.shape(X)
    return np.asarray(np.c_[X, np.ones(n_samples)])


def unscale(X,scaler):
    # only one instance
    return X*scaler.scale_ + scaler.mean_


def bisection_chance(instance, prediction_instance, prototype, SVM, FM_transform, mu=0, lam=0, p=0.5, acc = 0.00001):
    
    factor = lam*np.sqrt(2)*np.log(2*(1-p))
    ub = prototype
    lb = instance
    
    results = []
    
    while np.linalg.norm(ub-lb) > acc:
        
        bi = (ub+lb)/2
        pred = SVM.predict(FM_transform.transform(bi.reshape(1, -1)), noise=mu)
        sign = np.sign(prediction_instance*pred - factor*np.linalg.norm(FM_transform.transform(bi.reshape(1, -1)),ord=2,axis=1))
    
        if sign < 0:
            ub = bi.copy()
        else:
            lb = bi.copy()
        
        results.append(sign*np.linalg.norm((ub-lb).reshape(1, -1),axis=1))
        
    results_array = np.asarray(results).flatten()
    
    return bi, results_array


def counterfactual_explanation_linear(instance, F, b):
    # Define and solve the CVXPY problem.
    x = cp.Variable(F)
    t = cp.Variable(1)

    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [cp.SOC(t, x - np.asarray(instance).flatten()), b.T @ cp.hstack([x,1]) <= 0]

    prob = cp.Problem(cp.Minimize(t), soc_constraints)
    prob.solve()

    return x.value


def socp_opt(instance, F, b):
    # Define and solve the CVXPY problem.
    x = cp.Variable(F)
    t = cp.Variable(1)

    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [cp.SOC(t, x - np.asarray(instance).flatten()), cp.SOC(b.T @ cp.hstack([x,1]), cp.hstack([x,1]) )]

    prob = cp.Problem(cp.Minimize(t), soc_constraints)
    prob.solve()

    return x.value
            


