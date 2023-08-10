import cvxpy as cp
import numpy as np

class LinearSupportVectorMachine_noOffset(object):

    def __init__(self, C=1):
        self.C = C
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        
    def predict(self, X, noise=0):
        y_pred = []
        beta = np.dot(self.lagr_multipliers*self.support_vector_labels,self.support_vectors)
        for instance in X:
            predict = np.inner(np.add(beta,noise), instance)
            y_pred.append(predict.flatten())
        return np.array(y_pred)
    
    def fit(self, X, y):

        n_samples, n_features = np.shape(X)

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.inner(X[i], X[j])

        P = np.outer(y, y) * K
        q = np.ones(n_samples) * -1

        if not self.C:
            G = np.identity(n_samples) * -1
            h = np.zeros(n_samples)
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = np.vstack((G_max, G_min))
            h_max = np.zeros(n_samples)
            h_min = np.ones(n_samples) * self.C/n_samples
            h = np.hstack((h_max, h_min))

            x = cp.Variable(n_samples)
            prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),[G @ x <= h])
            prob.solve()

        lagr_mult = x.value

        # Indices for support vectors
        idx = lagr_mult > 1e-12

        self.lagr_multipliers = lagr_mult[idx]
        self.support_vectors = X[idx]
        self.support_vector_labels = y[idx]

        return self.lagr_multipliers, idx, self.support_vectors, self.support_vector_labels