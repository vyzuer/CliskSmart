import itertools
import numpy as np

from sklearn import mixture

def gmm(X, n_components_range, n_iter=100):
    
    lowest_bic = np.infty
    bic = []

    n_samples, n_dims = X.shape

    cv_types = ['spherical', 'tied', 'diag', 'full']

    best_gmm = None

    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type, n_iter=n_iter)
            # gmm = mixture.DPGMM(n_components=n_components, covariance_type=cv_type, n_iter=n_iter)
            gmm.fit(X)
            bic.append(np.abs(gmm.bic(X)))
            
            if np.abs(bic[-1]) < lowest_bic:
                lowest_bic = np.abs(bic[-1])
                best_gmm = gmm
    
    print 'No. of components: ', best_gmm.weights_.shape[0]

    return best_gmm, bic

