# AUC metric
import numpy as np

def auc_metric_(solution, prediction):
    ''' Normarlized Area under ROC curve (AUC).
    Return Gini index = 2*AUC-1 for  binary classification problems.'''
    r_ = tiedrank(prediction)
    s_ = solution.ravel()
    if sum(s_) == 0: print('WARNING: no positive class examples')
    npos = sum(s_ == 1)
    nneg = sum(s_ < 1)
    auc = (sum(r_[s_ == 1]) - npos * (npos + 1) / 2) / (nneg * npos)
    return 2 * auc - 1
    
    
def tiedrank(a):
    ''' Return the ranks (with base 1) of a list resolving ties by averaging.
     This works for numpy arrays.'''
    m = len(a)
    # Sort a in ascending order (sa=sorted vals, i=indices)
    i = a.argsort()
    sa = a[i]
    # Find unique values
    uval = np.unique(a)
    # Test whether there are ties 
    R = np.arange(m, dtype=float) + 1  # Ranks with base 1
    if len(uval) != m:
        # Average the ranks for the ties 
        oldval = sa[0]
        newval = sa[0]
        k0 = 0
        for k in range(1, m):
            newval = sa[k]
            if newval == oldval:
                # moving average
                R[k0:k + 1] = R[k - 1] * (k - k0) / (k - k0 + 1) + R[k] / (k - k0 + 1)
            else:
                k0 = k;
                oldval = newval
    # Invert the index
    S = np.empty(m)
    S[i] = R
    return S