'''
Preprocess STRING edge lists for use in deepNF.

This script reads the six STRING edge lists in $AGAPEDATA/deepNF and exports
six adjacency matrices in `.mat` format for use by deepNF.

Code originally by Vladimir Gligorijevi, adapted from
https://github.com/VGligorijevic/deepNF.

Usage:
    python preprocessing.py
    python preprocessing.py -i $AGAPEDATA/deepNF/test -o $AGAPEDATA/deepNF/test --genes 20
'''
import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix, csc_matrix
import os
import argparse
from agape.utils import directory_exists
import glob
import pandas as pd

##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-path',  default='$AGAPEDATA/deepNF/networks',
                    type=str)
parser.add_argument('-o', '--output-path', default='$AGAPEDATA/deepNF/networks',
                    type=str)
parser.add_argument('--K',     default=3,    type=int)
parser.add_argument('--alpha', default=.98,  type=float)
parser.add_argument('--genes', default=5100, type=int)
args = parser.parse_args()


######
# io #
######

input_path = directory_exists(args.input_path)
output_path = directory_exists(args.output_path)


########
# defs #
########

def _load_network(filename, mtrx='adj'):
    print(f"Loading {filename}")
    if mtrx == 'adj':
        i, j, val = np.loadtxt(filename).T
        A = coo_matrix((val, (i, j)), shape=(args.genes, args.genes))
        A = A.todense()
        A = np.squeeze(np.asarray(A))
        if A.min() < 0:
            print("### Negative entries in the matrix are not allowed!")
            A[A < 0] = 0
            print("### Matrix converted to nonnegative matrix.")
        if (A.T == A).all():
            pass
        else:
            print("### Matrix not symmetric!")
            A = A + A.T
            print("### Matrix converted to symmetric.")
    else:
        print("### Wrong mtrx type. Possible: {'adj', 'inc'}")
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=1) == 0)
    return A


def load_networks(filenames, mtrx='adj'):
    """Load networks from edge lists.
    """
    Nets = []
    for filename in filenames:
        Nets.append(_load_network(filename, mtrx))
    return Nets


def _net_normalize(X):
    """Normalize networks according to node degrees.
    """
    if X.min() < 0:
        print("### Negative entries in the matrix are not allowed!")
        X[X < 0] = 0
        print("### Matrix converted to nonnegative matrix.")
    if (X.T == X).all():
        pass
    else:
        print("### Matrix not symmetric.")
        X = X + X.T - np.diag(np.diag(X))
        print("### Matrix converted to symmetric.")
    # normalizing the matrix
    deg = X.sum(axis=1).flatten()
    deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0
    D = np.diag(deg)
    X = D.dot(X.dot(D))
    return X


def net_normalize(Net):
    """Normalize network/networks.
    """
    if isinstance(Net, list):
        for i in range(len(Net)):
            Net[i] = _net_normalize(Net[i])
    else:
        Net = _net_normalize(Net)
    return Net


def _scaleSimMat(A):
    """Scale rows of similarity matrix.
    """
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(np.float)/col[:, None]
    return A


def RWR(A, K=args.K, alpha=args.alpha):
    """Random Walk on graph.
    """
    A = _scaleSimMat(A)
    # Random surfing
    n = A.shape[0]
    P0 = np.eye(n, dtype=float)
    P = P0.copy()
    M = np.zeros((n, n), dtype=float)
    for i in range(0, K):
        P = alpha*np.dot(P, A) + (1. - alpha)*P0
        M = M + P
    return M


def PPMI_matrix(M):
    """Compute Positive Pointwise Mutual Information Matrix.
    """
    M = _scaleSimMat(M)
    n = M.shape[0]
    col = np.asarray(M.sum(axis=0), dtype=float)
    col = col.reshape((1, n))
    row = np.asarray(M.sum(axis=1), dtype=float)
    row = row.reshape((n, 1))
    D = np.sum(col)
    np.seterr(all='ignore')
    PPMI = np.log(np.divide(D*M, np.dot(row, col)))
    PPMI[np.isnan(PPMI)] = 0
    PPMI[PPMI < 0] = 0
    return PPMI


if __name__ == "__main__":
    print(__doc__)
    print(f"Command line arguments:\n    {args}\n")

    # Load STRING networks
    string_nets = sorted(['neighborhood', 'fusion',       'cooccurence',
                          'coexpression', 'experimental', 'database'])

    filenames = sorted(glob.glob(os.path.join(input_path, "*adjacency.txt")))

    print(filenames)

    Nets = load_networks(filenames)

    # e.g. 'yeast'
    output_basename = os.path.basename(filenames[0]).split("_")[0]

    # Compute RWR + PPMI
    for i, f in zip(range(len(Nets)), filenames):
        print(f"Computing PPMI for network {f}")
        net = Nets[i]
        net = RWR(net)
        net = PPMI_matrix(net)
        net = csc_matrix(net)
        print("### Writing output to file...")
        sio.savemat(
            os.path.join(
                output_path,
                f'{output_basename}_net_{i+1}_K{args.K}_alpha{args.alpha}'),
            {"Net": net})
