# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def backward_transfer(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = []
    for i in range(1, n_tasks):
        li.append(results[i - 1][i] - random_results[i])

    return np.mean(li)


def forgetting(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)

#######################################################################

def compute_fwt(R):
    """
    Forward Transfer (FWT)
    R[i, j] is test accuracy on task j after training task i.
    FWT averages all entries ABOVE the diagonal.
    """
    N = R.shape[0]
    # Upper triangular without diagonal
    upper = np.triu(R, k=1)
    # Count non-zero entries (valid FWT spots)
    count = N * (N - 1) / 2
    fwt = upper.sum() / count
    return fwt


def compute_bwt(R):
    """
    Backward Transfer (BWT)
    Measures how learning new tasks influences past tasks.
    Formula: avg over i>j of (R[i, j] - R[j, j])
    """
    N = R.shape[0]
    total = 0.0
    count = 0

    for i in range(1, N):
        for j in range(i):
            total += (R[i, j] - R[j, j])
            count += 1

    bwt = total / count
    return bwt


def compute_forgetting(R):
    """
    Forgetting (F)
    Average over tasks of:
    max accuracy achieved on task j minus final accuracy on task j.
    """
    N = R.shape[0]
    forgetting = []

    for j in range(N - 1):              # 마지막 task 제외
        best_past = np.max(R[j:, j])    # task j 이후 중 최고 성능
        final = R[N - 1, j]             # 최종 성능
        forgetting.append(best_past - final)
    return np.mean(forgetting)


def compute_rem_and_bwt_plus(BWT):
    """
    Splits BWT into:
    - REM = Remembering = 1 - |min(BWT, 0)|
    - BWT+ = positive backward transfer = max(BWT, 0)
    These follow the paper's mapping to [0,1].
    """
    rem = 1 - abs(min(BWT, 0))   # forgetting (negative BWT) mapped to [0,1]
    bwt_plus = max(BWT, 0)       # positive improvements
    return rem, bwt_plus