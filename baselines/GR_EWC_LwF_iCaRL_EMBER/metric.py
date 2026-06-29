import numpy as np

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