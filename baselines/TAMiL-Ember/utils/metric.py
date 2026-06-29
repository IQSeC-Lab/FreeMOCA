# TAMIL-Ember/metric.py

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


def compute_rem_and_bwt_plus(BWT):
    """
    Splits BWT into:
    - REM = Remembering = 1 − |min(BWT, 0)|
    - BWT+ = positive backward transfer = max(BWT, 0)
    These follow the paper's mapping to [0,1].
    """
    rem = 1 - abs(min(BWT, 0))   # forgetting (negative BWT) mapped to [0,1]
    bwt_plus = max(BWT, 0)       # positive improvements
    return rem, bwt_plus

def print_accuracy_matrix(R, task_names=None):

    n_tasks = R.shape[0]

    if task_names is None:
        task_names = [f'Task {i}' for i in range(n_tasks)]

    print("\n"+"="*70)
    print("Accuracy Matrix (R)")
    print("="*70)

    print(f"{'':<10}", end="")
    for name in task_names:
        print(f"{name:>7}", end="")
    print()

    for i, name in enumerate(task_names):
        print(f"{name:<10}", end="")
        for j in range(n_tasks):
            if j <= i:
                print(f"accuracy: {R[i, j]:.2f}", end=" ")
            else:
                print(f"{'-':>7}", end=" ")
        print()
    print("="*70+"\n")

def print_final_metrics(R):


    fwt = compute_fwt(R)
    bwt = compute_bwt(R)
    rem, bwt_plus = compute_rem_and_bwt_plus(bwt)

    print("\n" + "="*70)
    print(f"Final Metrics")
    print("="*70)
    print(f"Forward Transfer (FWT): {fwt:.4f}")
    print(f"Backward Transfer (BWT): {bwt:.4f}")
    print(f"Remembering (REM): {rem:.4f}")
    print(f"Positive Backward Transfer (BWT+): {bwt_plus:.4f}\n")
