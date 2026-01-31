import numpy as np

def compute_cl_metrics(R):
    """
    R: ndarray [T, T] with accuracies in [0,1] (NOT percent).
       R[i,j] = accuracy on task j after training task i.
    Returns: dict with FWT, BWT, BWT+, F, REM, ACC
    """
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    assert R.shape[1] == T, "R must be square [T,T]"

    # ---- FWT: mean of upper triangle (i<j) ----
    upper = [R[i, j] for i in range(T) for j in range(T) if i < j]
    FWT = float(np.mean(upper)) if upper else 0.0

    # ---- BWT: mean of (R[i,j] - R[j,j]) over lower triangle (i>j) ----
    lower_terms = [R[i, j] - R[j, j] for i in range(1, T) for j in range(0, i)]
    BWT = float(np.mean(lower_terms)) if lower_terms else 0.0

    # ---- BWT+ and Forgetting F (negative part) ----
    BWT_plus = max(BWT, 0.0)
    F = abs(min(BWT, 0.0))          # == max(-BWT, 0.0)

    # ---- REM: 1 - F ----
    REM = 1.0 - F

    # ---- Final average accuracy (optional but useful) ----
    ACC = float(np.mean(R[-1, :])) if T > 0 else 0.0

    # Safety clamp (optional; mathematically F<=1 if acc in [0,1])
    REM = float(np.clip(REM, 0.0, 1.0))

    return {
        "FWT": FWT,
        "BWT": BWT,
        "BWT+": BWT_plus,
        "F": F,
        "REM": REM,
        "ACC": ACC,
    }
