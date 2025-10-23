# hankel_dmd_eval.py (F-EDMD version)
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def build_hankel_columns(traj, Ld):
    T = traj.shape[0]
    D = 3 * Ld
    K = T - Ld
    if K <= 0:
        raise ValueError(f"Trajectory too short for Ld={Ld}. Need T >= Ld+1.")
    X_cols = np.zeros((D, K), dtype=float)
    Y_cols = np.zeros((D, K), dtype=float)

    def hankel_at(t):
        cols = []
        for ell in range(Ld):
            cols.append(traj[t - ell])
        return np.concatenate(cols, axis=0)

    for j, t in enumerate(range(Ld-1, T-1)):
        X_cols[:, j] = hankel_at(t)
        Y_cols[:, j] = hankel_at(t+1)
    return X_cols, Y_cols

def assemble_training_matrices(Z, train_indices, Ld):
    X_list, Y_list = [], []
    for idx in train_indices:
        Xi, Yi = build_hankel_columns(Z[idx], Ld)
        X_list.append(Xi); Y_list.append(Yi)
    X = np.concatenate(X_list, axis=1)
    Y = np.concatenate(Y_list, axis=1)
    return X, Y

# ---------- F-EDMD utilities ----------
def svd_pinv(A, rcond=None):
    """
    Compute Moore–Penrose pseudoinverse of A using SVD explicitly.
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    if rcond is None:
        eps = np.finfo(s.dtype if np.issubdtype(s.dtype, np.floating) else np.float64).eps
        rcond = max(A.shape) * eps
    cutoff = rcond * (s[0] if s.size else 0.0)
    s_inv = np.zeros_like(s)
    nonzero = s > cutoff
    s_inv[nonzero] = 1.0 / s[nonzero]
    return (Vh.conj().T) @ np.diag(s_inv) @ (U.conj().T)

def FEDMD(Y, X):
    """
    Compute P * Y * X^dagger where:
      - Y, X are n x m (same shape)
      - P is the Hermitian projector in C^{n x n} onto R(Y^*) ∩ R(X^*)
      - X^dagger is the Moore-Penrose pseudoinverse of X computed via SVD
    Returns: (n x n) matrix P @ Y @ X_pinv
    """
    if Y.shape != X.shape:
        raise ValueError("Y and X must have the same shape (n x m).")
    n, m = Y.shape

    # Step 1: Build augmented matrix C = [Y; X] in R^{2n x m}
    C = np.vstack([Y, X])

    # Step 2: SVD of C to get left singular vectors U
    U, s, Vh = np.linalg.svd(C, full_matrices=True)

    # Numerical rank of C
    r = np.linalg.matrix_rank(C)

    # Step 3: Take the "null" block from the top half (size n x (2n - r))
    U12 = U[:n, r:]  # (n x (2n - r))

    # Step 4: Orthonormal basis Q for col(U12) via reduced QR
    if U12.size == 0:
        Q = np.zeros((n, 0), dtype=Y.dtype)
    else:
        Q, _ = np.linalg.qr(U12, mode='reduced')

    # Step 5: Hermitian projector onto R(Y^*) ∩ R(X^*)
    P = Q @ (Q.conj().T)  # (n x n)

    # Step 6: Compute X^\dagger via SVD
    X_pinv = svd_pinv(X)

    # Step 7: Return P Y X^\dagger
    return P @ Y @ X_pinv

def compute_A(Y, X, **kwargs):
    """Compute A via F-EDMD: A = P @ Y @ X^+ (kwargs ignored, for interface compatibility)."""
    return FEDMD(Y, X)
# --------------------------------------

def plot_eigs_clean(A, a_param, dt, outpath, zero_tol=1e-10):
    """Plot eigenvalues of A excluding ~zero ones; do NOT draw unit circle."""
    vals = np.linalg.eigvals(A)
    # mask near-zero eigenvalues
    mask = np.abs(vals) > zero_tol
    vals_nz = vals[mask]
    lam_true = np.exp(a_param * dt)

    plt.figure(figsize=(6.2, 5.8))
    plt.scatter([np.real(lam_true)], [np.imag(lam_true)], marker="*", s=140, label=r"$\exp(a\,\Delta t)$")
    plt.scatter(np.real(vals_nz), np.imag(vals_nz), s=14, label="eig(A) (nonzero)")
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("Eigenvalues of A (zeros removed) vs discrete eigenvalue of x-subsystem")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def forecast_with_A(A, traj, Ld, N):
    def hankel_at(t):
        cols = []
        for ell in range(Ld):
            cols.append(traj[t - ell])
        return np.concatenate(cols, axis=0)

    v = hankel_at(Ld-1)
    preds = np.zeros((N, 3), dtype=float)
    for n in range(N):
        preds[n] = v[:3]
        v = (A @ v)
    return preds

def plot_forecast_time(truth, preds, dt, outpath, start_index, a_param=None):
    N = preds.shape[0]
    t = (start_index + np.arange(N)) * dt

    plt.figure(figsize=(8,5))
    plt.plot(t, truth[:,0], linewidth=1.3, label="x true")
    plt.plot(t, preds[:,0], linewidth=1.3, linestyle="--", label="x pred")
    plt.plot(t, truth[:,1], linewidth=1.0, label="theta true")
    plt.plot(t, preds[:,1], linewidth=1.0, linestyle="--", label="theta pred")
    plt.plot(t, truth[:,2], linewidth=1.0, label="omega true")
    plt.plot(t, preds[:,2], linewidth=1.0, linestyle="--", label="omega pred")
    plt.xlabel("time")
    plt.ylabel("state value")
    ttl = "Forecast on a test trajectory (Hankel-DMD, F-EDMD A)"
    if a_param is not None:
        ttl += f"  a={a_param:.3g}"
    plt.title(ttl)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_forecast_3d(truth, preds, outpath):
    """3D plot of trajectories in (x, theta, omega) space (truth vs pred)."""
    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(truth[:,0], truth[:,1], truth[:,2], linewidth=1.2, label="truth")
    ax.plot(preds[:,0], preds[:,1], preds[:,2], linewidth=1.2, linestyle="--", label="pred")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\theta$")
    ax.set_zlabel(r"$\omega$")
    ax.set_title("3D forecast trajectory (truth vs pred)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default='data/pend3d_dataset.npz',  help="Path to dataset .npz (from make_dataset.py)")
    ap.add_argument("--Ld", type=int, default=1, help="Number of delays (time-delay length)")
    ap.add_argument("--train_ratio", type=float, default=0.1, help="Fraction of trajectories for training")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for splitting")
    ap.add_argument("--test_idx", type=int, default=0, help="Index (within test set) of trajectory to forecast")
    ap.add_argument("--N", type=int, default=400, help="Forecast steps")
    ap.add_argument("--outdir", type=str, default="f_edmd_outputs", help="Directory to save outputs")
    ap.add_argument("--zero_tol", type=float, default=1e-10, help="Tolerance to filter zero eigenvalues")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    D = np.load(args.data, allow_pickle=True)
    T = D["T"]
    Z = D["Z"]
    IC = D["IC"]
    meta = json.loads(str(D["meta"]))
    dt = float(meta.get("dt", T[1]-T[0]))
    a_param = float(meta.get("a_param", -0.2))

    m, Tlen, _ = Z.shape
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(m)
    m_train = int(np.floor(args.train_ratio * m))
    train_idx = perm[:m_train]
    test_idx_all = perm[m_train:]
    if len(test_idx_all) == 0:
        raise ValueError("No test trajectories (train_ratio too high).")
    test_choice = test_idx_all[min(args.test_idx, len(test_idx_all)-1)]

    # Assemble and solve
    X, Y = assemble_training_matrices(Z, train_idx, args.Ld)
    A = compute_A(Y, X)  # F-EDMD: A = P Y X^+

    # (1) Output rank of A
    rankA = np.linalg.matrix_rank(A)
    print("A shape:", A.shape, "  rank(A) =", rankA)

    # (2) Clean eigen plot (zero eigenvalues removed, no unit circle)
    eigfig = os.path.join(args.outdir, "eigvals_vs_true_x_clean.png")
    plot_eigs_clean(A, a_param, dt, eigfig, zero_tol=args.zero_tol)

    # (3) Forecast on one test trajectory & plots
    traj = Z[test_choice]
    N = min(args.N, Tlen - args.Ld) if Tlen > args.Ld else max(1, args.N)
    preds = forecast_with_A(A, traj, args.Ld, N=N)
    s = args.Ld - 1
    truth = traj[s:s+N]

    fcfig_time = os.path.join(args.outdir, "forecast_test_traj.png")
    plot_forecast_time(truth, preds, dt, fcfig_time, start_index=s, a_param=a_param)

    fcfig_3d = os.path.join(args.outdir, "forecast_test_traj_3d.png")
    plot_forecast_3d(truth, preds, fcfig_3d)

    # Save artifacts (filename updated for F-EDMD)
    np.savez_compressed(
        os.path.join(args.outdir, "learned_A_FEDMD_and_split.npz"),
        A=A, train_idx=train_idx, test_idx_all=test_idx_all, perm=perm, Ld=args.Ld, rankA=rankA
    )

    print("Saved:")
    print(" -", eigfig)
    print(" -", fcfig_time)
    print(" -", fcfig_3d)
    print(" -", os.path.join(args.outdir, "learned_A_FEDMD_and_split.npz"))

if __name__ == "__main__":
    main()
