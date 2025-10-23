# hankel_residual_forecast.py
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ---------- Hankel helpers (same ordering) ----------
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

def hankel_vector_at(traj, Ld, t):
    cols = []
    for ell in range(Ld):
        cols.append(traj[t - ell])
    return np.concatenate(cols, axis=0)

# ---------- Model (must match training) ----------
class ResidualMLP(nn.Module):
    def __init__(self, d_in, hidden=256, depth=3, d_out=None, act='tanh', dropout=0.0):
        super().__init__()
        if d_out is None:
            d_out = d_in
        acts = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'silu': nn.SiLU(),
        }
        layers = []
        dims = [d_in] + [hidden]*(depth-1) + [d_out]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), acts.get(act, nn.Tanh())]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, u):
        return self.net(u)

# ---------- Forecast ----------
def rollout_AF(A, model, v0, mu_X, sg_X, N, device):
    """
    Roll in Hankel space: v_{k+1} = A v_k + F(v_k), F expects normalized input.
    Return preds (N,3): first 3 entries of each v_k.
    """
    d = v0.shape[0]
    v = v0.copy()
    preds = np.zeros((N, 3), dtype=float)
    with torch.no_grad():
        for k in range(N):
            preds[k] = v[:3]
            # normalize v as column (d,1)
            v_col = v.reshape(d, 1)
            v_n = (v_col - mu_X) / sg_X
            vb = torch.from_numpy(v_n.T.astype(np.float32)).to(device)  # shape (1,d)
            r = model(vb).cpu().numpy().reshape(d)                       # residual F(X)
            v = (A @ v) + r
    return preds

def plot_time_series(t, truth, preds, out_png, title="A+F forecast vs truth"):
    plt.figure(figsize=(8.0, 5.0))
    labels = ["x", "theta", "omega"]
    for i in range(3):
        plt.plot(t, truth[:, i], linewidth=1.2, label=f"{labels[i]} true")
        plt.plot(t, preds[:, i], linewidth=1.2, linestyle="--", label=f"{labels[i]} pred")
    plt.xlabel("time")
    plt.ylabel("state value")
    plt.title(title)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_3d(truth, preds, out_png):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(truth[:,0], truth[:,1], truth[:,2], linewidth=1.2, label="truth")
    ax.plot(preds[:,0], preds[:,1], preds[:,2], linewidth=1.2, linestyle="--", label="pred")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\theta$")
    ax.set_zlabel(r"$\omega$")
    ax.set_title("3D forecast: A+F vs truth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/pend3d_dataset.npz")
    ap.add_argument("--split", type=str, default="hdmd_outputs/learned_A_and_split.npz")
    ap.add_argument("--model", type=str, default="residual_model_out/residual_F.pt")
    ap.add_argument("--Ld", type=int, default=20)
    ap.add_argument("--test_idx", type=int, default=0, help="Index within test set to forecast")
    ap.add_argument("--N", type=int, default=30, help="Forecast steps")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--outdir", type=str, default="residual_forecast_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Load dataset & split/A
    D = np.load(args.data, allow_pickle=True)
    Z = D["Z"]
    T = D["T"]
    meta = json.loads(str(D["meta"]))
    dt = float(meta.get("dt", T[1]-T[0]))

    S = np.load(args.split, allow_pickle=True)
    A = S["A"]
    test_idx_all = S["test_idx_all"]

    # Load model
    ckpt = torch.load(args.model, map_location=device)
    cfg = ckpt["config"]
    mu_X = ckpt["norm_stats"]["mu_X"]
    sg_X = ckpt["norm_stats"]["sg_X"]
    d_in = cfg["d_in"]
    model = ResidualMLP(d_in=d_in, hidden=cfg["hidden"], depth=cfg["depth"], act=cfg["act"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Choose a test trajectory
    if len(test_idx_all) == 0:
        raise ValueError("Empty test set in split file.")
    test_choice = test_idx_all[min(args.test_idx, len(test_idx_all)-1)]
    traj = Z[test_choice]  # (Tlen,3)
    Tlen = traj.shape[0]
    if Tlen <= args.Ld:
        raise ValueError(f"Chosen trajectory too short (Tlen={Tlen}) for Ld={args.Ld}.")

    # Initial Hankel vector at t0 = Ld-1
    v0 = hankel_vector_at(traj, args.Ld, t=args.Ld-1)
    N = min(args.N, Tlen - args.Ld)
    preds = rollout_AF(A, model, v0, mu_X, sg_X, N, device)

    # Align truth from s = Ld-1
    s = args.Ld - 1
    truth = traj[s:s+N]
    t_axis = (s + np.arange(N)) * dt

    # RMSE metrics
    mse = np.mean((preds - truth)**2, axis=0)
    rmse = np.sqrt(mse)
    rmse_total = float(np.sqrt(np.mean((preds - truth)**2)))
    print(f"RMSE per dim [x,theta,omega] = {rmse}")
    print(f"RMSE total = {rmse_total:.6e}")

    # Plots
    tfig = os.path.join(args.outdir, "forecast_time_AF.png")
    plot_time_series(t_axis, truth, preds, tfig, title="A+F Hankel forecast vs truth")

    d3fig = os.path.join(args.outdir, "forecast_3d_AF.png")
    plot_3d(truth, preds, d3fig)

    # Save metrics
    np.savez_compressed(os.path.join(args.outdir, "metrics_forecast_AF.npz"),
                        rmse=rmse, rmse_total=rmse_total, N=N, Ld=args.Ld, test_choice=test_choice)

    print("Saved:")
    print(" -", tfig)
    print(" -", d3fig)
    print(" - metrics_forecast_AF.npz")

if __name__ == "__main__":
    main()
