# hankel_two_stage_forecast.py
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def hankel_vector_at(traj, Ld, t):
    cols = []
    for ell in range(Ld):
        cols.append(traj[t - ell])
    return np.concatenate(cols, axis=0)

class MLP(nn.Module):
    def __init__(self, d_in, hidden=256, depth=3, d_out=None, act='tanh', dropout=0.0):
        super().__init__()
        if d_out is None:
            d_out = d_in
        acts = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'gelu': nn.GELU(), 'elu': nn.ELU(), 'silu': nn.SiLU()}
        layers = []
        dims = [d_in] + [hidden]*(depth-1) + [d_out]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), acts.get(act, nn.Tanh())]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def rollout_two_stage(A, P, G, v0, mu_X, sg_X, mu_UA, sg_UA, N, device):
    d = v0.shape[0]
    v = v0.copy()
    preds = np.zeros((N, 3), dtype=float)
    A_t = torch.from_numpy(A).float().to(device)
    muX_t = torch.from_numpy(mu_X.T).float().to(device)
    sgX_t = torch.from_numpy(sg_X.T).float().to(device)
    muUA_t = torch.from_numpy(mu_UA.T).float().to(device)
    sgUA_t = torch.from_numpy(sg_UA.T).float().to(device)
    with torch.no_grad():
        for k in range(N):
            preds[k] = v[:3]
            v_col = v.reshape(1, d).astype(np.float32)
            x_b = torch.from_numpy(v_col).to(device)
            x_n = (x_b - muX_t) / sgX_t
            GX = G(x_n)
            Av = x_b @ A_t.T
            U = Av + GX
            U_n = (U - muUA_t) / sgUA_t
            v_next = P(U_n).cpu().numpy().reshape(d)
            v = v_next
    return preds

def plot_time_series(t, truth, preds, out_png, title="Two-stage (A+G+P) forecast vs truth"):
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
    ax.set_title("3D forecast: Two-stage (A+G+P) vs truth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/pend3d_dataset.npz")
    ap.add_argument("--split", type=str, default="hdmd_outputs/learned_A_and_split.npz")
    ap.add_argument("--ckpt", type=str, default="two_stage_out/two_stage_PG.pt")
    ap.add_argument("--Ld", type=int, default=20)
    ap.add_argument("--test_idx", type=int, default=3)
    ap.add_argument("--N", type=int, default=900)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--outdir", type=str, default="two_stage_forecast_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    D = np.load(args.data, allow_pickle=True)
    Z = D["Z"]; T = D["T"]
    meta = json.loads(str(D["meta"])); dt = float(meta.get("dt", T[1]-T[0]))

    S = np.load(args.split, allow_pickle=True)
    A = S["A"]; test_idx_all = S["test_idx_all"]

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    mu_X = ckpt["norm_stats"]["mu_X"]; sg_X = ckpt["norm_stats"]["sg_X"]
    mu_UA = ckpt["norm_stats"]["mu_UA"]; sg_UA = ckpt["norm_stats"]["sg_UA"]

    d = int(mu_X.shape[0])
    P = MLP(d_in=d, hidden=cfg["hidden_P"], depth=cfg["depth_P"], act=cfg["act"], dropout=cfg["dropout"]).to(device)
    G = MLP(d_in=d, hidden=cfg["hidden_G"], depth=cfg["depth_G"], act=cfg["act"], dropout=cfg["dropout"]).to(device)
    P.load_state_dict(ckpt["P_state_dict"]); G.load_state_dict(ckpt["G_state_dict"])
    P.eval(); G.eval()

    if len(test_idx_all) == 0: raise ValueError("Empty test set in split file.")
    test_choice = test_idx_all[min(args.test_idx, len(test_idx_all)-1)]
    traj = Z[test_choice]; Tlen = traj.shape[0]
    if Tlen <= args.Ld: raise ValueError(f"Chosen trajectory too short (Tlen={Tlen}) for Ld={args.Ld}.")

    v0 = hankel_vector_at(traj, args.Ld, t=args.Ld-1)
    N = min(args.N, Tlen - args.Ld)
    preds = rollout_two_stage(A, P, G, v0, mu_X, sg_X, mu_UA, sg_UA, N, device)

    s = args.Ld - 1
    truth = traj[s:s+N]; t_axis = (s + np.arange(N)) * dt

    mse = np.mean((preds - truth)**2, axis=0); rmse = np.sqrt(mse); rmse_total = float(np.sqrt(np.mean((preds - truth)**2)))
    print(f"RMSE per dim [x,theta,omega] = {rmse}")
    print(f"RMSE total = {rmse_total:.6e}")

    tfig = os.path.join(args.outdir, "forecast_time_two_stage.png"); plot_time_series(t_axis, truth, preds, tfig)
    d3fig = os.path.join(args.outdir, "forecast_3d_two_stage.png"); plot_3d(truth, preds, d3fig)

    np.savez_compressed(os.path.join(args.outdir, "metrics_two_stage_forecast.npz"), rmse=rmse, rmse_total=rmse_total, N=N, Ld=args.Ld, test_choice=test_choice)
    print("Saved:"); print(" -", tfig); print(" -", d3fig); print(" - metrics_two_stage_forecast.npz")

if __name__ == "__main__":
    main()
