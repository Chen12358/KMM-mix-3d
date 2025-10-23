# eval_kmm_hyper.py
# 1) Plot eigenvalues of A = I + A_param
# 2) Free-run rollout on a chosen trajectory and visualize time series & 3D phase space

import os, json, argparse, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------
# Utils (compatible with train script)
# --------------------------
def set_seed(seed=0):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_dataset(npz_path: str):
    D = np.load(npz_path, allow_pickle=True)
    T = D["T"]; Z = D["Z"]
    meta = json.loads(str(D["meta"]))
    dt = float(meta.get("dt", float(T[1]-T[0]) if len(T) >= 2 else 1.0))
    return T, Z, dt

def split_indices(m, train_ratio=0.7, seed=0):
    import math
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    m_train = max(1, int(math.floor(train_ratio*m)))
    train_idx = perm[:m_train]
    test_idx_all = perm[m_train:] if m_train < m else perm[m_train-1:m_train]
    return train_idx, test_idx_all, perm

# --------------------------
# Model bits (must match training)
# --------------------------
def make_mlp(in_dim, out_dim, hidden=128, depth=2, act="tanh"):
    acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    Act = acts.get(act.lower(), nn.Tanh)
    layers = []
    if depth <= 0:
        layers = [nn.Linear(in_dim, out_dim)]
    else:
        layers.append(nn.Linear(in_dim, hidden)); layers.append(Act())
        for _ in range(max(0, depth-1)):
            layers.append(nn.Linear(hidden, hidden)); layers.append(Act())
        layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)

class LowRankHyperLift(nn.Module):
    """
    g(x) = W(x) x, where W(x) = W0 + sum_{r=1}^R alpha_r(x) * a_r b_r^T
    """
    def __init__(self, d, n, R=8, gate_act="tanh"):
        super().__init__()
        self.d, self.n, self.R = d, n, R
        self.W0 = nn.Parameter(torch.zeros(n, d))
        self.a = nn.Parameter(torch.randn(R, n) / (n ** 0.5))
        self.b = nn.Parameter(torch.randn(R, d) / (d ** 0.5))
        self.u = nn.Parameter(torch.randn(R, d) / (d ** 0.5))
        self.c = nn.Parameter(torch.zeros(R))
        acts = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}
        self.act = acts.get(gate_act.lower(), nn.Tanh)()

    def forward(self, x):  # x: (B,d)
        alpha = self.act(x @ self.u.T + self.c)     # (B,R)
        base  = x @ self.W0.T                       # (B,n)
        bx    = x @ self.b.T                        # (B,R)
        ax    = alpha * bx                          # (B,R)
        return base + ax @ self.a                   # (B,n)

class KMM(nn.Module):
    def __init__(self, d=3, n=8,
                 g_rank=8, g_gate_act="tanh",
                 G_hidden=128, G_depth=2, G_act="tanh"):
        super().__init__()
        assert n >= d
        self.d, self.n = d, n
        self.g   = LowRankHyperLift(d, n, R=g_rank, gate_act=g_gate_act)
        self.Gnl = make_mlp(n, n, hidden=G_hidden, depth=G_depth, act=G_act)
        self.A_param = nn.Parameter(torch.zeros(n, n))     # value will be loaded from ckpt
        self.C = nn.Linear(n, d, bias=False)               # weights loaded from ckpt

    def get_A(self):
        I = torch.eye(self.n, device=self.A_param.device, dtype=self.A_param.dtype)
        return I + self.A_param

    def observable_step(self, x):
        z = self.g(x)
        A = self.get_A()
        z1_hat = z @ A.T + self.Gnl(z)
        return z, z1_hat

    def state_step(self, x):
        _, z1_hat = self.observable_step(x)
        return self.C(z1_hat)

    def rollout(self, x0, steps):
        B = x0.shape[0]
        preds = torch.zeros(B, steps, self.d, device=x0.device, dtype=x0.dtype)
        x = x0
        for k in range(steps):
            x = self.state_step(x)
            preds[:, k, :] = x
        return preds

# --------------------------
# Plot helpers
# --------------------------
def plot_eigs(A_eff: np.ndarray, out_png: str):
    vals = np.linalg.eigvals(A_eff)
    theta = np.linspace(0, 2*np.pi, 512)
    uc = np.exp(1j*theta)

    plt.figure(figsize=(5.2,5.2))
    plt.plot(uc.real, uc.imag, linewidth=1.0, label="unit circle")
    plt.scatter(np.real(vals), np.imag(vals), s=26, marker="o", label="eig(A)")
    lim = 1.1 * max(1.0, np.max(np.abs(vals)) + 0.05)
    plt.axhline(0, linewidth=0.5, color='k'); plt.axvline(0, linewidth=0.5, color='k')
    plt.gca().set_aspect('equal', 'box')
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.xlabel("Re(λ)"); plt.ylabel("Im(λ)")
    plt.title("Eigenvalues of A = I + A_param")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()

def plot_time_series(t, truth_dn, preds_dn, out_png):
    d = truth_dn.shape[1]
    lbls = [f"x{i+1}" for i in range(d)]
    plt.figure(figsize=(8,5))
    for i in range(d):
        plt.plot(t, truth_dn[:,i], label=f"{lbls[i]} true", linewidth=1.1)
        plt.plot(t, preds_dn[:,i], "--", label=f"{lbls[i]} pred", linewidth=1.1)
    plt.xlabel("time"); plt.ylabel("state value"); plt.title("KMM rollout")
    plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()

def plot_phase3d(truth_dn, preds_dn, out_png):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.0,5.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(truth_dn[:,0], truth_dn[:,1], truth_dn[:,2], label="truth", linewidth=1.2)
    ax.plot(preds_dn[:,0], preds_dn[:,1], preds_dn[:,2], "--", label="pred", linewidth=1.2)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title("3D phase trajectory")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_png, dpi=160); plt.close(fig)

# --------------------------
# Main eval
# --------------------------
def main(args):
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ----- Load ckpt -----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved_args = ckpt.get("args", {})
    d  = int(saved_args.get("d", args.d))
    n  = int(saved_args.get("n", args.n))
    dt = float(ckpt.get("dt", 1.0))
    mean = torch.from_numpy(ckpt.get("mean", np.zeros(d))).float()
    std  = torch.from_numpy(ckpt.get("std",  np.ones(d))).float()
    A_eff_np = ckpt.get("A_effective", None)

    device = torch.device(args.device)

    # ----- Plot eigenvalues of A -----
    if A_eff_np is None:
        # Fallback: reconstruct from model after loading
        print("A_effective not found in ckpt; will reconstruct from model after loading state_dict.")
    else:
        plot_eigs(A_eff_np, os.path.join(args.outdir, "A_eigs.png"))
        print("Saved:", os.path.join(args.outdir, "A_eigs.png"))

    # ----- Build & load model for rollout -----
    model = KMM(
        d=d, n=n,
        g_rank=int(saved_args.get("g_rank", 8)),
        g_gate_act=str(saved_args.get("g_gate_act", "tanh")),
        G_hidden=int(saved_args.get("G_hidden", 128)),
        G_depth=int(saved_args.get("G_depth", 2)),
        G_act=str(saved_args.get("G_act", "tanh")),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # If eigs not plotted yet, do it now from model.get_A()
    if A_eff_np is None:
        with torch.no_grad():
            A_eff = model.get_A().detach().cpu().numpy()
        plot_eigs(A_eff, os.path.join(args.outdir, "A_eigs.png"))
        print("Saved:", os.path.join(args.outdir, "A_eigs.png"))

    # ----- Load data & choose a trajectory for free-run -----
    T, Z, _ = load_dataset(args.data)
    m, Tlen, d_data = Z.shape
    assert d_data == d, f"Data dim {d_data} != ckpt dim {d}"

    # normalization like training (if mean/std provided)
    Z_t = (torch.from_numpy(Z).float() - mean) / (std + 1e-8)

    # pick test trajectory (using the same split rule)
    train_ratio = float(saved_args.get("train_ratio", 0.7))
    seed = int(saved_args.get("seed", 0))
    _, test_idx_all, _ = split_indices(m, train_ratio, seed)
    test_choice = int(test_idx_all[min(args.test_idx, len(test_idx_all)-1)])

    s = args.start if args.start is not None else 0
    s = int(max(0, min(Tlen-args.N-1, s)))

    x0 = Z_t[test_choice, s:s+1, :].to(device)   # (1,d)
    preds = model.rollout(x0, args.N).squeeze(0).detach().cpu().numpy()
    truth = Z_t[test_choice, s:s+args.N, :].cpu().numpy()

    # de-normalize
    truth_dn = truth * (std.numpy()) + mean.numpy()
    preds_dn = preds * (std.numpy()) + mean.numpy()
    t = np.arange(s, s+args.N) * dt

    # plots
    ts_png = os.path.join(args.outdir, "rollout_time.png")
    plot_time_series(t, truth_dn, preds_dn, ts_png)
    print("Saved:", ts_png)

    if d == 3:
        ph_png = os.path.join(args.outdir, "rollout_3d.png")
        plot_phase3d(truth_dn, preds_dn, ph_png)
        print("Saved:", ph_png)

    # also save arrays
    np.savez_compressed(os.path.join(args.outdir, "eval_outputs.npz"),
                        preds=preds_dn, truth=truth_dn, t=t)

    print("Saved:", os.path.join(args.outdir, "eval_outputs.npz"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="KMM_hyper/kmm_hyper.pt", help="path to training checkpoint")
    ap.add_argument("--data", type=str, default="data/pend3d_dataset.npz")
    ap.add_argument("--d", type=int, default=3, help="fallback if ckpt missing args")
    ap.add_argument("--n", type=int, default=16, help="fallback if ckpt missing args")
    ap.add_argument("--N", type=int, default=600, help="rollout length")
    ap.add_argument("--test_idx", type=int, default=100, help="which test trajectory")
    ap.add_argument("--start", type=int, default=500, help="start index within chosen trajectory")
    ap.add_argument("--outdir", type=str, default="KMM_hyper_eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    main(args)
