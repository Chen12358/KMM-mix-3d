# train_kmm_hyper.py
# x_{t+1} ≈ C [ A g(x_t) + G(g(x_t)) ], with A = I + A_param (learnable)
import os, json, math, random, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------
# Utilities
# --------------------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_dataset(npz_path: str):
    D = np.load(npz_path, allow_pickle=True)
    T = D["T"]; Z = D["Z"]
    meta = json.loads(str(D["meta"]))
    dt = float(meta.get("dt", float(T[1]-T[0]) if len(T) >= 2 else 1.0))
    return T, Z, dt

def split_indices(m, train_ratio=0.7, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    m_train = max(1, int(math.floor(train_ratio*m)))
    train_idx = perm[:m_train]
    test_idx_all = perm[m_train:] if m_train < m else perm[m_train-1:m_train]
    return train_idx, test_idx_all, perm

def build_pairs(Z, indices):
    xs, ys = [], []
    for idx in indices:
        traj = Z[idx]
        xs.append(traj[:-1]); ys.append(traj[1:])
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()

# --------------------------
# Small MLP builder (for G)
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

# --------------------------
# Low-Rank Hypernetwork g(x)=W(x)x
# --------------------------
class LowRankHyperLift(nn.Module):
    """
    g(x) = W(x) x, where W(x) = W0 + sum_{r=1}^R alpha_r(x) * a_r b_r^T
    - a_r \in R^n, b_r \in R^d, alpha_r(x) = act(u_r^T x + c_r)
    - R << min(n,d)  controls complexity and stability
    """
    def __init__(self, d, n, R=8, gate_act="tanh"):
        super().__init__()
        self.d, self.n, self.R = d, n, R

        # Base matrix W0
        self.W0 = nn.Parameter(torch.zeros(n, d))

        # Low-rank factors a_r, b_r
        self.a = nn.Parameter(torch.randn(R, n) / (n ** 0.5))
        self.b = nn.Parameter(torch.randn(R, d) / (d ** 0.5))

        # Gates alpha_r(x) = act(u_r^T x + c_r)
        self.u = nn.Parameter(torch.randn(R, d) / (d ** 0.5))
        self.c = nn.Parameter(torch.zeros(R))

        acts = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}
        self.act = acts.get(gate_act.lower(), nn.Tanh)()

    def forward(self, x):  # x: (B,d)
        # alpha: (B,R)
        alpha = self.act(x @ self.u.T + self.c)
        # base: W0 x -> (B,n)
        base = x @ self.W0.T
        # bx: b_r^T x -> (B,R)
        bx = x @ self.b.T
        # ax: alpha_r(x) * (b_r^T x) -> (B,R)
        ax = alpha * bx
        # sum_r ax_r * a_r  -> (B,n)
        return base + ax @ self.a

# --------------------------
# KMM model
# --------------------------
class KMM(nn.Module):
    def __init__(self, d=3, n=8,
                 g_rank=8, g_gate_act="tanh",      # hypernetwork params
                 G_hidden=128, G_depth=2, G_act="tanh",
                 A_param_init="zeros", A_param_scale=0.05):
        super().__init__()
        assert n >= d, f"observable dim n={n} must be >= state dim d={d}"
        self.d, self.n = d, n

        # g: Low-rank hypernetwork W(x)x
        self.g   = LowRankHyperLift(d, n, R=g_rank, gate_act=g_gate_act)

        # residual nonlinearity G(g)
        self.Gnl = make_mlp(n, n, hidden=G_hidden, depth=G_depth, act=G_act)

        # learnable delta-A, so A = I + A_param
        if A_param_init == "zeros":
            A0 = torch.zeros(n, n)
        elif A_param_init == "randn":
            A0 = torch.randn(n, n) * A_param_scale
        elif A_param_init == "orth":
            M = torch.randn(n, n); q, _ = torch.linalg.qr(M)
            A0 = (q - torch.eye(n)) * A_param_scale
        else:
            A0 = torch.zeros(n, n)
        self.A_param = nn.Parameter(A0)

        # learnable decoder C in R^{d x n}
        self.C = nn.Linear(n, d, bias=False)

    def get_A(self):
        I = torch.eye(self.n, device=self.A_param.device, dtype=self.A_param.dtype)
        return I + self.A_param

    def observable_step(self, x):
        z = self.g(x)                         # (B,n)
        A = self.get_A()                      # (n,n)
        z_next_hat = z @ A.T + self.Gnl(z)    # (B,n)
        return z, z_next_hat

    def state_step(self, x):
        _, z1_hat = self.observable_step(x)
        return self.C(z1_hat)                 # (B,d)

    def rollout(self, x0, steps):
        B = x0.shape[0]
        preds = torch.zeros(B, steps, self.d, device=x0.device, dtype=x0.dtype)
        x = x0
        for k in range(steps):
            x_next = self.state_step(x)
            preds[:, k, :] = x_next
            x = x_next
        return preds

# --------------------------
# Optional spectral penalty
# --------------------------
def power_spectral_norm(A, iters=5):
    with torch.no_grad():
        n = A.shape[0]
        v = torch.randn(n, device=A.device); v = v / (v.norm() + 1e-12)
        for _ in range(iters):
            v = (A.T @ (A @ v)); v = v / (v.norm() + 1e-12)
        s = torch.sqrt((A @ v).pow(2).sum())
    return s

# --------------------------
# Training
# --------------------------
def train(args):
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # load & split
    _, Z, dt = load_dataset(args.data)
    m, Tlen, d = Z.shape
    assert d == args.d, f"Data dim {d} != --d {args.d}"
    assert args.n >= d, f"--n ({args.n}) must be >= --d ({d})"
    device = torch.device(args.device)

    train_idx, test_idx_all, _ = split_indices(m, args.train_ratio, args.seed)

    # normalization
    X_all, Y_all = build_pairs(Z, train_idx)
    if args.normalize:
        mean = X_all.mean(0); std = X_all.std(0) + 1e-8
        X_all = (X_all - mean) / std; Y_all = (Y_all - mean) / std
        Z_t = (torch.from_numpy(Z).float() - mean) / std
    else:
        mean = torch.zeros(d); std = torch.ones(d)
        Z_t = torch.from_numpy(Z).float()

    dl = DataLoader(TensorDataset(X_all, Y_all), batch_size=args.batch_size,
                    shuffle=True, drop_last=True)

    # model
    model = KMM(
        d=d, n=args.n,
        g_rank=args.g_rank, g_gate_act=args.g_gate_act,
        G_hidden=args.G_hidden, G_depth=args.G_depth, G_act=args.G_act,
        A_param_init=args.A_param_init, A_param_scale=args.A_param_scale
    ).to(device)

    # optimizer + scheduler
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)

    # train
    his_1, his_roll = [], []
    for epoch in range(1, args.epochs+1):
        model.train()
        ep1 = epR = 0.0

        # Progressive rollout horizon & sampling bias
        progress = (epoch - 1) / max(1, args.epochs - 1)          # [0,1]
        K_cur = int(round(args.rollout_K_min + progress * (args.rollout_K_max - args.rollout_K_min)))
        K_cur = max(2, min(K_cur, args.rollout_K_max))
        alpha = args.bias_long_min + progress * (args.bias_long_max - args.bias_long_min)

        L_values = np.arange(2, K_cur + 1, dtype=np.int32)
        weights = np.power(L_values.astype(np.float64), np.maximum(0.0, alpha))
        weights = weights / weights.sum()

        for Xb, Yb in dl:
            Xb = Xb.to(device); Yb = Yb.to(device)

            # single-step loss
            X1_hat = model.state_step(Xb)
            loss_1 = ((X1_hat - Yb)**2).mean()

            # rollout loss
            L = int(np.random.choice(L_values, p=weights))
            B = Xb.shape[0]
            x0 = torch.zeros(B, d, device=device)
            target = torch.zeros(B, L, d, device=device)
            for i in range(B):
                ti = int(np.random.choice(train_idx))
                smax = Z.shape[1] - (L + 1)
                s = 0 if smax <= 0 else int(np.random.randint(0, smax))
                target[i] = Z_t[ti, s:s+L, :].to(device)
                x0[i, :]  = Z_t[ti, s, :].to(device)

            pred = model.rollout(x0, L)
            loss_roll = ((pred - target)**2).mean()

            # regularization on effective A
            reg = 0.0
            A_eff = model.get_A()
            if args.lambda_A_fro > 0:
                reg = reg + args.lambda_A_fro * (A_eff.pow(2).sum())
            if args.lambda_A_spec > 0:
                s_max = power_spectral_norm(A_eff, iters=5)
                reg = reg + args.lambda_A_spec * torch.clamp(s_max - args.spec_target, min=0.0)

            loss = args.lambda_1*loss_1 + args.lambda_roll*loss_roll + reg

            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            ep1 += loss_1.item(); epR += loss_roll.item()

        nb = max(1, len(dl))
        his_1.append(ep1/nb); his_roll.append(epR/nb)
        scheduler.step()

        if epoch % max(1, args.log_every) == 0:
            cur_lr = scheduler.get_last_lr()[0]
            A_eff = model.get_A().detach()
            print(f"[Epoch {epoch:03d}] state1={his_1[-1]:.6f} | roll={his_roll[-1]:.6f} "
                  f"| ||A||F={A_eff.norm().item():.3f} | lr={cur_lr:.2e} | K_cur={K_cur} | alpha={alpha:.2f}")

    # save
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, "kmm_hyper.pt")
    A_effective = model.get_A().detach().cpu().numpy()
    torch.save({
        "state_dict": model.state_dict(),     # 包含 A_param 与 C
        "A_effective": A_effective,           # 直接存有效 A，方便画谱
        "args": vars(args),
        "mean": mean.numpy(),
        "std": std.numpy(),
        "dt": dt
    }, ckpt_path)

    # curves
    plt.figure(figsize=(6,4))
    plt.plot(his_1,   label="single-step (state)")
    plt.plot(his_roll,label=f"rollout (<= {args.rollout_K_max})")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "train_curves.png"), dpi=160); plt.close()

    # test free-run
    model.eval()
    test_choice = int(test_idx_all[min(args.test_idx, len(test_idx_all)-1)])
    truth = Z_t[test_choice]
    s = args.start if args.start is not None else 0
    s = int(max(0, min(Tlen-args.N-1, s)))
    x0 = truth[s:s+1, :].to(device)
    preds = model.rollout(x0, args.N).squeeze(0)

    # denorm & plots
    def denorm(x): return x*std + mean
    truth_dn = denorm(truth[s:s+args.N].cpu()).numpy()
    preds_dn = denorm(preds.detach().cpu()).numpy()
    t = np.arange(s, s+args.N) * dt

    lbls = [f"x{i+1}" for i in range(d)]
    plt.figure(figsize=(8,5))
    for i in range(d):
        plt.plot(t, truth_dn[:,i], label=f"{lbls[i]} true", linewidth=1.1)
        plt.plot(t, preds_dn[:,i], "--", label=f"{lbls[i]} pred", linewidth=1.1)
    plt.xlabel("time"); plt.ylabel("state value"); plt.title("KMM (linear decoder C) rollout")
    plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "test_rollout_time.png"), dpi=160); plt.close()

    if d == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(7.0,5.8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(truth_dn[:,0], truth_dn[:,1], truth_dn[:,2], label="truth", linewidth=1.2)
        ax.plot(preds_dn[:,0], preds_dn[:,1], preds_dn[:,2], "--", label="pred", linewidth=1.2)
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
        ax.set_title("3D rollout (KMM, linear decoder C)")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "test_rollout_3d.png"), dpi=160); plt.close(fig)

    np.savez_compressed(os.path.join(args.outdir, "kmm_predictions.npz"),
                        preds=preds_dn, truth=truth_dn, t=t)

    print("Saved:")
    print(" -", ckpt_path)
    print(" -", os.path.join(args.outdir, "train_curves.png"))
    print(" -", os.path.join(args.outdir, "test_rollout_time.png"))
    if d == 3:
        print(" -", os.path.join(args.outdir, "test_rollout_3d.png"))
    print(" -", os.path.join(args.outdir, "kmm_predictions.npz"))

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data", type=str, default="data/pend3d_dataset.npz")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--train_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normalize", action="store_true")

    # dims
    ap.add_argument("--n", type=int, default=16, help="observable dim; must be >= d")

    # g (low-rank hypernetwork) params
    ap.add_argument("--g_rank", type=int, default=8, help="low-rank R for hypernetwork g")
    ap.add_argument("--g_gate_act", type=str, default="tanh",
                    choices=["tanh","sigmoid","silu","gelu","relu"])

    # G network
    ap.add_argument("--G_hidden", type=int, default=64)
    ap.add_argument("--G_depth",  type=int, default=3)
    ap.add_argument("--G_act",    type=str, default="gelu", choices=["tanh","relu","gelu","silu"])

    # A parameterization (A = I + A_param)
    ap.add_argument("--A_param_init", type=str, default="zeros",
                    choices=["zeros","randn","orth"])
    ap.add_argument("--A_param_scale", type=float, default=0.05)

    # training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # LR schedule
    ap.add_argument("--lr_step", type=int, default=20)
    ap.add_argument("--lr_gamma", type=float, default=0.5)

    # losses & regularizers
    ap.add_argument("--lambda_1",   type=float, default=1.0)
    ap.add_argument("--lambda_roll",type=float, default=2.0)
    ap.add_argument("--lambda_A_fro",  type=float, default=0.0)
    ap.add_argument("--lambda_A_spec", type=float, default=0.0)
    ap.add_argument("--spec_target",   type=float, default=1.0)

    # Progressive rollout
    ap.add_argument("--rollout_K_min", type=int, default=5)
    ap.add_argument("--rollout_K_max", type=int, default=50)
    ap.add_argument("--bias_long_min", type=float, default=0.0)
    ap.add_argument("--bias_long_max", type=float, default=2.0)

    # eval
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--test_idx", type=int, default=0)
    ap.add_argument("--start", type=int, default=None)

    # misc
    ap.add_argument("--outdir", type=str, default="KMM_hyper")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--log_every", type=int, default=10)

    args = ap.parse_args()
    train(args)
