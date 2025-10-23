# train_kmm_autonomous_no_obs.py
# KMM without observable consistency loss:
# x_{t+1} ≈ dec( A g(x_t) + G(g(x_t)) )
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
    T = D["T"]
    Z = D["Z"]
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
        xs.append(traj[:-1])
        ys.append(traj[1:])
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()

# --------------------------
# Model blocks
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

class KMM(nn.Module):
    def __init__(self, d=3, n=8,
                 g_hidden=128, g_depth=2, g_act="tanh",
                 G_hidden=128, G_depth=2, G_act="tanh",
                 dec_hidden=128, dec_depth=2, dec_act="tanh",
                 A_init="identity", A_scale=0.99):
        super().__init__()
        self.d, self.n = d, n
        self.g   = make_mlp(d, n, hidden=g_hidden, depth=g_depth, act=g_act)
        self.Gnl = make_mlp(n, n, hidden=G_hidden, depth=G_depth, act=G_act)
        self.dec = make_mlp(n, d, hidden=dec_hidden, depth=dec_depth, act=dec_act)

        if A_init == "identity":
            A0 = torch.eye(n) * A_scale
        elif A_init == "random_ortho":
            M = torch.randn(n, n); q, _ = torch.linalg.qr(M); A0 = q * A_scale
        else:
            A0 = torch.randn(n, n) * 0.05
        self.A = nn.Parameter(A0)

    def observable_step(self, x):
        z = self.g(x)
        z_next_hat = z @ self.A.T + self.Gnl(z)
        return z, z_next_hat

    def state_step(self, x):
        _, z1_hat = self.observable_step(x)
        x1_hat = self.dec(z1_hat)
        return x1_hat

    def rollout(self, x0, steps):
        """Free-run in state space for 'steps' steps. Returns (B,steps,d)."""
        B = x0.shape[0]
        preds = torch.zeros(B, steps, self.d, device=x0.device, dtype=x0.dtype)
        x = x0
        for k in range(steps):
            x_next = self.state_step(x)     # 避免把 x 设成 preds 的视图
            preds[:, k, :] = x_next
            x = x_next
        return preds

# --------------------------
# Optional spectral penalty
# --------------------------
def power_spectral_norm(A, iters=5):
    with torch.no_grad():
        n = A.shape[0]
        v = torch.randn(n, device=A.device)
        v = v / (v.norm() + 1e-12)
        for _ in range(iters):
            v = (A.T @ (A @ v))
            v = v / (v.norm() + 1e-12)
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
    device = torch.device(args.device)

    train_idx, test_idx_all, _ = split_indices(m, args.train_ratio, args.seed)

    # normalization（state）
    X_all, Y_all = build_pairs(Z, train_idx)
    if args.normalize:
        mean = X_all.mean(0); std = X_all.std(0) + 1e-8
        def norm(x):   return (x - mean) / std
        def denorm(x): return x*std + mean
        X_all = norm(X_all); Y_all = norm(Y_all)
        Z_t = (torch.from_numpy(Z).float() - mean) / std
    else:
        mean = torch.zeros(d); std = torch.ones(d)
        def norm(x):   return x
        def denorm(x): return x
        Z_t = torch.from_numpy(Z).float()

    dl = DataLoader(TensorDataset(X_all, Y_all), batch_size=args.batch_size,
                    shuffle=True, drop_last=True)

    # model
    model = KMM(
        d=d, n=args.n,
        g_hidden=args.g_hidden, g_depth=args.g_depth, g_act=args.g_act,
        G_hidden=args.G_hidden, G_depth=args.G_depth, G_act=args.G_act,
        dec_hidden=args.dec_hidden, dec_depth=args.dec_depth, dec_act=args.dec_act,
        A_init=args.A_init, A_scale=args.A_scale
    ).to(device)

    # warm-start A (optional)
    if args.A_warmstart and os.path.isfile(args.A_warmstart):
        pack = np.load(args.A_warmstart, allow_pickle=True)
        A0 = pack["A"]; assert A0.shape == (args.n, args.n)
        with torch.no_grad():
            model.A.copy_(torch.from_numpy(A0).float())

    # warm-start A (optional)
    if args.A_warmstart and os.path.isfile(args.A_warmstart):
        pack = np.load(args.A_warmstart, allow_pickle=True)
        A0 = pack["A"]; assert A0.shape == (args.n, args.n)
        with torch.no_grad():
            model.A.copy_(torch.from_numpy(A0).float())

    # >>> 冻结 A（可选）
    if args.freeze_A:
        model.A.requires_grad_(False)

    # 优化器只拿 requires_grad=True 的参数
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                        lr=args.lr, weight_decay=args.wd)

    # optimizer + scheduler  ✅ 学习率调度
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)

    # train
    his_1, his_roll = [], []
    for epoch in range(1, args.epochs+1):
        model.train()
        ep1 = epR = 0.0

        # ✅ 课程式滚动：逐段增大最大滚动长度
        if args.curriculum:
            if epoch <= args.curr_ep1:
                K_cur = min(5, args.rollout_K)
            elif epoch <= args.curr_ep2:
                K_cur = min(10, args.rollout_K)
            else:
                K_cur = args.rollout_K
        else:
            K_cur = args.rollout_K

        for Xb, Yb in dl:
            Xb = Xb.to(device); Yb = Yb.to(device)

            # single-step state loss
            X1_hat = model.state_step(Xb)
            loss_1 = ((X1_hat - Yb)**2).mean()

            # rollout loss（随机 L∈[2, K_cur]）
            L = random.randint(2, K_cur)
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

            # regularization (optional)
            reg = 0.0
            if args.lambda_A_fro > 0:
                reg = reg + args.lambda_A_fro * (model.A.pow(2).sum())
            if args.lambda_A_spec > 0:
                s_max = power_spectral_norm(model.A, iters=5)
                reg = reg + args.lambda_A_spec * torch.clamp(s_max - args.spec_target, min=0.0)

            loss = args.lambda_1*loss_1 + args.lambda_roll*loss_roll + reg

            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            ep1 += loss_1.item(); epR += loss_roll.item()

        # 记录 + 调度器步进
        nb = max(1, len(dl))
        his_1.append(ep1/nb); his_roll.append(epR/nb)
        scheduler.step()

        if epoch % max(1, args.log_every) == 0:
            cur_lr = scheduler.get_last_lr()[0]
            print(f"[Epoch {epoch:03d}] state1={his_1[-1]:.6f} | roll={his_roll[-1]:.6f} "
                  f"| ||A||F={model.A.detach().norm().item():.3f} | lr={cur_lr:.2e} | K_cur={K_cur}")

    # save
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, "kmm_autonomous_no_obs.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "args": vars(args),
        "mean": mean.numpy(),
        "std": std.numpy(),
        "dt": dt
    }, ckpt_path)

    # curves
    plt.figure(figsize=(6,4))
    plt.plot(his_1,   label="single-step (state)")
    plt.plot(his_roll,label=f"rollout (<= {args.rollout_K})")
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

    # denorm & save plots
    def denorm(x): return x*std + mean
    truth_dn = denorm(truth[s:s+args.N].cpu()).numpy()
    preds_dn = denorm(preds.detach().cpu()).numpy()
    t = np.arange(s, s+args.N) * dt

    lbls = [f"x{i+1}" for i in range(d)]
    plt.figure(figsize=(8,5))
    for i in range(d):
        plt.plot(t, truth_dn[:,i], label=f"{lbls[i]} true", linewidth=1.1)
        plt.plot(t, preds_dn[:,i], "--", label=f"{lbls[i]} pred", linewidth=1.1)
    plt.xlabel("time"); plt.ylabel("state value"); plt.title("KMM (no obs-consistency) rollout")
    plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "test_rollout_time.png"), dpi=160); plt.close()

    if d == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(7.0,5.8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(truth_dn[:,0], truth_dn[:,1], truth_dn[:,2], label="truth", linewidth=1.2)
        ax.plot(preds_dn[:,0], preds_dn[:,1], preds_dn[:,2], "--", label="pred", linewidth=1.2)
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
        ax.set_title("3D rollout (KMM, no obs-consistency)")
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
    ap.add_argument("--train_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normalize", action="store_true")

    # KMM dims
    ap.add_argument("--n", type=int, default=3)

    # nets  ✅ 更平滑/容量调整（默认更稳）
    ap.add_argument("--g_hidden", type=int, default=64)
    ap.add_argument("--g_depth",  type=int, default=2)
    ap.add_argument("--g_act",    type=str, default="gelu", choices=["tanh","relu","gelu","silu"])

    ap.add_argument("--G_hidden", type=int, default=128)
    ap.add_argument("--G_depth",  type=int, default=2)
    ap.add_argument("--G_act",    type=str, default="gelu", choices=["tanh","relu","gelu","silu"])

    ap.add_argument("--dec_hidden", type=int, default=128)
    ap.add_argument("--dec_depth",  type=int, default=2)
    ap.add_argument("--dec_act",    type=str, default="gelu", choices=["tanh","relu","gelu","silu"])

    # A init / warmstart
    ap.add_argument("--A_init", type=str, default="identity", choices=["identity","random_ortho","random"])
    ap.add_argument("--A_scale", type=float, default=0.99)
    ap.add_argument("--A_warmstart", type=str, default= "", help="npz with key 'A' for warm starting") #f_edmd_outputs/learned_A_FEDMD_and_split.npz
    ap.add_argument("--freeze_A", action="store_true",
                    help="If set, use (warm-started) A as fixed and do not train it")

    # training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # ✅ 学习率调度参数
    ap.add_argument("--lr_step", type=int, default=20, help="StepLR step size (epochs)")
    ap.add_argument("--lr_gamma", type=float, default=0.5, help="StepLR decay factor")

    # losses
    ap.add_argument("--lambda_1",   type=float, default=1.0)
    ap.add_argument("--lambda_roll",type=float, default=2.0)
    ap.add_argument("--rollout_K",  type=int,   default=20)

    # ✅ 课程式滚动参数
    ap.add_argument("--curriculum", action="store_true",
                    help="enable curriculum schedule for rollout horizon")
    ap.add_argument("--curr_ep1", type=int, default=30,
                    help="epochs <= curr_ep1: use K<=5")
    ap.add_argument("--curr_ep2", type=int, default=60,
                    help="curr_ep1 < epochs <= curr_ep2: use K<=10")

    # regularizers
    ap.add_argument("--lambda_A_fro",  type=float, default=0.0)
    ap.add_argument("--lambda_A_spec", type=float, default=0.0)
    ap.add_argument("--spec_target",   type=float, default=1.0)

    # eval
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--test_idx", type=int, default=0)
    ap.add_argument("--start", type=int, default=None)

    # misc
    ap.add_argument("--outdir", type=str, default="kmm_outputs_no_obs")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--log_every", type=int, default=10)

    args = ap.parse_args()
    train(args)
