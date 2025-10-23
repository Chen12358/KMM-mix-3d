# train_residual_R.py
import os, json, argparse, math, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_dataset(npz_path):
    D = np.load(npz_path, allow_pickle=True)
    T = D["T"]            # (Tlen,)
    Z = D["Z"]            # (m, Tlen, 3)
    meta = json.loads(str(D["meta"]))
    dt = float(meta.get("dt", float(T[1]-T[0]) if len(T) >= 2 else 1.0))
    return T, Z, dt

def load_A(npz_path):
    pack = np.load(npz_path, allow_pickle=True)
    A = pack["A"]         # expected shape (3,3) for Ld=1
    return A

def build_pairs_from_trajectories(Z, indices):
    """Return tensors X (N,3), Y (N,3) of single-step pairs from chosen trajectories."""
    xs, ys = [], []
    for idx in indices:
        traj = Z[idx]             # (Tlen, 3)
        xs.append(traj[:-1])
        ys.append(traj[1:])
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()

def split_indices(m, train_ratio=0.1, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    m_train = max(1, int(math.floor(train_ratio*m)))
    train_idx = perm[:m_train]
    test_idx_all = perm[m_train:]
    if len(test_idx_all) == 0:
        test_idx_all = perm[m_train-1:m_train]  # ensure non-empty
    return train_idx, test_idx_all, perm

# ----------------------------
# Model: residual MLP R(z)
# ----------------------------
class ResidualMLP(nn.Module):
    def __init__(self, in_dim=3, hidden=128, depth=3, out_dim=3, act="tanh"):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "silu": nn.SiLU
        }
        Act = acts.get(act.lower(), nn.Tanh)
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(Act())
        for _ in range(max(0, depth-1)):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(Act())
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)  # R(z)

# ----------------------------
# Rollout with fixed A and learned R
# ----------------------------
def rollout_A_plus_R(A_torch, Rnet, z0, steps):
    """z0: (B,3); returns preds: (B, steps, 3)."""
    B = z0.shape[0]
    preds = torch.zeros(B, steps, 3, device=z0.device, dtype=z0.dtype)
    z = z0
    for k in range(steps):
        preds[:, k, :] = z
        z = (z @ A_torch.T) + Rnet(z)
    return preds

# ----------------------------
# Training
# ----------------------------
def train(args):
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load data and A
    _, Z, dt = load_dataset(args.data)
    A_np = load_A(args.A_path)
    assert A_np.shape == (3,3), f"A must be 3x3 for Ld=1, got {A_np.shape}"
    A_torch = torch.from_numpy(A_np).float().to(args.device)

    # 2) Split trajectories and build single-step pairs for training
    m, Tlen, three = Z.shape
    assert three == 3
    train_idx, test_idx_all, perm = split_indices(m, args.train_ratio, args.seed)
    X_all, Y_all = build_pairs_from_trajectories(Z, train_idx)

    # Optional normalization (off by default; enable if needed)
    if args.normalize:
        mean = X_all.mean(0)
        std = X_all.std(0) + 1e-8
        def norm(x): return (x - mean) / std
        def denorm(x): return x*std + mean
        X_all = norm(X_all); Y_all = norm(Y_all)
        Z_norm = (torch.from_numpy(Z).float() - mean) / std
    else:
        mean = torch.zeros(3); std = torch.ones(3)
        def norm(x): return x
        def denorm(x): return x
        Z_norm = torch.from_numpy(Z).float()

    ds = TensorDataset(X_all, Y_all)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 3) Model/opt
    Rnet = ResidualMLP(
        in_dim=3, hidden=args.hidden, depth=args.depth, out_dim=3, act=args.act
    ).to(args.device)
    opt = torch.optim.Adam(Rnet.parameters(), lr=args.lr, weight_decay=args.wd)

    # 4) Training loop
    his_single, his_roll = [], []
    for epoch in range(1, args.epochs+1):
        Rnet.train()
        ep_single_loss, ep_roll_loss = 0.0, 0.0
        for Xb, Yb in dl:
            Xb = Xb.to(args.device)
            Yb = Yb.to(args.device)

            # ---- single-step loss:  z_{t+1} ≈ A z_t + R(z_t)
            Yhat_1 = (Xb @ A_torch.T) + Rnet(Xb)
            loss_single = ((Yhat_1 - Yb)**2).mean()

            # ---- multi-step rollout loss
            # Sample random rollout length in [2, K], and random starting indices from training trajectories
            K = args.rollout_K
            roll_len = random.randint(2, K)  # at least 2
            # Construct a mini-batch of windows from training trajectories
            B = Xb.shape[0]
            z0_batch = torch.zeros(B, 3, device=args.device)
            target_seq = torch.zeros(B, roll_len, 3, device=args.device)

            # Efficient sampling: randomly pick trajectories and start times
            for i in range(B):
                ti = int(np.random.choice(train_idx))
                # ensure enough room: [s, s+roll_len]
                smax = Z.shape[1] - roll_len - 1
                if smax <= 0:
                    s = 0
                else:
                    s = int(np.random.randint(0, smax))
                truth_seq = Z_norm[ti, s:s+roll_len, :].to(args.device)   # length roll_len
                z0 = Z_norm[ti, s, :].to(args.device)
                z0_batch[i] = z0
                target_seq[i] = truth_seq

            pred_seq = rollout_A_plus_R(A_torch, Rnet, z0_batch, roll_len)  # (B, roll_len, 3)
            loss_roll = ((pred_seq - target_seq)**2).mean()

            loss = args.lambda_single * loss_single + args.lambda_roll * loss_roll

            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(Rnet.parameters(), args.grad_clip)
            opt.step()

            ep_single_loss += loss_single.item()
            ep_roll_loss   += loss_roll.item()

        n_batches = len(dl)
        ep_single_loss /= max(1, n_batches)
        ep_roll_loss   /= max(1, n_batches)
        his_single.append(ep_single_loss)
        his_roll.append(ep_roll_loss)

        if epoch % max(1, args.log_every) == 0:
            print(f"[Epoch {epoch:03d}] single={ep_single_loss:.6f} | roll={ep_roll_loss:.6f}")

    # 5) Save model and logs
    save_ckpt = os.path.join(args.outdir, "residual_R.pt")
    torch.save({
        "A": A_np,
        "model": Rnet.state_dict(),
        "args": vars(args),
        "mean": mean.numpy(),
        "std": std.numpy()
    }, save_ckpt)

    # Plot training curves
    plt.figure(figsize=(6,4))
    plt.plot(his_single, label="single-step MSE")
    plt.plot(his_roll, label=f"rollout MSE (up to {args.rollout_K})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "train_curves.png"), dpi=160)
    plt.close()

    # 6) Visualize on one test trajectory
    Rnet.eval()
    test_choice = int(test_idx_all[min(args.test_idx, len(test_idx_all)-1)])
    truth = Z_norm[test_choice].to(args.device)  # (Tlen,3)
    # pick a start index and predict N steps
    s = args.start if args.start is not None else 0
    s = int(max(0, min(truth.shape[0]-args.N-1, s)))
    z0 = truth[s:s+1, :]  # (1,3)
    preds = rollout_A_plus_R(A_torch, Rnet, z0, args.N).squeeze(0)  # (N,3)

    # denormalize if needed
    truth_dn = denorm(truth[s:s+args.N].cpu()).numpy()
    preds_dn = denorm(preds.detach().cpu()).numpy()

    # time axis
    t = np.arange(s, s+args.N) * dt

    # time plot
    plt.figure(figsize=(8,5))
    lbls = ["x","theta","omega"]
    for i in range(3):
        plt.plot(t, truth_dn[:,i], label=f"{lbls[i]} true", linewidth=1.2)
        plt.plot(t, preds_dn[:,i], "--", label=f"{lbls[i]} pred", linewidth=1.2)
    plt.xlabel("time")
    plt.ylabel("state value")
    plt.title("A + R rollout on test trajectory")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "test_rollout_time.png"), dpi=160)
    plt.close()

    # 3D plot
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(truth_dn[:,0], truth_dn[:,1], truth_dn[:,2], label="truth", linewidth=1.2)
    ax.plot(preds_dn[:,0], preds_dn[:,1], preds_dn[:,2], "--", label="pred", linewidth=1.2)
    ax.set_xlabel("x"); ax.set_ylabel(r"$\theta$"); ax.set_zlabel(r"$\omega$")
    ax.set_title("3D rollout (A + R)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "test_rollout_3d.png"), dpi=160)
    plt.close(fig)

    # Save predictions
    np.savez_compressed(
        os.path.join(args.outdir, "A_plus_R_predictions.npz"),
        preds=preds_dn, truth=truth_dn, t=t, A=A_np
    )

    print("Saved:")
    print(" -", save_ckpt)
    print(" -", os.path.join(args.outdir, "train_curves.png"))
    print(" -", os.path.join(args.outdir, "test_rollout_time.png"))
    print(" -", os.path.join(args.outdir, "test_rollout_3d.png"))
    print(" -", os.path.join(args.outdir, "A_plus_R_predictions.npz"))


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # data & A
    ap.add_argument("--data", type=str, default="data/pend3d_dataset.npz",
                    help="dataset path (npz with T, Z, meta)")
    ap.add_argument("--A_path", type=str, default="f_edmd_outputs/learned_A_FEDMD_and_split.npz",
                    help="npz file that contains A (key 'A')")
    # training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--act", type=str, default="tanh", choices=["tanh","relu","gelu","silu"])
    ap.add_argument("--lambda_single", type=float, default=1.0,
                    help="weight for single-step MSE")
    ap.add_argument("--lambda_roll", type=float, default=1.0,
                    help="weight for multi-step rollout MSE")
    ap.add_argument("--rollout_K", type=int, default=20,
                    help="max rollout steps used in rollout MSE (sampled in [2,K])")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--normalize", action="store_true",
                    help="apply z-standardization using training pairs")
    # eval/vis
    ap.add_argument("--N", type=int, default=100, help="forecast steps on test trajectory")
    ap.add_argument("--test_idx", type=int, default=0, help="which test trajectory index to visualize")
    ap.add_argument("--start", type=int, default=None, help="start index inside chosen test trajectory")
    # misc
    ap.add_argument("--train_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="fedmd_res_outputs")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--log_every", type=int, default=10,
                    help="print training logs every N epochs")

    args = ap.parse_args()
    train(args)
