# train_gru_pend3d.py
# GRU 残差式时间序列模型（可调层数/宽度，≤50步滚动MSE）
# 现在支持按“轨迹百分比”切分：训练/验证/测试
#
# 示例：
#   python train_gru_pend3d.py --data data/pend3d_dataset.npz --epochs 30 \
#     --hidden 64 --layers 1 --rollout_k 20 --bs 256 --lr 1e-3 \
#     --train_pct 0.8 --val_pct 0.1 --test_pct 0.1 \
#     --out outputs/gru_residual.pt --test_traj_rel_idx 0

import os, json, math, argparse, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------------
# Utils
# --------------------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_npz(path):
    D = np.load(path, allow_pickle=True)
    T = D["T"].astype(float)               # (L,)
    Z = D["Z"].astype(float)               # (m, L, 3)
    meta = json.loads(str(D["meta"]))
    return T, Z, meta

def to_device(obj, device):
    if isinstance(obj, (list, tuple)):
        return [to_device(o, device) for o in obj]
    return obj.to(device)

def wrap_angle(x):
    return (x + np.pi) % (2*np.pi) - np.pi

# --------------------------
# Dataset: 滑窗 (K+1) 长度，自回归训练
# --------------------------
class WindowedAutoregDataset(Dataset):
    def __init__(self, Z_all, K=20, stride=1):
        """
        Z_all: (m, L, d)
        K: rollout steps
        """
        self.Z = Z_all
        self.m, self.L, self.d = Z_all.shape
        self.K = int(K)
        self.idxs = []  # (traj_idx, start_t)
        max_start = self.L - (self.K + 1)
        for i in range(self.m):
            for t0 in range(0, max_start + 1, stride):
                self.idxs.append((i, t0))
        random.shuffle(self.idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i, t0 = self.idxs[idx]
        seq = self.Z[i, t0:t0+self.K+1, :]     # (K+1, d)
        z0  = seq[0]
        future = seq[1:]                       # (K, d)
        return {"z0": z0.astype(np.float32), "future": future.astype(np.float32)}

# --------------------------
# Model
# --------------------------
class GRUResidual(nn.Module):
    def __init__(self, input_dim=3, hidden=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, input_dim)

    def forward_step(self, z_t, h=None):
        out, h_new = self.gru(z_t, h)        # (B,1,H)
        f_hat = self.head(out[:, -1, :])     # (B,d)
        return f_hat, h_new

    @torch.no_grad()
    def rollout(self, z0, dt, steps, h0=None):
        self.eval()
        B, d = z0.shape
        traj = torch.zeros(B, steps+1, d, device=z0.device)
        traj[:,0,:] = z0
        h = h0
        z = z0
        for k in range(steps):
            f_hat, h = self.forward_step(z.unsqueeze(1), h)
            z = z + dt * f_hat
            traj[:,k+1,:] = z
        return traj

# --------------------------
# Split helpers
# --------------------------
def split_by_percentage(m, train_pct, val_pct, test_pct, seed=0):
    # 规范化：若 test_pct 未给出则作为剩余；若三者都有则要求和≈1
    if test_pct is None:
        test_pct = max(0.0, 1.0 - (train_pct + val_pct))
    total = train_pct + val_pct + test_pct
    if not (abs(total - 1.0) < 1e-6):
        # 归一化（防用户传入和不为1）
        train_pct, val_pct, test_pct = [p/total for p in (train_pct, val_pct, test_pct)]

    idxs = np.arange(m)
    rng = np.random.default_rng(seed)
    rng.shuffle(idxs)

    n_train = int(np.floor(m * train_pct))
    n_val   = int(np.floor(m * val_pct))
    n_test  = m - n_train - n_val

    tr_ids = idxs[:n_train]
    va_ids = idxs[n_train:n_train+n_val]
    te_ids = idxs[n_train+n_val:]
    return tr_ids, va_ids, te_ids

# --------------------------
# Training
# --------------------------
def train(
    data_path, out_path,
    hidden=64, layers=1,
    epochs=30, bs=256, lr=1e-3,
    rollout_k=20,  # ≤ 50
    w_one_step=1.0, w_roll=1.0,
    grad_clip=1.0, seed=0,
    train_pct=0.8, val_pct=0.1, test_pct=None,  # 百分比分割
    device=None
):
    assert rollout_k <= 50, "rollout_k should be ≤ 50"
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # load data
    T, Z_all, meta = load_npz(data_path)
    dt = float(T[1] - T[0])
    m, L, d = Z_all.shape
    print(f"[Data] m={m}, L={L}, d={d}, dt={dt}")

    # split by trajectories (percentages)
    tr_ids, va_ids, te_ids = split_by_percentage(m, train_pct, val_pct, test_pct, seed=seed)
    Z_tr, Z_val, Z_te = Z_all[tr_ids], Z_all[va_ids], Z_all[te_ids]
    print(f"[Split] train traj={len(tr_ids)}, val traj={len(va_ids)}, test traj={len(te_ids)}")

    # datasets/loaders
    train_ds = WindowedAutoregDataset(Z_tr, K=rollout_k, stride=1)
    val_ds   = WindowedAutoregDataset(Z_val, K=rollout_k, stride=1)
    test_ds  = WindowedAutoregDataset(Z_te, K=rollout_k, stride=1)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, drop_last=False)
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False, drop_last=False)

    print(f"[Windows] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # model/opt
    net = GRUResidual(input_dim=d, hidden=hidden, num_layers=layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    mse = nn.MSELoss()

    best_val = float("inf")
    history = {"train": [], "val": []}

    for ep in range(1, epochs+1):
        net.train()
        tr_loss = 0.0
        for batch in train_dl:
            z0 = batch["z0"].to(device)            # (B, d)
            gt = batch["future"].to(device)        # (B, K, d)
            B, K, d = gt.shape
            h = None
            z = z0
            loss_roll = 0.0
            loss_one = 0.0

            for k in range(K):
                f_hat, h = net.forward_step(z.unsqueeze(1), h)
                z_next_hat = z + dt * f_hat
                loss_one = loss_one + mse(z_next_hat, gt[:,k,:])
                loss_roll = loss_roll + mse(z_next_hat, gt[:,k,:])
                z = z_next_hat.detach()

            loss = w_one_step * (loss_one / K) + w_roll * (loss_roll / K)
            opt.zero_grad(); loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()
            tr_loss += loss.item()

        tr_loss /= len(train_dl)

        # eval on val
        net.eval()
        with torch.no_grad():
            va_loss = 0.0
            for batch in val_dl:
                z0 = batch["z0"].to(device)
                gt = batch["future"].to(device)
                B, K, d = gt.shape
                h = None
                z = z0
                loss_roll = 0.0
                for k in range(K):
                    f_hat, h = net.forward_step(z.unsqueeze(1), h)
                    z_next_hat = z + dt * f_hat
                    loss_roll = loss_roll + mse(z_next_hat, gt[:,k,:])
                    z = z_next_hat
                va_loss += (loss_roll / K).item()
            va_loss /= max(1, len(val_dl))

        sched.step(va_loss)
        history["train"].append(tr_loss)
        history["val"].append(va_loss)
        print(f"[Epoch {ep:03d}] train={tr_loss:.6f}  val={va_loss:.6f}")

        # save best
        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "state_dict": net.state_dict(),
                "config": dict(hidden=hidden, layers=layers, dt=dt, d=d, rollout_k=rollout_k),
                "meta": meta,
                "splits": dict(train_ids=tr_ids, val_ids=va_ids, test_ids=te_ids),
                "history": history
            }, out_path)
            print(f"  -> saved best to {out_path} (val={best_val:.6f})")

    # test eval (window MSE)
    te_loss = evaluate_roll_mse(net, test_dl, dt, mse)
    print(f"[Test] rollout_K={rollout_k}  MSE={te_loss:.6f}")

    # plot losses
    plot_loss(history, out_path.replace(".pt", "_loss.png"))
    print("Training done.")
    return out_path, dt, history

def evaluate_roll_mse(net, loader, dt, mse):
    net.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            z0 = batch["z0"].to(next(net.parameters()).device)
            gt = batch["future"].to(next(net.parameters()).device)
            B, K, d = gt.shape
            h = None
            z = z0
            loss_roll = 0.0
            for k in range(K):
                f_hat, h = net.forward_step(z.unsqueeze(1), h)
                z_next_hat = z + dt * f_hat
                loss_roll = loss_roll + mse(z_next_hat, gt[:,k,:])
                z = z_next_hat
            total += (loss_roll / K).item()
    return total / max(1, len(loader))

# --------------------------
# Plotting & Testing
# --------------------------
def plot_loss(history, save_path):
    plt.figure()
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()

def test_and_plot(data_path, ckpt_path, test_traj_rel_idx=0, rollout_steps=300, angle_wrap=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T, Z_all, meta = load_npz(data_path)
    dt = float(T[1]-T[0])
    m, L, d = Z_all.shape

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    splits = ckpt.get("splits", None)
    if splits is None:
        raise ValueError("Checkpoint missing 'splits'. Please re-train with this script version.")

    te_ids = np.array(splits["test_ids"])
    if len(te_ids) == 0:
        raise ValueError("Test set is empty based on provided percentages.")

    # 相对测试集索引 -> 原始轨迹索引
    ridx = int(np.clip(test_traj_rel_idx, 0, len(te_ids)-1))
    i = int(te_ids[ridx])

    net = GRUResidual(input_dim=d, hidden=cfg["hidden"], num_layers=cfg["layers"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    # 这条测试轨迹做滚动预测
    z_true = Z_all[i]                       # (L, d)
    steps = min(rollout_steps, L-1)
    z0 = torch.from_numpy(z_true[0]).float().unsqueeze(0).to(device)  # (1,d)

    with torch.no_grad():
        traj_hat = net.rollout(z0, dt=dt, steps=steps)[0].cpu().numpy()  # (steps+1, d)
    traj_true = z_true[:steps+1]

    if angle_wrap:
        traj_true[:,1] = wrap_angle(traj_true[:,1])
        traj_hat[:,1]  = wrap_angle(traj_hat[:,1])

    # 时间序列图
    t = T[:steps+1]
    names = ["x", "theta", "omega"]
    fig, axes = plt.subplots(3,1, figsize=(8,7), sharex=True)
    for j in range(d):
        axes[j].plot(t, traj_true[:,j], lw=1.2, label="true")
        axes[j].plot(t, traj_hat[:,j], lw=1.0, ls="--", label="pred")
        axes[j].set_ylabel(names[j]); axes[j].grid(alpha=0.3)
    axes[-1].set_xlabel("time")
    axes[0].legend()
    ts_path = ckpt_path.replace(".pt", f"_traj_TEST_ridx{ridx}_T{steps}.png")
    fig.tight_layout(); fig.savefig(ts_path, dpi=160); plt.close(fig)

    # 相图
    fig = plt.figure(figsize=(5.5,4.8))
    plt.plot(traj_true[:,1], traj_true[:,2], lw=1.2, label="true")
    plt.plot(traj_hat[:,1],  traj_hat[:,2],  lw=1.0, ls="--", label="pred")
    plt.xlabel("theta"); plt.ylabel("omega"); plt.legend()
    plt.grid(alpha=0.3); plt.tight_layout()
    ph_path = ckpt_path.replace(".pt", f"_phase_TEST_ridx{ridx}_T{steps}.png")
    plt.savefig(ph_path, dpi=160); plt.close(fig)

    print(f"[Test Plot] saved:\n  {ts_path}\n  {ph_path}")
    print(f"[Test Info] used TEST trajectory rel_idx={ridx}, global_idx={i}")

# --------------------------
# Main
# --------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/pend3d_dataset.npz", help="path to dataset .npz")
    p.add_argument("--out",  type=str, default="outputs/gru_residual.pt", help="ckpt output path")
    p.add_argument("--hidden", type=int, default=64, help="GRU hidden width")
    p.add_argument("--layers", type=int, default=1, help="GRU num layers")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rollout_k", type=int, default=20, help="rollout steps <= 50")
    p.add_argument("--w_one_step", type=float, default=1.0)
    p.add_argument("--w_roll", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    # 百分比分割
    p.add_argument("--train_pct", type=float, default=0.1, help="fraction of trajectories for TRAIN")
    p.add_argument("--val_pct",   type=float, default=0.1, help="fraction of trajectories for VAL")
    p.add_argument("--test_pct",  type=float, default=0.1, help="fraction of trajectories for TEST; if None, uses 1-train_pct-val_pct")

    # 测试绘图
    p.add_argument("--test_traj_rel_idx", type=int, default=0, help="index within TEST split")
    p.add_argument("--test_steps", type=int, default=300)
    p.add_argument("--angle_wrap", action="store_true", help="wrap theta for plotting only")
    args = p.parse_args()

    ckpt_path, dt, hist = train(
        data_path=args.data, out_path=args.out,
        hidden=args.hidden, layers=args.layers,
        epochs=args.epochs, bs=args.bs, lr=args.lr,
        rollout_k=args.rollout_k, w_one_step=args.w_one_step, w_roll=args.w_roll,
        grad_clip=args.grad_clip, seed=args.seed,
        train_pct=args.train_pct, val_pct=args.val_pct, test_pct=args.test_pct
    )

    test_and_plot(
        data_path=args.data, ckpt_path=ckpt_path,
        test_traj_rel_idx=args.test_traj_rel_idx,
        rollout_steps=args.test_steps,
        angle_wrap=args.angle_wrap
    )

if __name__ == "__main__":
    main()
