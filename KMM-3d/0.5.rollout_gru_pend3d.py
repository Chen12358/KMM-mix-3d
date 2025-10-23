# rollout_gru_pend3d.py
# 读取已训练 GRU 模型，对单条轨迹做滚动预测并画图：
#  - 时域三条曲线对比（x, theta, omega）
#  - 三维相空间轨迹对比（x, theta, omega）
#
# 用法示例（按全局索引选择第 0 条轨迹，滚动 500 步）：
#   python rollout_gru_pend3d.py --data data/pend3d_dataset.npz \
#       --ckpt outputs/gru_residual.pt --traj_idx 0 --steps 500
#
# 用法示例（把 traj_idx 当作“测试集内相对索引”，需训练时保存了 splits）：
#   python rollout_gru_pend3d.py --data data/pend3d_dataset.npz \
#       --ckpt outputs/gru_residual.pt --traj_idx 2 --use_test_split --steps 500

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch
import torch.nn as nn

# --------------------------
# Data & utils
# --------------------------
def load_npz(path):
    D = np.load(path, allow_pickle=True)
    T = D["T"].astype(float)               # (L,)
    Z = D["Z"].astype(float)               # (m, L, 3)
    meta = json.loads(str(D["meta"]))
    return T, Z, meta

def wrap_angle(x):
    return (x + np.pi) % (2*np.pi) - np.pi

# --------------------------
# Model: 与训练脚本一致
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
# Plot helpers
# --------------------------
def plot_time_series(T, true_traj, pred_traj, save_path):
    names = ["x", "theta", "omega"]
    d = true_traj.shape[1]
    fig, axes = plt.subplots(d, 1, figsize=(8, 7), sharex=True)
    for j in range(d):
        axes[j].plot(T, true_traj[:, j], lw=1.2, label="true")
        axes[j].plot(T, pred_traj[:, j], lw=1.0, ls="--", label="pred")
        axes[j].set_ylabel(names[j]); axes[j].grid(alpha=0.3)
    axes[-1].set_xlabel("time")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

def plot_phase3d(true_traj, pred_traj, save_path):
    fig = plt.figure(figsize=(6.8, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(true_traj[:,0], true_traj[:,1], true_traj[:,2], lw=1.2, label="true")
    ax.plot(pred_traj[:,0], pred_traj[:,1], pred_traj[:,2], lw=1.0, ls="--", label="pred")
    ax.set_xlabel("x"); ax.set_ylabel("theta"); ax.set_zlabel("omega")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/pend3d_dataset.npz", help="path to dataset .npz")
    ap.add_argument("--ckpt", type=str, default="outputs/gru_residual.pt", help="trained checkpoint .pt")
    ap.add_argument("--traj_idx", type=int, default=300, help="trajectory index (global or TEST-relative)")
    ap.add_argument("--use_test_split", action="store_true",
                    help="interpret traj_idx as relative index inside TEST split saved in ckpt")
    ap.add_argument("--steps", type=int, default=1000, help="rollout steps (<= L-1)")
    ap.add_argument("--angle_wrap", action="store_true", help="wrap theta to (-pi, pi] for plotting only")
    ap.add_argument("--out_dir", type=str, default="outputs_rollout", help="where to save plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data + ckpt
    T, Z_all, meta = load_npz(args.data)
    dt = float(T[1] - T[0])
    m, L, d = Z_all.shape

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    hidden = int(cfg.get("hidden", 64))
    layers = int(cfg.get("layers", 1))

    # Build model and load weights
    net = GRUResidual(input_dim=d, hidden=hidden, num_layers=layers).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    # Determine which trajectory to use
    if args.use_test_split:
        splits = ckpt.get("splits", None)
        if splits is None:
            raise ValueError("Checkpoint has no 'splits'. Re-train with train_gru_pend3d.py that saves splits, or drop --use_test_split.")
        te_ids = np.array(splits["test_ids"])
        if len(te_ids) == 0:
            raise ValueError("Test split is empty.")
        ridx = int(np.clip(args.traj_idx, 0, len(te_ids)-1))
        traj_idx_global = int(te_ids[ridx])
        id_str = f"TEST_ridx{ridx}_gidx{traj_idx_global}"
    else:
        traj_idx_global = int(np.clip(args.traj_idx, 0, m-1))
        id_str = f"GIDX{traj_idx_global}"

    # Prepare rollout
    steps = min(max(1, args.steps), L-1)
    z_true_full = Z_all[traj_idx_global]   # (L, d)
    z0 = torch.from_numpy(z_true_full[0]).float().unsqueeze(0).to(device)  # (1,d)

    with torch.no_grad():
        traj_pred = net.rollout(z0, dt=dt, steps=steps)[0].cpu().numpy()   # (steps+1, d)
    traj_true = z_true_full[:steps+1]                                      # (steps+1, d)

    # Optional angle wrap for plotting
    if args.angle_wrap:
        traj_true[:,1] = wrap_angle(traj_true[:,1])
        traj_pred[:,1] = wrap_angle(traj_pred[:,1])

    # Compute simple MSE over rollout
    mse = np.mean((traj_pred - traj_true)**2)
    print(f"[Info] traj={id_str}, steps={steps}, rollout MSE={mse:.6e}")

    # Save plots
    ts_path = os.path.join(args.out_dir, f"ts_{id_str}_T{steps}.png")
    ph3d_path = os.path.join(args.out_dir, f"phase3d_{id_str}_T{steps}.png")
    plot_time_series(T[:steps+1], traj_true, traj_pred, ts_path)
    plot_phase3d(traj_true, traj_pred, ph3d_path)

    print(f"[Saved]\n  {ts_path}\n  {ph3d_path}")

if __name__ == "__main__":
    main()
