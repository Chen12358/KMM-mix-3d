# eval_kmm_IplusA_rollout.py
# 1) 从 ckpt 取出 A = I + A_param，计算并绘制其特征值
# 2) 用保存的模型在数据上做滚动预测，画时域+相空间
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ---------- tiny modules (与训练时一致的最小定义) ----------
def make_mlp(in_dim, out_dim, hidden=128, depth=2, act="gelu"):
    acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    Act = acts.get(act.lower(), nn.GELU)
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
    """与训练脚本一致的结构：A = I + A_param，decoder = slice 前 d 维"""
    def __init__(self, d=3, n=8,
                 g_hidden=128, g_depth=3, g_act="gelu",
                 G_hidden=128, G_depth=3, G_act="gelu"):
        super().__init__()
        assert n >= d
        self.d, self.n = d, n
        self.g   = make_mlp(d, n, hidden=g_hidden, depth=g_depth, act=g_act)
        self.Gnl = make_mlp(n, n, hidden=G_hidden, depth=G_depth, act=G_act)
        # A_param 的实际权重会从 state_dict 里加载
        self.A_param = nn.Parameter(torch.zeros(n, n))

    def get_A(self):
        I = torch.eye(self.n, device=self.A_param.device, dtype=self.A_param.dtype)
        return I + self.A_param

    def observable_step(self, x):
        z = self.g(x)
        A = self.get_A()
        z1 = z @ A.T + self.Gnl(z)
        return z, z1

    def state_step(self, x):
        _, z1 = self.observable_step(x)
        return z1[:, :self.d]

    def rollout(self, x0, steps):
        B, d = x0.shape[0], self.d
        preds = torch.zeros(B, steps, d, device=x0.device, dtype=x0.dtype)
        x = x0
        for k in range(steps):
            x = self.state_step(x)
            preds[:, k, :] = x
        return preds

# ---------- plotting helpers ----------
def plot_eigs(A, out_png, title_prefix="Eigenvalues of (I+A_param)"):
    vals = np.linalg.eigvals(A)
    rankA = np.linalg.matrix_rank(A)
    plt.figure(figsize=(6.2,5.8))
    plt.scatter(np.real(vals), np.imag(vals), s=22, label="eig(I+A)")
    plt.axhline(0, linewidth=0.8); plt.axvline(0, linewidth=0.8)
    plt.xlabel("Re(λ)"); plt.ylabel("Im(λ)")
    plt.title(f"{title_prefix}  (rank={rankA}, n={A.shape[0]})")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()
    print(f"[INFO] A shape={A.shape}, rank={rankA}  -> eig plot saved to {out_png}")

def plot_time_series(t, truth, preds, out_png, title="KMM rollout (time domain)"):
    d = truth.shape[1]
    lbls = [f"x{i+1}" for i in range(d)]
    plt.figure(figsize=(8,5))
    for i in range(d):
        plt.plot(t, truth[:,i], label=f"{lbls[i]} true", linewidth=1.1)
        plt.plot(t, preds[:,i], "--", label=f"{lbls[i]} pred", linewidth=1.1)
    plt.xlabel("time"); plt.ylabel("state value"); plt.title(title)
    plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()
    print(f"[INFO] time-series plot saved to {out_png}")

def plot_phase3d(truth, preds, out_png, title="KMM rollout (phase space)"):
    assert truth.shape[1] == 3, "3D 相空间图仅支持 d=3"
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.0,5.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(truth[:,0], truth[:,1], truth[:,2], linewidth=1.2, label="truth")
    ax.plot(preds[:,0], preds[:,1], preds[:,2], "--", linewidth=1.2, label="pred")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    print(f"[INFO] 3D phase plot saved to {out_png}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="kmm_outputs_1/kmm_autonomous_no_obs_slice_dec.pt",
                    help="path to trained checkpoint")
    ap.add_argument("--data", type=str, default="data/pend3d_dataset.npz",
                    help="dataset npz used in training (for trajectories & dt)")
    ap.add_argument("--outdir", type=str, default="kmm_eval_outputs",
                    help="directory to save figures")
    ap.add_argument("--test_idx", type=int, default=10, help="which trajectory to evaluate")
    ap.add_argument("--start", type=int, default=500, help="start index on that trajectory")
    ap.add_argument("--N", type=int, default=500, help="rollout steps")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- load ckpt & data ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    args_saved = ckpt.get("args", {})
    d = int(args_saved.get("d", 3))
    n = int(args_saved.get("n", 8))
    dt = ckpt.get("dt", 1.0)

    mean = torch.from_numpy(ckpt.get("mean", np.zeros(d))).float()
    std  = torch.from_numpy(ckpt.get("std",  np.ones(d))).float()

    D = np.load(args.data, allow_pickle=True)
    T = D["T"]; Z = D["Z"]                 # (m, Tlen, d)
    if "meta" in D:
        meta = json.loads(str(D["meta"]))
        dt = float(meta.get("dt", float(T[1]-T[0]) if len(T)>=2 else dt))

    m, Tlen, d_data = Z.shape
    assert d_data == d, f"dataset d={d_data} != ckpt d={d}"

    device = torch.device(args.device)

    # ---- 1) eigenvalues of I + A_param ----
    if "A_effective" in ckpt:
        A_eff = ckpt["A_effective"]
    else:
        # 从 state_dict 组装
        sd = ckpt["state_dict"]
        # 兼容可能的前缀（例如 'module.'）
        a_keys = [k for k in sd.keys() if k.endswith("A_param")]
        if not a_keys:
            raise KeyError("Cannot find 'A_param' in state_dict.")
        A_param = sd[a_keys[0]].detach().cpu().numpy()
        A_eff = np.eye(A_param.shape[0]) + A_param

    eig_png = os.path.join(args.outdir, "A_eigs.png")
    plot_eigs(A_eff, eig_png, title_prefix="Eigenvalues of (I + A_param)")

    # ---- 2) rollout on one trajectory ----
    # 构建与训练一致的模型结构（超参从 ckpt 读取）
    model = KMM(
        d=d, n=n,
        g_hidden=int(args_saved.get("g_hidden", 128)),
        g_depth =int(args_saved.get("g_depth", 3)),
        g_act   =args_saved.get("g_act", "gelu"),
        G_hidden=int(args_saved.get("G_hidden", 128)),
        G_depth =int(args_saved.get("G_depth", 3)),
        G_act   =args_saved.get("G_act", "gelu"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    # 取一条轨迹，做同样的标准化
    Z_t = torch.from_numpy(Z).float()
    if (std > 0).all():
        Z_n = (Z_t - mean) / std
    else:
        Z_n = Z_t.clone()

    test_idx = max(0, min(m-1, args.test_idx))
    traj_n = Z_n[test_idx]                  # (Tlen, d)
    s = max(0, min(Tlen-args.N-1, args.start))
    x0 = traj_n[s:s+1, :].to(device)
    with torch.no_grad():
        preds_n = model.rollout(x0, args.N).squeeze(0).cpu().numpy()  # (N,d)

    # 反标准化
    truth = Z[s:s+args.N, test_idx, :] if Z.shape == (T.shape[0],) else None  # just avoid confusion
    truth = Z[test_idx, s:s+args.N, :]                # (N,d)
    preds = preds_n * std.numpy() + mean.numpy()

    t = np.arange(s, s+args.N) * dt

    # 画图
    ts_png = os.path.join(args.outdir, "rollout_time.png")
    plot_time_series(t, truth, preds, ts_png, title="KMM rollout (time)")

    if d == 3:
        phase_png = os.path.join(args.outdir, "rollout_phase3d.png")
        plot_phase3d(truth, preds, phase_png, title="KMM rollout (phase space)")

    # 另存一份数值
    np.savez_compressed(os.path.join(args.outdir, "eval_artifacts.npz"),
                        A=A_eff, eigvals=np.linalg.eigvals(A_eff),
                        truth=truth, preds=preds, t=t)

    print("Done.")

if __name__ == "__main__":
    main()
