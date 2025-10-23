# eval_kmm_rollout_IplusA_fixed.py
# 1) 画 eig(I + A_param)（优先使用 ckpt 中保存的 A_effective）
# 2) 选择一条轨迹做滚动预测：时域对比 + 3D 相空间对比（若 d=3）
# 3) 严格对齐训练时的模型结构（Identity Lift），严格加载 state_dict

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ----------------- Model blocks (与训练时完全一致) -----------------
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


class GIdentityLift(nn.Module):
    """Hard identity lift: g(x) = concat[x, extra(x)]，前 d 维硬等于 x"""
    def __init__(self, d, n, hidden=128, depth=2, act="tanh"):
        super().__init__()
        assert n >= d
        self.d, self.n = d, n
        extra_dim = n - d
        self.extra = make_mlp(d, extra_dim, hidden=hidden, depth=depth, act=act) if extra_dim > 0 else None

    def forward(self, x):  # x: (B,d)
        if self.n == self.d:
            return x
        extra = self.extra(x)  # (B, n-d)
        return torch.cat([x, extra], dim=-1)


class KMM(nn.Module):
    """A = I + A_param, decoder = slice 前 d 维"""
    def __init__(self, d=3, n=8,
                 g_hidden=128, g_depth=3, g_act="gelu",
                 G_hidden=128, G_depth=3, G_act="gelu"):
        super().__init__()
        assert n >= d
        self.d, self.n = d, n
        self.g   = GIdentityLift(d, n, hidden=g_hidden, depth=g_depth, act=g_act)
        self.Gnl = make_mlp(n, n, hidden=G_hidden, depth=G_depth, act=G_act)
        self.A_param = nn.Parameter(torch.zeros(n, n))  # 将从 state_dict 加载

    def get_A(self):
        I = torch.eye(self.n, device=self.A_param.device, dtype=self.A_param.dtype)
        return I + self.A_param

    def observable_step(self, x):
        z = self.g(x)                 # (B,n), 前 d 维 == x（硬约束）
        A = self.get_A()              # (n,n)
        z1 = z @ A.T + self.Gnl(z)    # (B,n)
        return z, z1

    def state_step(self, x):
        _, z1 = self.observable_step(x)
        return z1[:, :self.d]         # slice decoder

    def rollout(self, x0, steps):
        B, d = x0.shape[0], self.d
        preds = torch.zeros(B, steps, d, device=x0.device, dtype=x0.dtype)
        x = x0
        for k in range(steps):
            x = self.state_step(x)
            preds[:, k, :] = x
        return preds


# ----------------- plotting helpers -----------------
def plot_eigs(A, out_png, title_prefix="Eigenvalues of (I + A_param)"):
    vals = np.linalg.eigvals(A)
    rankA = np.linalg.matrix_rank(A)
    plt.figure(figsize=(6.2, 5.8))
    plt.scatter(np.real(vals), np.imag(vals), s=22, label="eig(I+A)")
    plt.axhline(0, linewidth=0.8); plt.axvline(0, linewidth=0.8)
    plt.xlabel("Re(λ)"); plt.ylabel("Im(λ)")
    plt.title(f"{title_prefix}  (rank={rankA}, n={A.shape[0]})")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()
    print(f"[INFO] A shape={A.shape}, rank(A)={rankA} -> saved {out_png}")


def plot_time_series(t, truth, preds, out_png, title="KMM rollout (time domain)"):
    d = truth.shape[1]
    lbls = [f"x{i+1}" for i in range(d)]
    plt.figure(figsize=(8, 5))
    for i in range(d):
        plt.plot(t, truth[:, i], label=f"{lbls[i]} true", linewidth=1.1)
        plt.plot(t, preds[:, i], "--", label=f"{lbls[i]} pred", linewidth=1.1)
    plt.xlabel("time"); plt.ylabel("state value"); plt.title(title)
    plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()
    print(f"[INFO] saved {out_png}")


def plot_phase3d(truth, preds, out_png, title="KMM rollout (phase space)"):
    assert truth.shape[1] == 3, "3D 相空间图仅支持 d=3"
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], linewidth=1.2, label="truth")
    ax.plot(preds[:, 0], preds[:, 1], preds[:, 2], "--", linewidth=1.2, label="pred")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    print(f"[INFO] saved {out_png}")


# ----------------- utilities -----------------
def load_dataset(npz_path: str):
    D = np.load(npz_path, allow_pickle=True)
    T = D["T"]; Z = D["Z"]
    dt = None
    if "meta" in D:
        meta = json.loads(str(D["meta"]))
        dt = float(meta.get("dt", float(T[1]-T[0]) if len(T) >= 2 else 1.0))
    else:
        dt = float(T[1]-T[0]) if len(T) >= 2 else 1.0
    return T, Z, dt


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="kmm_outputs_no_obs_slice_dec_idlift/kmm_autonomous_no_obs_slice_dec_idlift.pt",
                    help="path to trained checkpoint (*.pt)")
    ap.add_argument("--data", type=str, default="data/pend3d_dataset.npz",
                    help="dataset npz used in training")
    ap.add_argument("--outdir", type=str, default="kmm2_eval_outputs",
                    help="directory to save outputs")
    ap.add_argument("--test_idx", type=int, default=10, help="trajectory index to evaluate")
    ap.add_argument("--start", type=int, default=500, help="start index on trajectory")
    ap.add_argument("--N", type=int, default=500, help="rollout length")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    # ----- load checkpoint -----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved_args = ckpt.get("args", {})
    d = int(saved_args.get("d", 3))
    n = int(saved_args.get("n", 8))

    mean = torch.from_numpy(ckpt.get("mean", np.zeros(d))).float()
    std  = torch.from_numpy(ckpt.get("std",  np.ones(d))).float()
    dt   = ckpt.get("dt", 1.0)

    # ----- dataset -----
    T, Z, dt2 = load_dataset(args.data)
    if dt2 is not None:
        dt = dt2
    m, Tlen, d_data = Z.shape
    assert d_data == d, f"dataset d={d_data} != ckpt d={d}"

    # ----- 1) eig(I + A_param) -----
    if "A_effective" in ckpt:
        A_eff = ckpt["A_effective"]
    else:
        sd = ckpt["state_dict"]
        a_keys = [k for k in sd.keys() if k.endswith("A_param")]
        if not a_keys:
            raise KeyError("Cannot find 'A_param' in state_dict.")
        A_param = sd[a_keys[0]].detach().cpu().numpy()
        A_eff = np.eye(A_param.shape[0]) + A_param

    plot_eigs(A_eff, os.path.join(args.outdir, "A_eigs.png"))

    # ----- 2) build model with EXACT same hyperparams and load weights -----
    model = KMM(
        d=d, n=n,
        g_hidden=int(saved_args.get("g_hidden", 128)),
        g_depth =int(saved_args.get("g_depth", 3)),
        g_act   =saved_args.get("g_act", "gelu"),
        G_hidden=int(saved_args.get("G_hidden", 128)),
        G_depth =int(saved_args.get("G_depth", 3)),
        G_act   =saved_args.get("G_act", "gelu"),
    ).to(device)

    # 先尝试 strict=False 打印差异，帮助排查；随后 strict=True 真正加载
    diff = model.load_state_dict(ckpt["state_dict"], strict=False)
    if hasattr(diff, "missing_keys") and hasattr(diff, "unexpected_keys"):
        print("[DEBUG] (precheck) missing keys:", diff.missing_keys)
        print("[DEBUG] (precheck) unexpected keys:", diff.unexpected_keys)
    # 现在严格加载（若失败，提示用户检查模型定义是否一致）
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # ----- sanity check: Identity Lift 是否生效 -----
    with torch.no_grad():
        x_dummy = torch.randn(4, d, device=device)
        z_dummy = model.g(x_dummy)
        err = (z_dummy[:, :d] - x_dummy).abs().max().item()
        print(f"[SANITY] max|g(x)[:d] - x| = {err:.3e}  (应接近 0)")
        if err > 1e-6:
            print("[WARN] Identity lift 前 d 维与 x 不一致，请确认模型结构与训练时一致。")

    # ----- 3) rollout on a chosen trajectory -----
    Z_t = torch.from_numpy(Z).float()
    Z_n = (Z_t - mean) / (std + 1e-8)

    test_idx = max(0, min(m-1, args.test_idx))
    s = max(0, min(Tlen-args.N-1, args.start))
    x0 = Z_n[test_idx, s:s+1, :].to(device)

    with torch.no_grad():
        preds_n = model.rollout(x0, args.N).squeeze(0).cpu().numpy()  # (N,d)

    # 去规范化
    truth = Z[test_idx, s:s+args.N, :]                 # (N,d)
    preds = preds_n * std.numpy() + mean.numpy()
    t = np.arange(s, s+args.N) * dt

    # 画图
    plot_time_series(t, truth, preds,
                     os.path.join(args.outdir, "rollout_time.png"),
                     title="KMM rollout (time domain)")
    if d == 3:
        plot_phase3d(truth, preds,
                     os.path.join(args.outdir, "rollout_phase3d.png"),
                     title="KMM rollout (phase space)")

    # 保存数值便于复现
    np.savez_compressed(os.path.join(args.outdir, "eval_dump.npz"),
                        A=A_eff, eigvals=np.linalg.eigvals(A_eff),
                        truth=truth, preds=preds, t=t)
    print("[DONE] Outputs saved to:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
