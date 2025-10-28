import os, argparse, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========== 与训练脚本一致的模块（支持可变 gru_layers/hidden、h_dim 动态） ==========

class MemoryGRUStack(nn.Module):
    """
    多层 GRUCell 串联；返回拼接后的 h_{t+1} ∈ R^{h_dim}，h_dim = num_layers * hidden_size
    """
    def __init__(self, input_size=3, hidden_size=3, num_layers=1):
        super().__init__()
        assert num_layers >= 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    @property
    def h_dim(self):
        return self.num_layers * self.hidden_size

    def forward(self, x_t, h_t):
        """
        x_t: (B, x_dim)
        h_t: (B, h_dim)  = concat per-layer hidden; 若为 None 则从 0 初始化
        """
        B = x_t.size(0)
        if h_t is None:
            hs = [None] * self.num_layers
        else:
            hs = torch.split(h_t, self.hidden_size, dim=1)
        new_hs = []
        inp = x_t
        for i, cell in enumerate(self.cells):
            hi = hs[i] if hs[i] is not None else torch.zeros(B, self.hidden_size, device=x_t.device, dtype=x_t.dtype)
            hi_new = cell(inp, hi)
            new_hs.append(hi_new)
            inp = hi_new
        return torch.cat(new_hs, dim=1)  # (B, h_dim)


class DictionaryG(nn.Module):
    """
    g(x,h) = [x, h, x^2..x^deg, h^2..h^deg]  (无交叉项)
    输出维度 n = (x_dim + h_dim) * deg [+1 若 use_bias_const]
    """
    def __init__(self, x_dim=3, h_dim=3, deg=2, use_bias_const=False):
        super().__init__()
        assert deg >= 1
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.deg = deg
        self.use_bias_const = use_bias_const
        base = x_dim + h_dim
        self.n = base * deg + (1 if use_bias_const else 0)

    def forward(self, x, h):
        feats = [x, h]
        if self.deg >= 2:
            for p in range(2, self.deg+1):
                feats += [x**p, h**p]
        if self.use_bias_const:
            feats.append(torch.ones_like(x[:, :1]))
        return torch.cat(feats, dim=1)  # (B, n)


class LinearA(nn.Module):
    """A ∈ R^{n×n}, bias-free; y = A g"""
    def __init__(self, n):
        super().__init__()
        self.lin = nn.Linear(n, n, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.lin.weight)  # 近似恒等（与训练初始化一致）
    def forward(self, g):
        return self.lin(g)


class MKMD(nn.Module):
    """
    两种 rollout：
      1) interleaved（默认）：(x_t,h_t) -> g -> y=A g -> x_{t+1}=P_x y, h'_{t+1}=P_h y (记录用)
                              然后 h_{t+1}=GRUStack(x_t,h_t)
                              下一步用 (x_{t+1}, h_{t+1})
      2) Aonly：teacher-forcing 把 h 推到 t0，然后固定 h，纯 A 线性推进 g
    """
    def __init__(self, x_dim=3, gru_hidden=3, gru_layers=1, deg=2, use_bias_const=False):
        super().__init__()
        self.x_dim = x_dim
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.h_dim = gru_layers * gru_hidden

        self.mem = MemoryGRUStack(input_size=x_dim, hidden_size=gru_hidden, num_layers=gru_layers)
        self.gnet = DictionaryG(x_dim=x_dim, h_dim=self.h_dim, deg=deg, use_bias_const=use_bias_const)
        self.A = LinearA(self.gnet.n)
        self.n = self.gnet.n

    def g(self, x, h):
        return self.gnet(x, h)

    # ---- 投影切片：g = [x(x_dim), h(h_dim), x^2(x_dim), h^2(h_dim), ...]
    def _px(self, y):   # 取出 x 的投影
        return y[:, :self.x_dim]
    def _ph(self, y):   # 取出 h 的线性投影（分析用）
        return y[:, self.x_dim:self.x_dim+self.h_dim]

    # ---- 交替耦合一步
    @torch.no_grad()
    def step_interleaved(self, x_t, h_t):
        """
        返回: x_next, h_next, h_lin, y_full
        """
        g_t = self.g(x_t, h_t)            # (B, n)
        y   = self.A(g_t)                 # (B, n)
        x_next  = self._px(y)             # (B, x_dim)
        h_lin   = self._ph(y)             # (B, h_dim) 仅用于分析
        h_next  = self.mem(x_t, h_t)      # (B, h_dim)
        return x_next, h_next, h_lin, y

    # ---- 交替耦合 rollout
    @torch.no_grad()
    def rollout_interleaved(self, x0, h0, K):
        """
        从给定 (x0, h0) 开始做 K 步交替耦合 rollout。
        返回：X_pred [K,x_dim], H_pred [K,h_dim], H_lin [K,h_dim]
        """
        xs, hs, hs_lin = [], [], []
        x_t, h_t = x0, h0
        for _ in range(K):
            x_next, h_next, h_lin, _ = self.step_interleaved(x_t, h_t)
            xs.append(x_next[0].cpu().numpy())
            hs.append(h_next[0].cpu().numpy())
            hs_lin.append(h_lin[0].cpu().numpy())
            x_t, h_t = x_next, h_next
        return (np.stack(xs, 0), np.stack(hs, 0), np.stack(hs_lin, 0))

    # ---- 原始 A-only 推进（冻结 h）
    @torch.no_grad()
    def rollout_A(self, g0, K):
        """ 纯 A 线性推进，不更新 GRU """
        g = g0
        outs = []
        for _ in range(K):
            g = self.A(g)
            outs.append(g)
        return outs

# ========== 工具 ==========
def load_model_from_ckpt(ckpt_path, device, override=None):
    """
    训练脚本保存的是 state_dict（best_state），不含超参。
    从同目录 info.txt 读取 deg 与 gru 配置；若提供 override 则校验一致性。
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    info_path = os.path.join(ckpt_dir, "info.txt")

    # 默认
    deg = 2
    x_dim = 3
    gru_layers = 1
    gru_hidden = 3

    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        deg = int(info.get("deg", deg))
        # 读取训练时的 GRU 设置（如果有）
        gru_info = info.get("gru", {})
        gru_layers = int(gru_info.get("layers", gru_layers))
        gru_hidden = int(gru_info.get("hidden", gru_hidden))

    # 若用户传入 override，进行一致性检查（防止形状不匹配导致加载失败）
    if override is not None:
        if "gru_layers" in override and int(override["gru_layers"]) != gru_layers:
            raise ValueError(f"Provided --gru_layers={override['gru_layers']} "
                             f"!= training layers={gru_layers} recorded in info.txt")
        if "gru_hidden" in override and int(override["gru_hidden"]) != gru_hidden:
            raise ValueError(f"Provided --gru_hidden={override['gru_hidden']} "
                             f"!= training hidden={gru_hidden} recorded in info.txt")

    model = MKMD(x_dim=x_dim, gru_hidden=gru_hidden, gru_layers=gru_layers, deg=deg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)  # 形状必须一致
    model.eval()
    meta = {"deg": deg, "x_dim": x_dim, "gru_layers": gru_layers, "gru_hidden": gru_hidden, "h_dim": model.h_dim}
    return model, meta

def plot_eigs(W, out_png):
    eigs = np.linalg.eigvals(W)
    rank = np.linalg.matrix_rank(W)
    n = W.shape[0]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(eigs.real, eigs.imag, s=12, alpha=0.8, label="eig(A)")
    ang = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(ang), np.sin(ang), lw=1.0)
    ax.axhline(0, lw=0.5, color="k", alpha=0.4)
    ax.axvline(0, lw=0.5, color="k", alpha=0.4)
    ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")
    ax.set_title("Eigenvalues of A")
    ax.set_aspect("equal", adjustable="box")
    txt = f"size(A) = {n}×{n}\nrank(A) = {rank}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    return eigs, rank, n

@torch.no_grad()
def rollout_on_traj_Aonly(model, Z, traj_id=0, t0=200, K=200, device="cpu"):
    """
    A-only：
    - teacher-forcing: 用真值把 GRU 记忆推进到 t0（与训练一致）
    - 以 g_t0 为锚，做 A-线性 rollout K 步（h 冻结）
    """
    Zt = torch.from_numpy(Z[traj_id]).float().to(device)  # [L,x_dim]
    L = Zt.shape[0]
    assert 1 <= t0 < L-1, "t0 要在 [1, L-2] 内"
    K = int(min(K, L-1-t0))

    # teacher forcing 到 t0
    h = torch.zeros(1, model.h_dim, device=device)
    for t in range(t0):
        x_t = Zt[t:t+1, :]
        h = model.mem(x_t, h)

    # g(t0) 与线性 rollout
    g0 = model.g(Zt[t0:t0+1, :], h)   # (1, n)
    outs = model.rollout_A(g0, K)
    Xhat = [gk[:, :model.x_dim].detach().cpu().numpy()[0] for gk in outs]
    Xhat = np.stack(Xhat, axis=0)      # [K,x_dim]

    # 真值对齐到未来步
    Xtrue = Z[traj_id, t0+1:t0+1+K, :]
    ts = np.arange(t0+1, t0+1+K)
    return ts, Xtrue, Xhat

@torch.no_grad()
def rollout_on_traj_interleaved(model, Z, traj_id=0, t0=0, K=200, device="cpu"):
    """
    交替耦合：从 (x0=Z[traj_id,t0], h0=0) 开始，自回归 rollout。
    返回：ts, Xtrue, Xhat, Hhat, Hlin
    """
    Zt = torch.from_numpy(Z[traj_id]).float().to(device)  # [L,x_dim]
    L = Zt.shape[0]
    assert 0 <= t0 < L-1, "t0 要在 [0, L-2] 内"
    K = int(min(K, L-1-t0))

    x0 = Zt[t0:t0+1, :]                           # (1,x_dim)
    h0 = torch.zeros(1, model.h_dim, device=device)   # 初始化 h0=0

    Xhat, Hhat, Hlin = model.rollout_interleaved(x0, h0, K)
    Xtrue = Z[traj_id, t0+1:t0+1+K, :]
    ts = np.arange(t0+1, t0+1+K)
    return ts, Xtrue, Xhat, Hhat, Hlin

def plot_time_series(ts, Xtrue, Xhat, out_png, dt=0.01, title="Time-domain trajectories: true vs pred", names=None):
    t_real = ts * dt
    x_dim = Xtrue.shape[1]
    if names is None or len(names) != x_dim:
        names = [f"x{i+1}" for i in range(x_dim)]
    fig, axs = plt.subplots(x_dim, 1, figsize=(8, 2.2*x_dim), sharex=True)
    if x_dim == 1:
        axs = [axs]
    for i in range(x_dim):
        axs[i].plot(t_real, Xtrue[:, i], lw=1.5, label=f"{names[i]} true")
        axs[i].plot(t_real, Xhat[:, i], lw=1.2, linestyle="--", label=f"{names[i]} pred")
        axs[i].set_ylabel(names[i]); axs[i].grid(alpha=0.3); axs[i].legend(loc="best", fontsize=9)
    axs[-1].set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(out_png, dpi=160); plt.close(fig)

def plot_phase_space(Xtrue, Xhat, out_png):
    # 针对 x_dim=3 的常用视图；若不是 3 维，仅画 2D 投影
    x_dim = Xtrue.shape[1]
    if x_dim >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(10,8))
        ax3d = fig.add_subplot(2,2,1, projection="3d")
        ax3d.plot(Xtrue[:,0], Xtrue[:,1], Xtrue[:,2], lw=1.5, label="true")
        ax3d.plot(Xhat[:,0], Xhat[:,1], Xhat[:,2], lw=1.2, linestyle="--", label="pred")
        ax3d.set_xlabel("x1"); ax3d.set_ylabel("x2"); ax3d.set_zlabel("x3")
        ax3d.set_title("Phase space (3D)"); ax3d.legend()

        ax = fig.add_subplot(2,2,2)
        ax.plot(Xtrue[:,1], Xtrue[:,2], lw=1.5, label="true")
        ax.plot(Xhat[:,1], Xhat[:,2], lw=1.2, linestyle="--", label="pred")
        ax.set_xlabel("x2"); ax.set_ylabel("x3")
        ax.set_title("(x2, x3) projection"); ax.grid(alpha=0.3); ax.legend()

        ax = fig.add_subplot(2,2,3)
        ax.plot(Xtrue[:,0], Xtrue[:,1], lw=1.5, label="true")
        ax.plot(Xhat[:,0], Xhat[:,1], lw=1.2, linestyle="--", label="pred")
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.set_title("(x1, x2) projection"); ax.grid(alpha=0.3); ax.legend()

        ax = fig.add_subplot(2,2,4)
        ax.plot(Xtrue[:,0], Xtrue[:,2], lw=1.5, label="true")
        ax.plot(Xhat[:,0], Xhat[:,2], lw=1.2, linestyle="--", label="pred")
        ax.set_xlabel("x1"); ax.set_ylabel("x3")
        ax.set_title("(x1, x3) projection"); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    else:
        # 简化：只画 2D
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(Xtrue[:,0], Xtrue[:,1], lw=1.5, label="true")
        ax.plot(Xhat[:,0], Xhat[:,1], lw=1.2, linestyle="--", label="pred")
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.set_title("Phase space (2D)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def plot_h_series(ts, Hhat, Hlin, out_png, dt=0.01, title="Memory trajectories: GRU(h) vs linear h'"):
    t_real = ts * dt
    h_dim = Hhat.shape[1]
    rows = h_dim
    fig, axs = plt.subplots(rows, 1, figsize=(8, 2.0*rows), sharex=True)
    if rows == 1:
        axs = [axs]
    for i in range(h_dim):
        axs[i].plot(t_real, Hhat[:, i], lw=1.5, label=f"h{i+1} (GRU)")
        axs[i].plot(t_real, Hlin[:, i], lw=1.2, linestyle="--", label=f"h{i+1} (P_h A g)")
        axs[i].set_ylabel(f"h{i+1}")
        axs[i].grid(alpha=0.3); axs[i].legend(loc="best", fontsize=9)
    axs[-1].set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(out_png, dpi=160); plt.close(fig)

# ========== 主函数 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="outputs_mkmd/mkmd_best.pt",
                    help="Path to saved model checkpoint (state_dict)")
    ap.add_argument("--data", type=str, default="data/pend3d_dataset.npz",
                    help="Dataset .npz path")
    ap.add_argument("--traj_id", type=int, default=0, help="which trajectory to visualize")
    ap.add_argument("--t0", type=int, default=100, help="anchor index for rollout (Aonly) or start index (interleaved)")
    ap.add_argument("--K", type=int, default=400, help="rollout steps")
    ap.add_argument("--mode", type=str, default="interleaved", choices=["interleaved", "Aonly"],
                    help="rollout mode: interleaved (默认) 或 Aonly")
    ap.add_argument("--outdir", type=str, default="viz_mkmd")

    # 新增：可输入（用于一致性校验）。注意：必须与训练时相同，否则无法加载权重。
    ap.add_argument("--gru_layers", type=int, default=1, help="should match training layers (optional check)")
    ap.add_argument("--gru_hidden", type=int, default=3, help="should match training hidden size (optional check)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 载入模型（自动读取 info.txt 推断 deg/gru；若传入 --gru_* 就校验一致性）
    override = {}
    if args.gru_layers is not None: override["gru_layers"] = args.gru_layers
    if args.gru_hidden is not None: override["gru_hidden"] = args.gru_hidden
    model, meta = load_model_from_ckpt(args.ckpt, device, override=override)
    print(f"[Model] loaded with deg={meta['deg']}, gru_layers={meta['gru_layers']}, gru_hidden={meta['gru_hidden']}, h_dim={meta['h_dim']}")

    # 2) 画 A 的谱
    W = model.A.lin.weight.detach().cpu().numpy()
    eigs_png = os.path.join(args.outdir, "A_eigs.png")
    eigs, rank, n = plot_eigs(W, eigs_png)
    print(f"[Eigs] size(A)={n}x{n}, rank={rank}. Saved scatter to {eigs_png}")

    # 3) 读取数据
    npz = np.load(args.data, allow_pickle=True)
    Z = npz["Z"]   # [M,L,x_dim]
    dt = 0.01
    if "meta" in npz.files:
        try:
            dt = float(json.loads(npz["meta"].item())["dt"])
        except Exception:
            pass

    # 4) rollout & 可视化
    if args.mode == "Aonly":
        ts, Xtrue, Xhat = rollout_on_traj_Aonly(
            model, Z, traj_id=args.traj_id, t0=args.t0, K=args.K, device=device
        )
        ts_png = os.path.join(args.outdir, "time_series_compare_Aonly.png")
        plot_time_series(ts, Xtrue, Xhat, ts_png, dt=dt,
                         title="Time-domain (A-only, teacher-forced h to t0)")
        print(f"[Time] saved {ts_png}")

        phase_png = os.path.join(args.outdir, "phase_space_compare_Aonly.png")
        if Xtrue.shape[1] >= 2:
            plot_phase_space(Xtrue, Xhat, phase_png)
            print(f"[Phase] saved {phase_png}")
    else:
        ts, Xtrue, Xhat, Hhat, Hlin = rollout_on_traj_interleaved(
            model, Z, traj_id=args.traj_id, t0=args.t0, K=args.K, device=device
        )
        ts_png = os.path.join(args.outdir, "time_series_compare_interleaved.png")
        plot_time_series(ts, Xtrue, Xhat, ts_png, dt=dt,
                         title="Time-domain (interleaved: x,h -> A -> P -> x', GRU -> h')",
                         names=[ "x", "theta", "omega"] if Xtrue.shape[1]==3 else None)
        print(f"[Time] saved {ts_png}")

        if Xtrue.shape[1] >= 2:
            phase_png = os.path.join(args.outdir, "phase_space_compare_interleaved.png")
            plot_phase_space(Xtrue, Xhat, phase_png)
            print(f"[Phase] saved {phase_png}")

        h_png = os.path.join(args.outdir, "memory_series_interleaved.png")
        plot_h_series(ts, Hhat, Hlin, h_png, dt=dt,
                      title="Memory: GRU h vs linear h' = P_h A g")
        print(f"[Memory] saved {h_png}")

if __name__ == "__main__":
    main()