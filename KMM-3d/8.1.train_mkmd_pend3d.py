import os, argparse, json, random, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_torch(x, device):
    """Accept numpy or tensor; move to device float32."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    else:
        return torch.tensor(x, dtype=torch.float32, device=device)

# ----------------------------
# Dataset
# ----------------------------
class TrajSet(Dataset):
    def __init__(self, npz_path):
        dat = np.load(npz_path, allow_pickle=True)
        self.Z = dat["Z"].astype(np.float32)   # (m, L, 3)
        self.T = dat["T"].astype(np.float32)   # (L,)
        self.m, self.L, self.d = self.Z.shape
        meta_json = dat.get("meta", None)
        self.meta = json.loads(meta_json.item()) if meta_json is not None else {}
    def __len__(self):  return self.m
    def __getitem__(self, idx):
        return self.Z[idx]   # (L, 3)

# ----------------------------
# Model
# ----------------------------
class MemoryGRUStack(nn.Module):
    """
    多层 GRUCell 串联。
    - 输入: x_t ∈ R^{x_dim}，拼接隐状态 h_t ∈ R^{h_dim}，h_dim = num_layers * hidden_size
    - 维护每层各自的隐状态 (B, hidden_size)，逐层更新
    - 返回拼接后的 h_{t+1} (B, h_dim)
    """
    def __init__(self, input_size=3, hidden_size=3, num_layers=1):
        super().__init__()
        assert num_layers >= 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([nn.GRUCell(input_size if i == 0 else hidden_size,
                                               hidden_size) for i in range(num_layers)])

    @property
    def h_dim(self):
        return self.num_layers * self.hidden_size

    def forward(self, x_t, h_t):
        """
        x_t: (B, x_dim)
        h_t: (B, h_dim)  = concat([h^{(1)}, ..., h^{(L)}], dim=1)
        """
        B = x_t.size(0)
        # 切出每层隐状态
        hs = torch.split(h_t, self.hidden_size, dim=1) if h_t is not None and h_t.numel() > 0 else [None]*self.num_layers
        new_hs = []
        inp = x_t
        for i, cell in enumerate(self.cells):
            hi = hs[i] if hs[i] is not None else torch.zeros(B, self.hidden_size, device=x_t.device, dtype=x_t.dtype)
            hi_new = cell(inp, hi)
            new_hs.append(hi_new)
            inp = hi_new
        # 拼接所有层的隐状态作为新的 h_{t+1}
        h_next = torch.cat(new_hs, dim=1)
        return h_next

class DictionaryG(nn.Module):
    """
    g(x,h) = [x, h, x^2..x^deg, h^2..h^deg]  (不含交叉项)
    x_dim 任意，h_dim = gru_layers * gru_hidden
    输出维度 n = (x_dim + h_dim) + (x_dim + h_dim)*(deg-1) [+1 若加常数]
               = (x_dim + h_dim) * deg  [+1 if bias]
    """
    def __init__(self, x_dim=3, h_dim=3, deg=2, use_bias_const=False):
        super().__init__()
        assert deg >= 1
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.deg = deg
        self.use_bias_const = use_bias_const
        base = x_dim + h_dim
        high = base*(deg-1) if deg >= 2 else 0
        self.n = base + high + (1 if use_bias_const else 0)

    def forward(self, x, h):
        feats = [x, h]
        if self.deg >= 2:
            for p in range(2, self.deg+1):
                feats += [x**p, h**p]
        if self.use_bias_const:
            feats.append(torch.ones_like(x[:, :1]))
        return torch.cat(feats, dim=1)  # (B, n)

class DiagScaler(nn.Module):
    """可学习对角缩放 S = diag(exp(s))，稳定地缩放 g 的各通道。"""
    def __init__(self, n, init=0.0):
        super().__init__()
        self.s = nn.Parameter(torch.full((n,), float(init)))
    def forward(self, g):
        return g * torch.exp(self.s)  # element-wise

class LinearA(nn.Module):
    """A ∈ R^{n×n}, bias-free; y = A g"""
    def __init__(self, n):
        super().__init__()
        self.lin = nn.Linear(n, n, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.lin.weight)
    def forward(self, g):
        return self.lin(g)

class MKMD(nn.Module):
    """
    单步：
      g_t = g(x_t, h_t)
      y   = A S g_t       （新增 S：可学习对角缩放）
      [x̂_{t+1}; ĥ'_{t+1}] = P y  (P取前 x_dim+h_dim 维；其中 ĥ'_{t+1} 仅作线性对比)
      h_{t+1}^{TF} = GRUStack(x_t, h_t)  (teacher forcing 目标的一部分)
    """
    def __init__(self, x_dim=3, gru_hidden=3, gru_layers=1, deg=2, use_bias_const=False, use_scaler=True, scaler_init=0.0):
        super().__init__()
        self.x_dim = x_dim
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.h_dim = gru_layers * gru_hidden

        self.mem = MemoryGRUStack(input_size=x_dim, hidden_size=gru_hidden, num_layers=gru_layers)
        self.gnet = DictionaryG(x_dim=x_dim, h_dim=self.h_dim, deg=deg, use_bias_const=use_bias_const)
        self.n = self.gnet.n
        self.scaler = DiagScaler(self.n, init=scaler_init) if use_scaler else nn.Identity()
        self.A = LinearA(self.n)

    def forward_step(self, x_t, h_t):
        g_t = self.gnet(x_t, h_t)        # (B, n)
        g_t = self.scaler(g_t)           # (B, n)  // 新增缩放
        y   = self.A(g_t)                # (B, n)
        pred_proj = y[:, :self.x_dim + self.h_dim]   # [x̂_{t+1}; ĥ'_{t+1}]
        h_next_tf = self.mem(x_t, h_t)   # teacher forcing 产生 h_{t+1}
        return pred_proj, h_next_tf, g_t, y

# ----------------------------
# Training / Eval
# ----------------------------

def rollout_loss_train(model, x, device, K=5, stride=50, max_starts=1):
    """
    训练期 rollout loss（有梯度）：
    x_{t+1} = (A S g(x_t, h_t))[:x_dim]
    h_{t+1} = GRUStack(x_t, h_t)
    仅在 x 端对齐真值做 MSE。
    """
    B, L, _ = x.shape
    starts = list(range(0, max(L-1-K, 0)+1, stride))[:max_starts]
    if not starts:
        return torch.zeros((), device=device, dtype=torch.float32)

    total = 0.0; cnt = 0
    x_dim, h_dim = model.x_dim, model.h_dim
    for s in starts:
        x_cur = x[:, s, :]                              # (B, x_dim)
        h_cur = torch.zeros(B, h_dim, device=device)
        for k in range(1, K+1):
            g = model.gnet(x_cur, h_cur)
            g = model.scaler(g)
            y = model.A(g)
            x_next_from_A = y[:, :x_dim]
            h_next_from_GRU = model.mem(x_cur, h_cur)

            x_true = x[:, s+k, :]
            total = total + torch.mean((x_true - x_next_from_A)**2)
            cnt += 1

            x_cur = x_next_from_A
            h_cur = h_next_from_GRU
    return total / max(cnt, 1)


def train_one_epoch(model, loader, device, optimizer, grad_clip=1.0,
                    roll_w=1.0, roll_K=15, roll_stride=50, roll_max_starts=2, roll_every=1, epoch_idx=1,
                    alpha_x=1.0, alpha_h=0.1, norm_by_dim=True):
    """
    L = 一步监督 MSE(带权) + roll_w * rollout MSE（x 端）
    - 一步监督：对 x 与 h 的误差分别按维度归一，并用 alpha_x / alpha_h 配权（默认强调 x）。
    - rollout 仅每 roll_every 个 epoch 执行一次。
    """
    model.train()
    total = 0.0; count = 0
    x_dim, h_dim = model.x_dim, model.h_dim

    for batch in loader:
        batch = to_torch(batch, device)
        B, L, d = batch.shape
        x = batch

        h = torch.zeros(B, h_dim, device=device)
        optimizer.zero_grad()
        # 一步监督损失（加权/归一化）
        loss_1 = 0.0
        for t in range(L-1):
            x_t   = x[:, t, :]
            x_tp1 = x[:, t+1, :]
            pred_proj, h_next_tf, _, _ = model.forward_step(x_t, h)
            x_hat = pred_proj[:, :x_dim]
            h_hat = pred_proj[:, x_dim:x_dim+h_dim]

            # 维度归一化（避免 h 的维度主导）
            denom_x = x_dim if norm_by_dim else 1.0
            denom_h = h_dim if norm_by_dim else 1.0
            loss_x = torch.mean((x_tp1 - x_hat)**2) / denom_x
            loss_h = torch.mean((h_next_tf - h_hat)**2) / denom_h

            loss_1 = loss_1 + (alpha_x * loss_x + alpha_h * loss_h)
            h = h_next_tf  # teacher forcing 递推
        loss_1 = loss_1 / (L-1)

        # rollout 损失（按需计算）
        if (epoch_idx % roll_every) == 0 and roll_w > 0:
            loss_roll = rollout_loss_train(
                model, x, device,
                K=roll_K, stride=roll_stride, max_starts=roll_max_starts
            )
        else:
            loss_roll = torch.zeros((), device=device, dtype=torch.float32)

        loss = loss_1 + roll_w * loss_roll
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total += loss.detach().item() * B
        count += B
    return total / max(count, 1)

@torch.no_grad()
def evaluate_one_step(model, loader, device, alpha_x=1.0, alpha_h=0.1, norm_by_dim=True):
    """与训练同分布的一步目标 MSE（监督 [x_{t+1}; h_{t+1}^{TF}]，带相同配权/归一化）"""
    model.eval()
    total = 0.0; count = 0
    x_dim, h_dim = model.x_dim, model.h_dim

    for batch in loader:
        batch = to_torch(batch, device)
        B, L, _ = batch.shape
        x = batch
        h = torch.zeros(B, h_dim, device=device)
        mse = 0.0
        for t in range(L-1):
            x_t   = x[:, t, :]
            x_tp1 = x[:, t+1, :]
            pred_proj, h_next_tf, _, _ = model.forward_step(x_t, h)
            x_hat = pred_proj[:, :x_dim]
            h_hat = pred_proj[:, x_dim:x_dim+h_dim]

            denom_x = x_dim if norm_by_dim else 1.0
            denom_h = h_dim if norm_by_dim else 1.0
            loss_x = torch.mean((x_tp1 - x_hat)**2) / denom_x
            loss_h = torch.mean((h_next_tf - h_hat)**2) / denom_h

            mse = mse + (alpha_x * loss_x + alpha_h * loss_h)
            h = h_next_tf
        mse = (mse / (L-1)).item()
        total += mse * B
        count += B
    return total / max(count, 1)

@torch.no_grad()
def evaluate_rollout(model, loader, device, K=50, stride=25, max_starts=4):
    """
    评估 rollout（无梯度）——
    A 产出 x_{t+1}，GRUStack 产出 h_{t+1}（用于下一步），只评估 x 误差。
    """
    model.eval()
    total = 0.0; denom = 0
    x_dim, h_dim = model.x_dim, model.h_dim

    for batch in loader:
        batch = to_torch(batch, device)
        B, L, _ = batch.shape
        x_true_all = batch

        starts = list(range(0, max(L-1-K, 0)+1, stride))[:max_starts]
        if not starts:
            continue

        for s in starts:
            x_cur = x_true_all[:, s, :]               # (B, x_dim)
            h_cur = torch.zeros(B, h_dim, device=device)  # h_0

            mse_sum = 0.0; steps = 0
            for k in range(1, K+1):
                g = model.gnet(x_cur, h_cur)
                g = model.scaler(g)
                y = model.A(g)
                x_next_from_A = y[:, :x_dim]
                h_next_from_GRU = model.mem(x_cur, h_cur)

                x_true = x_true_all[:, s+k, :]
                mse_sum = mse_sum + torch.mean((x_true - x_next_from_A)**2)
                steps += 1

                x_cur = x_next_from_A
                h_cur = h_next_from_GRU

            total += (mse_sum/steps).item() * B
            denom += B

    return total / max(denom, 1)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/pend3d_dataset.npz",
                    help="Path to dataset .npz (from make_dataset.py)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--deg", type=int, default=2)
    ap.add_argument("--val_ratio", type=float, default=0.3)
    ap.add_argument("--test_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", type=str, default="outputs_mkmd")

    # GRU 结构
    ap.add_argument("--gru_layers", type=int, default=1, help="number of GRU layers (stacked GRUCell)")
    ap.add_argument("--gru_hidden", type=int, default=3, help="hidden size per GRU layer")

    # rollout *training* loss params（增强）
    ap.add_argument("--roll_train_K", type=int, default=15, help="K steps for rollout loss during training")
    ap.add_argument("--roll_train_stride", type=int, default=50, help="stride for choosing rollout starts during training")
    ap.add_argument("--roll_train_max_starts", type=int, default=2, help="max starts per trajectory during training")
    ap.add_argument("--roll_w", type=float, default=1.0, help="weight of rollout loss")
    ap.add_argument("--roll_every", type=int, default=1, help="compute rollout loss every N epochs (1=every epoch)")

    # rollout *evaluation* params
    ap.add_argument("--roll_K", type=int, default=100)
    ap.add_argument("--roll_stride", type=int, default=100)
    ap.add_argument("--roll_max_starts", type=int, default=4)

    # 一步损失的配权与归一
    ap.add_argument("--alpha_x", type=float, default=1.0)
    ap.add_argument("--alpha_h", type=float, default=0.1)
    ap.add_argument("--norm_by_dim", action="store_true", help="divide MSE by dimension for x/h terms")

    # g 缩放器
    ap.add_argument("--use_scaler", action="store_true", help="use learnable diagonal scaler before A")
    ap.add_argument("--scaler_init", type=float, default=0.0, help="initial log-scale for scaler")

    args = ap.parse_args()

    os.makedirs(args.save, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    # Data
    ds_full = TrajSet(args.data)
    m = len(ds_full)
    n_test = int(round(m * args.test_ratio))
    n_val  = int(round(m * args.val_ratio))
    n_train = m - n_val - n_test
    train_set, val_set, test_set = random_split(
        ds_full, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    # Model & Optim
    model = MKMD(
        x_dim=ds_full.d,               # 通常为 3
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        deg=args.deg,
        use_bias_const=False,
        use_scaler=args.use_scaler,
        scaler_init=args.scaler_init
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    best_val = float("inf"); best_state = None
    for ep in range(1, args.epochs+1):
        tr_mse = train_one_epoch(
            model, train_loader, device, optimizer,
            roll_w=args.roll_w,
            roll_K=args.roll_train_K,
            roll_stride=args.roll_train_stride,
            roll_max_starts=args.roll_train_max_starts,
            roll_every=args.roll_every,
            epoch_idx=ep,
            alpha_x=args.alpha_x,
            alpha_h=args.alpha_h,
            norm_by_dim=args.norm_by_dim
        )
        val_mse = evaluate_one_step(model, val_loader, device,
                                    alpha_x=args.alpha_x, alpha_h=args.alpha_h, norm_by_dim=args.norm_by_dim)
        val_roll = evaluate_rollout(
            model, val_loader, device,
            K=args.roll_K, stride=args.roll_stride, max_starts=args.roll_max_starts
        )
        print(f"[Epoch {ep:03d}] train={tr_mse:.6e} | val1={val_mse:.6e} | val_roll@{args.roll_K}={val_roll:.6e}")

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Save best
    if best_state is not None:
        torch.save(best_state, os.path.join(args.save, "mkmd_best.pt"))
        with open(os.path.join(args.save, "info.txt"), "w") as f:
            f.write(json.dumps({
                "deg": args.deg,
                "seed": args.seed,
                "sizes": {"train": n_train, "val": n_val, "test": n_test},
                "best_val_mse": best_val,
                "gru": {
                    "layers": args.gru_layers,
                    "hidden": args.gru_hidden,
                    "h_dim": model.h_dim
                },
                "rollout_train": {
                    "K": args.roll_train_K,
                    "stride": args.roll_train_stride,
                    "max_starts": args.roll_train_max_starts,
                    "w": args.roll_w,
                    "every": args.roll_every
                },
                "loss_weights": {
                    "alpha_x": args.alpha_x,
                    "alpha_h": args.alpha_h,
                    "norm_by_dim": bool(args.norm_by_dim)
                },
                "scaler": {
                    "used": bool(args.use_scaler),
                    "init": args.scaler_init,
                    "n": model.n
                }
            }, indent=2))
        print("Saved:", os.path.join(args.save, "mkmd_best.pt"))

    # Test (one-step & rollout)
    if best_state is not None:
        model.load_state_dict(best_state)
    test1 = evaluate_one_step(model, test_loader, device,
                               alpha_x=args.alpha_x, alpha_h=args.alpha_h, norm_by_dim=args.norm_by_dim)
    test_roll = evaluate_rollout(
        model, test_loader, device,
        K=args.roll_K, stride=args.roll_stride, max_starts=args.roll_max_starts
    )
    print(f"[TEST] one-step={test1:.6e} | rollout@{args.roll_K}={test_roll:.6e}")

    # A 的谱半径（诊断）
    with torch.no_grad():
        W = model.A.lin.weight.detach().cpu().numpy()
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        print(f"A spectral radius (full): {rho:.4f}  |  n={W.shape[0]} | x_dim={model.x_dim} | h_dim={model.h_dim}")

if __name__ == "__main__":
    main()
