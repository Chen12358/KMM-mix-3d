# hankel_two_stage_train.py
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def build_hankel_columns(traj, Ld):
    T = traj.shape[0]
    D = 3 * Ld
    K = T - Ld
    if K <= 0:
        raise ValueError(f"Trajectory too short for Ld={Ld}. Need T >= Ld+1.")
    X_cols = np.zeros((D, K), dtype=float)
    Y_cols = np.zeros((D, K), dtype=float)
    def hankel_at(t):
        cols = []
        for ell in range(Ld):
            cols.append(traj[t - ell])
        return np.concatenate(cols, axis=0)
    for j, t in enumerate(range(Ld-1, T-1)):
        X_cols[:, j] = hankel_at(t)
        Y_cols[:, j] = hankel_at(t+1)
    return X_cols, Y_cols

def assemble_matrices(Z, indices, Ld):
    X_list, Y_list = [], []
    for idx in indices:
        Xi, Yi = build_hankel_columns(Z[idx], Ld)
        X_list.append(Xi); Y_list.append(Yi)
    X = np.concatenate(X_list, axis=1)
    Y = np.concatenate(Y_list, axis=1)
    return X, Y

class MLP(nn.Module):
    def __init__(self, d_in, hidden=256, depth=3, d_out=None, act='tanh', dropout=0.0):
        super().__init__()
        if d_out is None:
            d_out = d_in
        acts = {'tanh': nn.Tanh(),'relu': nn.ReLU(),'gelu': nn.GELU(),'elu': nn.ELU(),'silu': nn.SiLU()}
        layers = []
        dims = [d_in] + [hidden]*(depth-1) + [d_out]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), acts.get(act, nn.Tanh())]
            if dropout > 0: layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def compute_norm_stats(U):
    mu = U.mean(axis=1, keepdims=True); sigma = U.std(axis=1, keepdims=True) + 1e-8; return mu, sigma
def normalize(U, mu, sigma): return (U - mu) / sigma

def train(P, G, A, X_tr, Y_tr, X_va, Y_va, mu_X, sg_X, mu_UA, sg_UA, args, device):
    P = P.to(device); G = G.to(device); P.train(); G.train()
    tr_ds = TensorDataset(torch.from_numpy(X_tr.T).float(), torch.from_numpy(Y_tr.T).float())
    va_ds = TensorDataset(torch.from_numpy(X_va.T).float(), torch.from_numpy(Y_va.T).float())
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    opt = torch.optim.Adam(list(P.parameters()) + list(G.parameters()), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    A_t = torch.from_numpy(A).float().to(device)
    muX_t = torch.from_numpy(mu_X.T).float().to(device); sgX_t = torch.from_numpy(sg_X.T).float().to(device)
    muUA_t = torch.from_numpy(mu_UA.T).float().to(device); sgUA_t = torch.from_numpy(sg_UA.T).float().to(device)
    best_val = np.inf; best = None; epochs_no_improve = 0; train_curve, val_curve = [], []
    for ep in range(1, args.epochs+1):
        P.train(); G.train(); tr_loss = 0.0
        for xb, yb in tr_loader:
            xb = xb.to(device); yb = yb.to(device)
            xb_n = (xb - muX_t) / sgX_t
            GX = G(xb_n)
            U = (xb @ A_t.T) + GX
            U_n = (U - muUA_t) / sgUA_t
            Yhat = P(U_n)
            loss = torch.mean((Yhat - yb)**2)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr_ds)
        P.eval(); G.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device); yb = yb.to(device)
                xb_n = (xb - muX_t) / sgX_t
                GX = G(xb_n)
                U = (xb @ A_t.T) + GX
                U_n = (U - muUA_t) / sgUA_t
                Yhat = P(U_n)
                va_loss += torch.mean((Yhat - yb)**2).item() * xb.size(0)
        va_loss /= len(va_ds)
        train_curve.append(tr_loss); val_curve.append(va_loss); sched.step(va_loss)
        if va_loss < best_val - 1e-9:
            best_val = va_loss; best = {'P': P.state_dict(), 'G': G.state_dict(), 'epoch': ep}; epochs_no_improve = 0
        else: epochs_no_improve += 1
        if ep % max(1, args.print_every) == 0: print(f"Epoch {ep:4d} | train {tr_loss:.6e} | val {va_loss:.6e}")
        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(f"Early stopping at epoch {ep}. Best val {best_val:.6e}"); break
    if best is not None: P.load_state_dict(best['P']); G.load_state_dict(best['G'])
    return P, G, np.array(train_curve), np.array(val_curve), float(best_val)

def evaluate(P, G, A, X, Y, mu_X, sg_X, mu_UA, sg_UA, device):
    P.eval(); G.eval()
    with torch.no_grad():
        A_t = torch.from_numpy(A).float().to(device)
        muX_t = torch.from_numpy(mu_X.T).float().to(device); sgX_t = torch.from_numpy(sg_X.T).float().to(device)
        muUA_t = torch.from_numpy(mu_UA.T).float().to(device); sgUA_t = torch.from_numpy(sg_UA.T).float().to(device)
        Xb = torch.from_numpy(X.T).float().to(device); Yb = torch.from_numpy(Y.T).float().to(device)
        Xn = (Xb - muX_t) / sgX_t; GX = G(Xn); U = (Xb @ A_t.T) + GX; Un = (U - muUA_t) / sgUA_t; Yhat = P(Un)
        err = (Yhat - Yb).cpu().numpy(); mse = float(np.mean(err**2)); rmse = float(np.sqrt(mse))
    return mse, rmse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default='data/pend3d_dataset.npz')
    ap.add_argument("--split", type=str, default='hdmd_outputs/learned_A_and_split.npz')
    ap.add_argument("--Ld", type=int, default=20)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--hidden_P", type=int, default=167)
    ap.add_argument("--depth_P", type=int, default=3)
    ap.add_argument("--hidden_G", type=int, default=167)
    ap.add_argument("--depth_G", type=int, default=3)
    ap.add_argument("--act", type=str, default="tanh", choices=["tanh","relu","gelu","elu","silu"])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--outdir", type=str, default="two_stage_out")
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--early_stop", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    D = np.load(args.data, allow_pickle=True); Z = D["Z"]; meta = json.loads(str(D["meta"]))
    S = np.load(args.split, allow_pickle=True); A = S["A"]; train_idx = S["train_idx"]; test_idx_all = S["test_idx_all"]

    X_train_all, Y_train_all = assemble_matrices(Z, train_idx, args.Ld)
    X_test, Y_test = assemble_matrices(Z, test_idx_all, args.Ld)

    K_all = X_train_all.shape[1]; perm = np.random.permutation(K_all); K_val = int(np.floor(args.val_ratio * K_all))
    val_cols = perm[:K_val]; tr_cols = perm[K_val:]
    X_tr, Y_tr = X_train_all[:, tr_cols], Y_train_all[:, tr_cols]
    X_va, Y_va = X_train_all[:, val_cols], Y_train_all[:, val_cols]

    mu_X, sg_X = compute_norm_stats(X_tr)
    U_A_tr = A @ X_tr; mu_UA, sg_UA = compute_norm_stats(U_A_tr)

    Ddim = X_tr.shape[0]
    P = MLP(d_in=Ddim, hidden=args.hidden_P, depth=args.depth_P, act=args.act, dropout=args.dropout)
    G = MLP(d_in=Ddim, hidden=args.hidden_G, depth=args.depth_G, act=args.act, dropout=args.dropout)

    P, G, tr_curve, va_curve, best_val = train(P, G, A, X_tr, Y_tr, X_va, Y_va, mu_X, sg_X, mu_UA, sg_UA, args, device)

    tr_mse, tr_rmse = evaluate(P, G, A, X_tr, Y_tr, mu_X, sg_X, mu_UA, sg_UA, device)
    va_mse, va_rmse = evaluate(P, G, A, X_va, Y_va, mu_X, sg_X, mu_UA, sg_UA, device)
    te_mse, te_rmse = evaluate(P, G, A, X_test, Y_test, mu_X, sg_X, mu_UA, sg_UA, device)

    print(f"Train   MSE={tr_mse:.6e}  RMSE={tr_rmse:.6e}")
    print(f"Val     MSE={va_mse:.6e}  RMSE={va_rmse:.6e}")
    print(f"Test    MSE={te_mse:.6e}  RMSE={te_rmse:.6e}")

    torch.save({
        'P_state_dict': P.state_dict(),
        'G_state_dict': G.state_dict(),
        'A': A,
        'config': {'Ld': args.Ld,'hidden_P': args.hidden_P,'depth_P': args.depth_P,'hidden_G': args.hidden_G,'depth_G': args.depth_G,'act': args.act,'dropout': args.dropout},
        'norm_stats': {'mu_X': mu_X, 'sg_X': sg_X, 'mu_UA': mu_UA, 'sg_UA': sg_UA},
        'metrics': {'train': {'mse': tr_mse, 'rmse': tr_rmse}, 'val': {'mse': va_mse, 'rmse': va_rmse}, 'test': {'mse': te_mse, 'rmse': te_rmse}},
        'curves': {'train': tr_curve, 'val': va_curve},
    }, os.path.join(args.outdir, "two_stage_PG.pt"))
    print("Saved model to:", os.path.abspath(os.path.join(args.outdir, "two_stage_PG.pt")))

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(tr_curve, label="train"); plt.plot(va_curve, label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE loss"); plt.title("Training curves for P and G (Y ≈ P(A X + G(X)))")
    plt.legend(); plt.tight_layout()
    curves_png = os.path.join(args.outdir, "training_curves_PG.png")
    plt.savefig(curves_png, dpi=160); plt.close()
    print("Saved curves to:", os.path.abspath(curves_png))

if __name__ == "__main__":
    main()
