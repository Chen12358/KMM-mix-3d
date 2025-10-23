# hankel_residual_train.py  (F takes X as input; Y ≈ (A + F) X)
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

# ---------- Utils: Hankel construction (same ordering as before) ----------
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
    X = np.concatenate(X_list, axis=1)  # (D, K_all)
    Y = np.concatenate(Y_list, axis=1)  # (D, K_all)
    return X, Y

# ---------- Model ----------
class ResidualMLP(nn.Module):
    def __init__(self, d_in, hidden=256, depth=3, d_out=None, act='tanh', dropout=0.0):
        super().__init__()
        if d_out is None:
            d_out = d_in
        acts = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'silu': nn.SiLU(),
        }
        layers = []
        dims = [d_in] + [hidden]*(depth-1) + [d_out]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), acts.get(act, nn.Tanh())]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, u):
        return self.net(u)

# ---------- Normalization ----------
def compute_norm_stats(U):
    mu = U.mean(axis=1, keepdims=True)
    sigma = U.std(axis=1, keepdims=True) + 1e-8
    return mu, sigma

def normalize(U, mu, sigma):
    return (U - mu) / sigma

def denormalize(Uhat, mu, sigma):
    return Uhat * sigma + mu

# ---------- Training ----------
def train_model(A, X_train, Y_train, X_val, Y_val, args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Inputs to F are X; target residual is R = Y - (A @ X)
    R_train = Y_train - (A @ X_train)
    R_val   = Y_val   - (A @ X_val)

    # Normalize inputs X
    mu_X, sg_X = compute_norm_stats(X_train)
    X_train_n = normalize(X_train, mu_X, sg_X)
    X_val_n   = normalize(X_val,   mu_X, sg_X)

    d = X_train.shape[0]
    model = ResidualMLP(d_in=d, hidden=args.hidden, depth=args.depth, act=args.act, dropout=args.dropout).to(device)

    train_ds = TensorDataset(torch.from_numpy(X_train_n.T).float(), torch.from_numpy(R_train.T).float())
    val_ds   = TensorDataset(torch.from_numpy(X_val_n.T).float(),   torch.from_numpy(R_val.T).float())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)

    best_val = np.inf
    best_state = None
    patience = args.early_stop
    epochs_no_improve = 0
    train_curve, val_curve = [], []

    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)                          # F(X)
            loss = torch.mean((pred - yb)**2)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds)

        # validation
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss = torch.mean((pred - yb)**2)
                va_loss += loss.item() * xb.size(0)
        va_loss /= len(val_ds)

        train_curve.append(tr_loss); val_curve.append(va_loss)
        scheduler.step(va_loss)

        if va_loss < best_val - 1e-9:
            best_val = va_loss
            best_state = { 'model': model.state_dict(), 'epoch': epoch,
                           'opt': opt.state_dict() }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % max(1, args.print_every) == 0:
            print(f"Epoch {epoch:4d} | train {tr_loss:.6e} | val {va_loss:.6e}")

        if patience > 0 and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best val {best_val:.6e}.")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state['model'])

    stats = {
        'mu_X': mu_X, 'sg_X': sg_X,
        'train_curve': np.array(train_curve),
        'val_curve': np.array(val_curve),
        'best_val': float(best_val),
    }
    return model, stats

def evaluate(A, model, X, Y, mu_X, sg_X, device):
    with torch.no_grad():
        X_n = normalize(X, mu_X, sg_X)
        X_n_t = torch.from_numpy(X_n.T).float().to(device)
        Rhat = model(X_n_t).cpu().numpy().T       # F(X)
        Yhat = (A @ X) + Rhat                     # (A + F) X
        mse = float(np.mean((Yhat - Y)**2))
        rmse = float(np.sqrt(mse))
    return mse, rmse, Yhat

def plot_curves(train_curve, val_curve, out_png):
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(train_curve, label="train")
    plt.plot(val_curve, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Training curves (residual MLP F; input = X)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default='data/pend3d_dataset.npz', help="Path to dataset .npz")
    ap.add_argument("--split", type=str, default='hdmd_outputs/learned_A_and_split.npz', help="Path to learned_A_and_split.npz (from hankel_dmd_eval.py)")
    ap.add_argument("--Ld", type=int, default=20, help="Time-delay length used to build Hankel data")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of TRAIN set for validation (rest used for training)")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--act", type=str, default="tanh", choices=["tanh","relu","gelu","elu","silu"])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--early_stop", type=int, default=30, help="Early stopping patience (epochs). 0 to disable.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    ap.add_argument("--outdir", type=str, default="residual_model_out")
    ap.add_argument("--print_every", type=int, default=10,
                    help="How many epochs between progress prints (default: 10)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset and split/A
    D = np.load(args.data, allow_pickle=True)
    Z = D["Z"]
    meta = json.loads(str(D["meta"]))
    dt = float(meta.get("dt", 0.01))
    a_param = float(meta.get("a_param", -0.2))

    S = np.load(args.split, allow_pickle=True)
    A = S["A"]
    train_idx = S["train_idx"]
    test_idx_all = S["test_idx_all"]

    # Assemble Hankel matrices
    X_train_all, Y_train_all = assemble_matrices(Z, train_idx, args.Ld)
    X_test, Y_test = assemble_matrices(Z, test_idx_all, args.Ld)

    # Split train into train/val by columns
    K_all = X_train_all.shape[1]
    perm = np.random.permutation(K_all)
    K_val = int(np.floor(args.val_ratio * K_all))
    val_cols = perm[:K_val]
    tr_cols  = perm[K_val:]

    X_tr, Y_tr = X_train_all[:, tr_cols], Y_train_all[:, tr_cols]
    X_va, Y_va = X_train_all[:, val_cols], Y_train_all[:, val_cols]

    # Train model
    model, stats = train_model(A, X_tr, Y_tr, X_va, Y_va, args)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model.to(device)
    model.eval()

    # Evaluate on splits
    tr_mse, tr_rmse, _ = evaluate(A, model, X_tr, Y_tr, stats['mu_X'], stats['sg_X'], device)
    va_mse, va_rmse, _ = evaluate(A, model, X_va, Y_va, stats['mu_X'], stats['sg_X'], device)
    te_mse, te_rmse, _ = evaluate(A, model, X_test, Y_test, stats['mu_X'], stats['sg_X'], device)

    print(f"Train   MSE={tr_mse:.6e}  RMSE={tr_rmse:.6e}")
    print(f"Val     MSE={va_mse:.6e}  RMSE={va_rmse:.6e}")
    print(f"Test    MSE={te_mse:.6e}  RMSE={te_rmse:.6e}")

    # Save model and artifacts
    save_path = os.path.join(args.outdir, "residual_F.pt")
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'd_in': X_tr.shape[0],
            'hidden': args.hidden,
            'depth': args.depth,
            'act': args.act,
            'dropout': args.dropout,
            'Ld': args.Ld,
            'dt': dt,
            'a_param': a_param,
            'input_type': 'X',
            'target': 'Y - (A @ X)',
            'compose': 'Yhat = (A @ X) + F(X)'
        },
        'norm_stats': {
            'mu_X': stats['mu_X'],
            'sg_X': stats['sg_X'],
        },
        'A': A,
        'metrics': {
            'train': {'mse': tr_mse, 'rmse': tr_rmse},
            'val':   {'mse': va_mse, 'rmse': va_rmse},
            'test':  {'mse': te_mse, 'rmse': te_rmse},
        }
    }, save_path)
    print("Saved model to:", os.path.abspath(save_path))

    # Save curves
    curves_png = os.path.join(args.outdir, "training_curves.png")
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(stats['train_curve'], label="train")
    plt.plot(stats['val_curve'], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Training curves (residual MLP F; input = X)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_png, dpi=160)
    plt.close()
    print("Saved curves to:", os.path.abspath(curves_png))

if __name__ == "__main__":
    main()
