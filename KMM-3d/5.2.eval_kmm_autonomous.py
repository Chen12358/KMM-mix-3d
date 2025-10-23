# eval_kmm_autonomous.py
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ---------- Model (must match training) ----------
def make_mlp(in_dim, out_dim, hidden=64, depth=2, act="tanh"):
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
    def __init__(self, d=3, n=3,
                 g_hidden=64, g_depth=2, g_act="tanh",
                 G_hidden=128, G_depth=4, G_act="tanh",
                 dec_hidden=128, dec_depth=2, dec_act="tanh",
                 A_init="identity", A_scale=0.99):
        super().__init__()
        self.d, self.n = d, n
        self.g   = make_mlp(d, n, hidden=g_hidden, depth=g_depth, act=g_act)
        self.Gnl = make_mlp(n, n, hidden=G_hidden, depth=G_depth, act=G_act)
        self.dec = make_mlp(n, d, hidden=dec_hidden, depth=dec_depth, act=dec_act)
        # A will be loaded from checkpoint; init is placeholder
        self.A = nn.Parameter(torch.eye(n) * A_scale)

    def observable_step(self, x):
        z = self.g(x)
        z_next_hat = z @ self.A.T + self.Gnl(z)
        return z, z_next_hat

    def state_step(self, x):
        _, z1_hat = self.observable_step(x)
        x1_hat = self.dec(z1_hat)
        return x1_hat

    def rollout(self, x0, steps):
        B = x0.shape[0]
        preds = torch.zeros(B, steps, self.d, device=x0.device, dtype=x0.dtype)
        x = x0
        for k in range(steps):
            x_next = self.state_step(x)   # 避免把 x 设成 preds 切片的视图
            preds[:, k, :] = x_next
            x = x_next
        return preds

# ---------- Utils ----------
def load_dataset(npz_path):
    D = np.load(npz_path, allow_pickle=True)
    T = D["T"]            # (Tlen,)
    Z = D["Z"]            # (m, Tlen, d)
    meta = json.loads(str(D["meta"]))
    dt = float(meta.get("dt", float(T[1]-T[0]) if len(T) >= 2 else 1.0))
    a_param = meta.get("a_param", None)
    if a_param is not None:
        a_param = float(a_param)
    return T, Z, dt, a_param

def build_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args_ckpt = ckpt["args"]
    d = args_ckpt.get("d", 3)
    n = args_ckpt.get("n", 8)
    model = KMM(
        d=d, n=n,
        g_hidden=args_ckpt.get("g_hidden",64), g_depth=args_ckpt.get("g_depth",2), g_act=args_ckpt.get("g_act","tanh"),
        G_hidden=args_ckpt.get("G_hidden",64), G_depth=args_ckpt.get("G_depth",4), G_act=args_ckpt.get("G_act","tanh"),
        dec_hidden=args_ckpt.get("dec_hidden",64), dec_depth=args_ckpt.get("dec_depth",2), dec_act=args_ckpt.get("dec_act","tanh"),
        A_init="identity", A_scale=args_ckpt.get("A_scale",0.99)
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    mean = torch.from_numpy(ckpt["mean"]).float().to(device)
    std  = torch.from_numpy(ckpt["std"]).float().to(device)
    dt   = ckpt.get("dt", 1.0)
    return model, (mean, std), dt, args_ckpt

def denorm(x, mean, std):
    return x*std + mean

# ---------- Main eval ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/pend3d_dataset.npz",
                    help="dataset (npz with T, Z, meta)")
    ap.add_argument("--ckpt", type=str, default="kmm_A/kmm_autonomous_no_obs.pt",
                    help="trained KMM checkpoint")
    ap.add_argument("--outdir", type=str, default="kmm_A_eval_outputs")

    # eigen overlay: choose one
    ap.add_argument("--use_dataset_eig", action="store_true",
                    help="overlay exp(a*dt) from dataset meta (requires a_param & dt)")
    ap.add_argument("--lam_true_re", type=float, default=None,
                    help="manually provide discrete eigenvalue (real part)")
    ap.add_argument("--lam_true_im", type=float, default=0.0,
                    help="manually provide discrete eigenvalue (imag part)")

    # rollout config
    ap.add_argument("--test_idx", type=int, default=1, help="which trajectory in test pool")
    ap.add_argument("--start", type=int, default=0, help="start index inside the chosen trajectory")
    ap.add_argument("--L", type=int, default=400, help="rollout steps")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    # 1) Load model & dataset
    model, (mean, std), dt_ckpt, args_ckpt = build_model_from_ckpt(args.ckpt, device)
    T, Z, dt_data, a_param = load_dataset(args.data)
    d = args_ckpt.get("d", 3)
    assert Z.shape[2] == d, f"Data dim {Z.shape[2]} != model d={d}"

    # 2) Print rank(A)
    A_np = model.A.detach().cpu().numpy()
    rankA = np.linalg.matrix_rank(A_np)
    print(f"A shape: {A_np.shape}, rank(A) = {rankA}")

    # 3) Plot eig(A) and overlay a discrete-time eigenvalue
    vals = np.linalg.eigvals(A_np)
    lam_overlay = None
    note = ""
    if args.use_dataset_eig and a_param is not None:
        lam_overlay = np.exp(a_param * dt_data)
        note = rf"exp(a·Δt) from dataset, a={a_param:.3g}, dt={dt_data:.3g}"
    elif args.lam_true_re is not None:
        lam_overlay = complex(args.lam_true_re, args.lam_true_im)
        note = f"manual λ_true = {lam_overlay.real:.3g} + {lam_overlay.imag:.3g}i"
    # else: no overlay

    plt.figure(figsize=(6.2,5.8))
    if lam_overlay is not None:
        plt.scatter([lam_overlay.real], [lam_overlay.imag], marker="*", s=160, label="λ_true (overlay)")
    plt.scatter(np.real(vals), np.imag(vals), s=20, label="eig(A)")
    plt.axhline(0, linewidth=0.8); plt.axvline(0, linewidth=0.8)
    plt.xlabel("Re(λ)"); plt.ylabel("Im(λ)")
    ttl = "Eigenvalues of A"
    if note:
        ttl += f"\n({note})"
    plt.title(ttl)
    plt.legend(loc="best"); plt.tight_layout()
    eigfig = os.path.join(args.outdir, "eigvals_overlay.png")
    plt.savefig(eigfig, dpi=160); plt.close()

    # 4) Rollout on one trajectory
    # build test pool same as training split rule (train_ratio in ckpt args)
    m = Z.shape[0]; Tlen = Z.shape[1]
    train_ratio = args_ckpt.get("train_ratio", 0.1)
    rng = np.random.default_rng(args_ckpt.get("seed", 0))
    perm = rng.permutation(m)
    m_train = max(1, int(np.floor(train_ratio*m)))
    test_idx_all = perm[m_train:] if m_train < m else perm[m_train-1:m_train]

    if len(test_idx_all) == 0:
        raise ValueError("No test trajectories available with this split.")
    test_choice = int(test_idx_all[min(args.test_idx, len(test_idx_all)-1)])
    truth = torch.from_numpy(Z[test_choice]).float().to(device)  # (Tlen, d)

    # normalize using ckpt stats
    truth_n = (truth - mean) / std

    s = max(0, min(Tlen - args.L - 1, args.start))
    x0 = truth_n[s:s+1, :]
    preds = model.rollout(x0, args.L).squeeze(0)  # (L,d)

    # denorm for plotting
    truth_seg = truth[s:s+args.L, :].cpu().numpy()
    preds_dn  = denorm(preds, mean, std).detach().cpu().numpy()
    t = np.arange(s, s+args.L) * dt_data

    # time series
    lbls = [f"x{i+1}" for i in range(d)]
    plt.figure(figsize=(8.2,5.2))
    for i in range(d):
        plt.plot(t, truth_seg[:,i], label=f"{lbls[i]} true", linewidth=1.1)
        plt.plot(t, preds_dn[:,i],  "--", label=f"{lbls[i]} pred",  linewidth=1.1)
    plt.xlabel("time"); plt.ylabel("state value")
    plt.title("KMM rollout: time series")
    plt.legend(ncol=2); plt.tight_layout()
    tsfig = os.path.join(args.outdir, "rollout_time.png")
    plt.savefig(tsfig, dpi=160); plt.close()

    # phase space
    if d >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(7.0,5.8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(truth_seg[:,0], truth_seg[:,1], truth_seg[:,2], label="truth", linewidth=1.2)
        ax.plot(preds_dn[:,0],  preds_dn[:,1],  preds_dn[:,2],  "--", label="pred", linewidth=1.2)
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
        ax.set_title("KMM rollout: phase space (3D)")
        ax.legend(); fig.tight_layout()
        psfig = os.path.join(args.outdir, "rollout_phase3d.png")
        fig.savefig(psfig, dpi=160); plt.close(fig)
    else:
        plt.figure(figsize=(6.0,5.2))
        plt.plot(truth_seg[:,0], truth_seg[:,1], label="truth", linewidth=1.2)
        plt.plot(preds_dn[:,0],  preds_dn[:,1],  "--", label="pred", linewidth=1.2)
        plt.xlabel("x1"); plt.ylabel("x2"); plt.title("KMM rollout: phase space (2D)")
        plt.legend(); plt.tight_layout()
        psfig = os.path.join(args.outdir, "rollout_phase2d.png")
        plt.savefig(psfig, dpi=160); plt.close()

    # save info
    np.savez_compressed(
        os.path.join(args.outdir, "eval_artifacts.npz"),
        A=A_np, eigvals=vals, rankA=rankA, lam_overlay=lam_overlay if 'lam_overlay' in locals() else None,
        preds=preds_dn, truth=truth_seg, t=t, test_choice=test_choice, start=s
    )

    print("=== Eval summary ===")
    print(f"rank(A) = {rankA}")
    print("Saved:")
    print(" -", eigfig)
    print(" -", tsfig)
    print(" -", psfig)
    print(" -", os.path.join(args.outdir, "eval_artifacts.npz"))

if __name__ == "__main__":
    main()
