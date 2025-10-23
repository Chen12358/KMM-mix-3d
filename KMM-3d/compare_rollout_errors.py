# compare_rollout_errors.py
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ---------------- Hankel helpers ----------------
def hankel_vector_at(traj, Ld, t):
    cols = []
    for ell in range(Ld):
        cols.append(traj[t - ell])
    return np.concatenate(cols, axis=0)

# ---------------- Models (same as training) ----------------
class MLP(nn.Module):
    def __init__(self, d_in, hidden=256, depth=3, d_out=None, act='tanh', dropout=0.0):
        super().__init__()
        if d_out is None:
            d_out = d_in
        acts = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'gelu': nn.GELU(), 'elu': nn.ELU(), 'silu': nn.SiLU()}
        layers = []
        dims = [d_in] + [hidden]*(depth-1) + [d_out]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), acts.get(act, nn.Tanh())]
            if dropout > 0: layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ---------------- Rollouts ----------------
def rollout_A(A, v0, N):
    d = v0.shape[0]
    v = v0.copy()
    preds = np.zeros((N, 3), dtype=float)
    for k in range(N):
        preds[k] = v[:3]
        v = A @ v
    return preds

def rollout_A_F(A, F_model, norm_stats, v0, N, device, input_type='X'):
    d = v0.shape[0]
    v = v0.copy()
    preds = np.zeros((N, 3), dtype=float)
    A_t = torch.from_numpy(A).float().to(device)
    mu = norm_stats.get('mu_X') if 'mu_X' in norm_stats else norm_stats.get('mu_U')
    sg = norm_stats.get('sg_X') if 'sg_X' in norm_stats else norm_stats.get('sg_U')
    if mu is None or sg is None:
        raise ValueError("Missing normalization stats for F. Expected mu_X/sg_X or mu_U/sg_U.")
    mu_t = torch.from_numpy(mu.T).float().to(device)
    sg_t = torch.from_numpy(sg.T).float().to(device)
    with torch.no_grad():
        for k in range(N):
            preds[k] = v[:3]
            v_col = v.reshape(1, d).astype(np.float32)
            xb = torch.from_numpy(v_col).to(device)
            if input_type.upper().startswith('X'):
                inp = xb
            else:
                inp = xb @ A_t.T
            inp_n = (inp - mu_t) / sg_t
            r = F_model(inp_n).cpu().numpy().reshape(d)
            v = (A @ v) + r
    return preds

def rollout_two_stage(A, P, G, v0, mu_X, sg_X, mu_UA, sg_UA, N, device):
    d = v0.shape[0]
    v = v0.copy()
    preds = np.zeros((N, 3), dtype=float)
    A_t = torch.from_numpy(A).float().to(device)
    muX_t = torch.from_numpy(mu_X.T).float().to(device)
    sgX_t = torch.from_numpy(sg_X.T).float().to(device)
    muUA_t = torch.from_numpy(mu_UA.T).float().to(device)
    sgUA_t = torch.from_numpy(sg_UA.T).float().to(device)
    with torch.no_grad():
        for k in range(N):
            preds[k] = v[:3]
            v_col = v.reshape(1, d).astype(np.float32)
            xb = torch.from_numpy(v_col).to(device)
            x_n = (xb - muX_t) / sgX_t
            GX = G(x_n)
            Av = xb @ A_t.T
            U = Av + GX
            U_n = (U - muUA_t) / sgUA_t
            v_next = P(U_n).cpu().numpy().reshape(d)
            v = v_next
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/pend3d_dataset.npz")
    ap.add_argument("--split", type=str, default="hdmd_outputs/learned_A_and_split.npz")
    ap.add_argument("--F_ckpt", type=str, default="residual_model_out/residual_F.pt")
    ap.add_argument("--PG_ckpt", type=str, default="two_stage_out/two_stage_PG.pt")
    ap.add_argument("--Ld", type=int, default=20)
    ap.add_argument("--test_idx", type=int, default=0)
    ap.add_argument("--N", type=int, default=600)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--outdir", type=str, default="compare_rollout_out")
    ap.add_argument("--methods", type=str, default="A,PG",
                    help="Comma-separated subset of {A,AF,PG} to compare, e.g., 'A,PG' or 'AF'")
    args = ap.parse_args()

    methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    valid = {"A","AF","PG"}
    for m in methods:
        if m not in valid:
            raise ValueError(f"Unknown method '{m}'. Choose from A, AF, PG.")

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Load data and split/A
    D = np.load(args.data, allow_pickle=True)
    Z = D["Z"]; T = D["T"]
    meta = json.loads(str(D["meta"])); dt = float(meta.get("dt", T[1]-T[0]))

    S = np.load(args.split, allow_pickle=True)
    A = S["A"]; test_idx_all = S["test_idx_all"]

    # Choose test trajectory
    if len(test_idx_all) == 0: raise ValueError("Empty test set in split file.")
    tidx = test_idx_all[min(args.test_idx, len(test_idx_all)-1)]
    traj = Z[tidx]; Tlen = traj.shape[0]
    if Tlen <= args.Ld: raise ValueError(f"Chosen trajectory too short (Tlen={Tlen}) for Ld={args.Ld}.")
    v0 = hankel_vector_at(traj, args.Ld, t=args.Ld-1)
    N = min(args.N, Tlen - args.Ld)
    truth = traj[args.Ld-1:args.Ld-1+N]
    t_axis = (args.Ld - 1 + np.arange(N)) * dt

    preds = {}
    errs = {}

    if "A" in methods:
        preds["A"] = rollout_A(A, v0, N)
        errs["A"] = np.linalg.norm(preds["A"] - truth, axis=1)

    if "AF" in methods:
        F_ckpt = torch.load(args.F_ckpt, map_location=device)
        F_cfg = F_ckpt["config"]
        d_in = F_cfg["d_in"]
        F_model = MLP(d_in=d_in, hidden=F_cfg["hidden"], depth=F_cfg["depth"],
                      act=F_cfg["act"], dropout=F_cfg["dropout"]).to(device)
        F_model.load_state_dict(F_ckpt["state_dict"]); F_model.eval()
        input_type = F_cfg.get("input_type", "X")
        norm_stats = F_ckpt["norm_stats"]
        preds["AF"] = rollout_A_F(A, F_model, norm_stats, v0, N, device, input_type=input_type)
        errs["AF"] = np.linalg.norm(preds["AF"] - truth, axis=1)

    if "PG" in methods:
        PG_ckpt = torch.load(args.PG_ckpt, map_location=device)
        cfg = PG_ckpt["config"]
        mu_X = PG_ckpt["norm_stats"]["mu_X"]; sg_X = PG_ckpt["norm_stats"]["sg_X"]
        mu_UA = PG_ckpt["norm_stats"]["mu_UA"]; sg_UA = PG_ckpt["norm_stats"]["sg_UA"]
        d = int(mu_X.shape[0])
        P = MLP(d_in=d, hidden=cfg["hidden_P"], depth=cfg["depth_P"],
                act=cfg["act"], dropout=cfg["dropout"]).to(device)
        G = MLP(d_in=d, hidden=cfg["hidden_G"], depth=cfg["depth_G"],
                act=cfg["act"], dropout=cfg["dropout"]).to(device)
        P.load_state_dict(PG_ckpt["P_state_dict"]); G.load_state_dict(PG_ckpt["G_state_dict"])
        P.eval(); G.eval()
        preds["PG"] = rollout_two_stage(A, P, G, v0, mu_X, sg_X, mu_UA, sg_UA, N, device)
        errs["PG"] = np.linalg.norm(preds["PG"] - truth, axis=1)

    # Plot selected methods
    plt.figure(figsize=(7.5, 4.8))
    for key in methods:
        plt.plot(t_axis, errs[key], label={"A":"A only","AF":"A + F","PG":"A + G + P"}[key])
    plt.xlabel("time")
    plt.ylabel("‖pred - truth‖₂")
    ttl = "Rollout error over time"
    if len(methods) < 3:
        ttl += f" ({', '.join(methods)})"
    plt.title(ttl)
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(args.outdir, "rollout_error_over_time.png")
    plt.savefig(out_png, dpi=160); plt.close()

    # Save raw arrays
    np.savez_compressed(os.path.join(args.outdir, "rollout_errors_raw.npz"),
                        t=t_axis, methods=np.array(methods, dtype=object),
                        **{f"err_{k}": v for k,v in errs.items()},
                        truth=truth,
                        **{f"preds_{k}": v for k,v in preds.items()},
                        test_choice=int(tidx), Ld=args.Ld, N=int(N))

    print("Saved:")
    print(" -", out_png)
    print(" - rollout_errors_raw.npz")
    print("Methods compared:", methods)

if __name__ == "__main__":
    main()
