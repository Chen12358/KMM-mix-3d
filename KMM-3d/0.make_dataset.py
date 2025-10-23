# make_dataset.py
# Sample m initial conditions uniformly in [a,b]^3 for (x, θ, ω),
# simulate each trajectory with step dt for length L (L states, including the initial state),
# using the system:
#   x' = a * x
#   θ' = ω
#   ω' = -sin(θ)
# Save to a compressed .npz with fields:
#   T: (L,) time array starting at 0 with step dt
#   Z: (m, L, 3) trajectories with state ordering (x, θ, ω)
#   IC: (m, 3) initial conditions
#   meta: JSON string of metadata (dict)
#
# Usage example:
#   python make_dataset.py --a_param -0.2 --amin -2.0 --bmax 2.0 --m 200 --L 1001 --dt 0.01 --seed 0 --out data/pend3d_dataset.npz

import os, json
import argparse
import numpy as np

def f(z, a=-0.2):
    x, th, om = z
    return np.array([a*x, om, -np.sin(th)], dtype=float)

def rk4_step(fun, z, dt):
    k1 = fun(z)
    k2 = fun(z + 0.5*dt*k1)
    k3 = fun(z + 0.5*dt*k2)
    k4 = fun(z + dt*k3)
    return z + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate_L_steps(z0, L=1001, dt=0.01, a=-0.2):
    """Return time array T (length L) and states Z (L,3) starting at z0, step dt."""
    T = np.arange(L, dtype=float) * dt
    Z = np.zeros((L, 3), dtype=float)
    Z[0] = z0
    def vf(z): return f(z, a=a)
    for i in range(1, L):
        Z[i] = rk4_step(vf, Z[i-1], dt)
    return T, Z

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a_param", type=float, default=-0.2, help="parameter a in x' = a x")
    p.add_argument("--amin", type=float, default=-2.0, help="lower bound a for [a,b]^3 cube")
    p.add_argument("--bmax", type=float, default= 2.0, help="upper bound b for [a,b]^3 cube")
    p.add_argument("--m", type=int, default=200, help="number of trajectories")
    p.add_argument("--L", type=int, default=1001, help="length of each trajectory (number of states)")
    p.add_argument("--dt", type=float, default=0.01, help="time step")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--out", type=str, default="data/pend3d_dataset.npz", help="output .npz path")
    args = p.parse_args()

    # Sanity checks
    if not args.L >= 2:
        raise ValueError("L must be >= 2.")
    if not args.bmax > args.amin:
        raise ValueError("Require bmax > amin.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rng = np.random.default_rng(args.seed)
    # Sample m ICs uniformly from [a,b]^3
    low = args.amin; high = args.bmax
    IC = rng.uniform(low=low, high=high, size=(args.m, 3)).astype(float)

    # Simulate all trajectories
    T = None
    Z_all = np.zeros((args.m, args.L, 3), dtype=float)
    for i in range(args.m):
        Ti, Zi = simulate_L_steps(IC[i], L=args.L, dt=args.dt, a=args.a_param)
        if T is None:
            T = Ti
        Z_all[i] = Zi

    meta = dict(
        system="x'=a x; θ'=ω; ω'=-sin(θ)",
        a_param=args.a_param,
        amin=args.amin,
        bmax=args.bmax,
        m=args.m,
        L=args.L,
        dt=args.dt,
        state_order=["x", "theta", "omega"],
        note="Each trajectory has length L with states at times T[k]=k*dt, starting at sampled IC."
    )

    np.savez_compressed(args.out, T=T, Z=Z_all, IC=IC, meta=json.dumps(meta))
    print("Saved dataset to:", os.path.abspath(args.out))
    print("Shapes: T", T.shape, "Z", Z_all.shape, "IC", IC.shape)

if __name__ == "__main__":
    main()
