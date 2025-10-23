# make_plots.py
# Simulate a 3D system z = [x; y] with:
#   x' = a * x                                  (1D linearizable system)
#   θ' = ω,   ω' = -sin(θ)                      (conservative undamped pendulum)
# and save three figures:
#   1) x(t) vs time
#   2) Pendulum phase portrait (θ vs ω)
#   3) 3D trajectory (x, θ, ω)
#
# Usage example:
#   python make_plots.py --a -0.2 --x0 1.0 --theta0 1.2 --omega0 0.0 --T 40 --dt 0.01 --outdir outputs
#
# Figures:
#   plot1_x_time.png, plot2_pendulum_phase.png, plot3_z_3d_trajectory.png

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

def f(z, a=-0.2):
    """Vector field:
       z = [x, theta, omega]
       x' = a x
       theta' = omega
       omega' = -sin(theta)"""
    x, th, om = z
    return np.array([a*x, om, -np.sin(th)], dtype=float)

def rk4_step(fun, z, dt):
    k1 = fun(z)
    k2 = fun(z + 0.5*dt*k1)
    k3 = fun(z + 0.5*dt*k2)
    k4 = fun(z + dt*k3)
    return z + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate(z0, T=40.0, dt=0.01, a=-0.2):
    n = int(T/dt) + 1
    t = np.linspace(0.0, T, n)
    Z = np.zeros((n, 3), dtype=float)
    Z[0] = z0
    def vf(z): return f(z, a=a)
    for i in range(1, n):
        Z[i] = rk4_step(vf, Z[i-1], dt)
    return t, Z

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=float, default=-0.2, help="parameter a in x' = a x")
    p.add_argument("--x0", type=float, default=1.0, help="initial x")
    p.add_argument("--theta0", type=float, default=1.2, help="initial theta (rad)")
    p.add_argument("--omega0", type=float, default=0.0, help="initial omega")
    p.add_argument("--T", type=float, default=40.0, help="simulation horizon (seconds)")
    p.add_argument("--dt", type=float, default=0.01, help="time step")
    p.add_argument("--outdir", type=str, default="plot", help="directory to save figures")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    z0 = np.array([args.x0, args.theta0, args.omega0], dtype=float)
    t, Z = simulate(z0, T=args.T, dt=args.dt, a=args.a)
    x, theta, omega = Z[:, 0], Z[:, 1], Z[:, 2]

    # 1) x(t)
    plt.figure(figsize=(7, 4))
    plt.plot(t, x, linewidth=1.5)
    plt.xlabel("time")
    plt.ylabel("x(t)")
    plt.title("1) Time series of x(t)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "plot1_x_time.png"), dpi=150)
    plt.close()

    # 2) (theta, omega)
    plt.figure(figsize=(5.5, 5.5))
    plt.plot(theta, omega, linewidth=1.0)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\omega$")
    plt.title("2) Pendulum phase portrait (θ, ω)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "plot2_pendulum_phase.png"), dpi=150)
    plt.close()

    # 3) 3D (x, theta, omega)
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, theta, omega, linewidth=1.0)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\theta$")
    ax.set_zlabel(r"$\omega$")
    ax.set_title("3) 3D trajectory of z = (x, θ, ω)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "plot3_z_3d_trajectory.png"), dpi=150)
    plt.close(fig)

    print("Saved figures to:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
