"""
Generate all figures and animations for the Bayesian Optimization explainer.

Produces:
  images/01_blackbox_function.png        - Mystery function we want to optimize
  images/02_gp_prior.png                 - GP prior (before seeing data)
  images/03_gp_posterior_1.png           - GP posterior after 1 observation
  images/04_gp_posterior_3.png           - GP posterior after 3 observations
  images/05_acquisition_function.png     - Acquisition function (EI) with next query
  images/06_exploration_exploitation.png - Exploration vs exploitation trade-off
  images/07_bo_loop_success.gif          - BO loop finding global minimum (success)
  images/08_bo_success_result.png        - Final result of success run
  images/09_bo_loop_local.gif            - BO loop stuck at local minimum (failure)
  images/10_bo_local_result.png          - Final result of failure run
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import os
from PIL import Image
import io
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "figure.dpi": 150,
    }
)

COLORS = {
    "true_fn": "#2c3e50",
    "mean": "#2980b9",
    "ci": "#85c1e9",
    "obs": "#e74c3c",
    "acq": "#27ae60",
    "next": "#f39c12",
    "exploit": "#8e44ad",
    "explore": "#16a085",
}

OUT = "images"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Ground-truth objective (unknown to the optimizer)
# ---------------------------------------------------------------------------
def true_function(x):
    """A wiggly function with a clear global minimum."""
    return np.sin(3 * x) * x + 0.5 * np.cos(5 * x) + 0.1 * x**2 - 1.5


X_PLOT = np.linspace(0, 6, 500)
Y_TRUE = true_function(X_PLOT)


# ---------------------------------------------------------------------------
# Minimal GP implementation (RBF kernel, for pedagogical clarity)
# ---------------------------------------------------------------------------
def rbf_kernel(x1, x2, length_scale=1.0, variance=1.5):
    """Squared-exponential (RBF) kernel."""
    sq_dist = (x1[:, None] - x2[None, :]) ** 2
    return variance * np.exp(-0.5 * sq_dist / length_scale**2)


def gp_posterior(X_train, Y_train, X_test, noise=1e-6, length_scale=0.8, variance=1.5):
    """Compute GP posterior mean and std."""
    K = rbf_kernel(X_train, X_train, length_scale, variance) + noise * np.eye(
        len(X_train)
    )
    K_s = rbf_kernel(X_train, X_test, length_scale, variance)
    K_ss = rbf_kernel(X_test, X_test, length_scale, variance)
    K_inv = np.linalg.inv(K)

    mu = K_s.T @ K_inv @ Y_train
    cov = K_ss - K_s.T @ K_inv @ K_s
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    return mu, std


def gp_prior_samples(X_test, n_samples=5, length_scale=0.8, variance=1.5):
    """Draw samples from the GP prior."""
    K = rbf_kernel(X_test, X_test, length_scale, variance) + 1e-6 * np.eye(len(X_test))
    return np.random.multivariate_normal(np.zeros(len(X_test)), K, size=n_samples)


def expected_improvement(mu, std, y_best, xi=0.01):
    """Expected Improvement acquisition function."""
    with np.errstate(divide="ignore", invalid="ignore"):
        imp = y_best - mu - xi  # we minimize
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei


# ---------------------------------------------------------------------------
# Figure 1: Black-box function
# ---------------------------------------------------------------------------
def fig01_blackbox():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        X_PLOT,
        Y_TRUE,
        color=COLORS["true_fn"],
        lw=2.5,
        label="Unknown function  $f(x)$",
    )

    # Mark global min
    idx_min = np.argmin(Y_TRUE)
    ax.scatter(
        X_PLOT[idx_min],
        Y_TRUE[idx_min],
        s=120,
        zorder=5,
        color=COLORS["next"],
        edgecolors="k",
        linewidths=1.2,
        label="Global minimum (goal)",
    )

    # Gray-out the function to emphasize "unknown"
    ax.fill_between(X_PLOT, Y_TRUE - 3, Y_TRUE + 3, alpha=0.06, color="gray")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("The Problem: Find the Minimum of an Expensive Black-Box Function")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/01_blackbox_function.png")
    plt.close(fig)
    print("  [1/8] 01_blackbox_function.png")


# ---------------------------------------------------------------------------
# Figure 2: GP prior
# ---------------------------------------------------------------------------
def fig02_gp_prior():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    samples = gp_prior_samples(X_PLOT, n_samples=5)
    for i, s in enumerate(samples):
        ax.plot(
            X_PLOT, s, lw=1.2, alpha=0.7, label=f"Sample {i + 1}" if i < 3 else None
        )

    # Prior mean = 0, prior std
    K_diag = np.full_like(X_PLOT, 1.5)  # variance
    std_prior = np.sqrt(K_diag)
    ax.fill_between(
        X_PLOT,
        -2 * std_prior,
        2 * std_prior,
        alpha=0.15,
        color=COLORS["ci"],
        label="95% confidence",
    )
    ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("GP Prior — Before Seeing Any Data")
    ax.legend(loc="upper right", ncol=2)
    ax.set_xlim(0, 6)
    ax.set_ylim(-5, 5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/02_gp_prior.png")
    plt.close(fig)
    print("  [2/8] 02_gp_prior.png")


# ---------------------------------------------------------------------------
# Figure 3 & 4: GP posterior (1 and 3 observations)
# ---------------------------------------------------------------------------
def _plot_posterior(ax, X_obs, Y_obs, title, show_true=True):
    mu, std = gp_posterior(X_obs, Y_obs, X_PLOT)

    if show_true:
        ax.plot(
            X_PLOT,
            Y_TRUE,
            color=COLORS["true_fn"],
            lw=1.5,
            ls="--",
            alpha=0.4,
            label="True $f(x)$ (hidden)",
        )
    ax.plot(X_PLOT, mu, color=COLORS["mean"], lw=2, label="GP mean (prediction)")
    ax.fill_between(
        X_PLOT,
        mu - 2 * std,
        mu + 2 * std,
        alpha=0.25,
        color=COLORS["ci"],
        label="95% confidence",
    )
    ax.scatter(
        X_obs,
        Y_obs,
        s=100,
        zorder=5,
        color=COLORS["obs"],
        edgecolors="k",
        linewidths=1.2,
        label="Observations",
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(0, 6)


def fig03_posterior_1():
    X_obs = np.array([2.0])
    Y_obs = true_function(X_obs)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _plot_posterior(ax, X_obs, Y_obs, "GP Posterior — After 1 Observation")
    fig.tight_layout()
    fig.savefig(f"{OUT}/03_gp_posterior_1.png")
    plt.close(fig)
    print("  [3/8] 03_gp_posterior_1.png")


def fig04_posterior_3():
    X_obs = np.array([1.0, 3.0, 5.0])
    Y_obs = true_function(X_obs)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _plot_posterior(ax, X_obs, Y_obs, "GP Posterior — After 3 Observations")
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_gp_posterior_3.png")
    plt.close(fig)
    print("  [4/8] 04_gp_posterior_3.png")


# ---------------------------------------------------------------------------
# Figure 5: Acquisition function
# ---------------------------------------------------------------------------
def fig05_acquisition():
    X_obs = np.array([1.0, 3.0, 5.0])
    Y_obs = true_function(X_obs)
    mu, std = gp_posterior(X_obs, Y_obs, X_PLOT)
    y_best = Y_obs.min()
    ei = expected_improvement(mu, std, y_best)
    next_x_idx = np.argmax(ei)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top: GP posterior
    ax1.plot(
        X_PLOT,
        Y_TRUE,
        color=COLORS["true_fn"],
        lw=1.5,
        ls="--",
        alpha=0.4,
        label="True $f(x)$",
    )
    ax1.plot(X_PLOT, mu, color=COLORS["mean"], lw=2, label="GP mean")
    ax1.fill_between(
        X_PLOT,
        mu - 2 * std,
        mu + 2 * std,
        alpha=0.25,
        color=COLORS["ci"],
        label="95% CI",
    )
    ax1.scatter(
        X_obs,
        Y_obs,
        s=100,
        zorder=5,
        color=COLORS["obs"],
        edgecolors="k",
        linewidths=1.2,
        label="Observations",
    )
    ax1.axvline(X_PLOT[next_x_idx], color=COLORS["next"], ls="--", lw=2, alpha=0.7)
    ax1.set_ylabel("$f(x)$")
    ax1.set_title("Surrogate Model + Acquisition Function")
    ax1.legend(loc="upper left", fontsize=10)

    # Bottom: EI
    ax2.fill_between(X_PLOT, 0, ei, alpha=0.35, color=COLORS["acq"])
    ax2.plot(X_PLOT, ei, color=COLORS["acq"], lw=2, label="Expected Improvement")
    ax2.axvline(
        X_PLOT[next_x_idx],
        color=COLORS["next"],
        ls="--",
        lw=2,
        alpha=0.7,
        label=f"Next query  $x={X_PLOT[next_x_idx]:.2f}$",
    )
    ax2.scatter(
        X_PLOT[next_x_idx],
        ei[next_x_idx],
        s=120,
        zorder=5,
        color=COLORS["next"],
        edgecolors="k",
        linewidths=1.2,
    )
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("EI$(x)$")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_xlim(0, 6)

    fig.tight_layout()
    fig.savefig(f"{OUT}/05_acquisition_function.png")
    plt.close(fig)
    print("  [5/8] 05_acquisition_function.png")


# ---------------------------------------------------------------------------
# Figure 6: Exploration vs exploitation
# ---------------------------------------------------------------------------
def fig06_exploration_exploitation():
    X_obs = np.array([1.5, 2.5])
    Y_obs = true_function(X_obs)
    mu, std = gp_posterior(X_obs, Y_obs, X_PLOT)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(X_PLOT, mu, color=COLORS["mean"], lw=2, label="GP mean")
    ax.fill_between(X_PLOT, mu - 2 * std, mu + 2 * std, alpha=0.2, color=COLORS["ci"])
    ax.scatter(
        X_obs,
        Y_obs,
        s=100,
        zorder=5,
        color=COLORS["obs"],
        edgecolors="k",
        linewidths=1.2,
        label="Observations",
    )

    # Exploitation zone: near current best
    exploit_x = X_obs[np.argmin(Y_obs)]
    ax.axvspan(exploit_x - 0.5, exploit_x + 0.5, alpha=0.15, color=COLORS["exploit"])
    ax.annotate(
        "Exploitation\n(refine near best)",
        xy=(exploit_x, mu[np.argmin(np.abs(X_PLOT - exploit_x))] - 0.5),
        fontsize=12,
        fontweight="bold",
        color=COLORS["exploit"],
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3", fc="white", ec=COLORS["exploit"], alpha=0.9
        ),
    )

    # Exploration zone: far from data, high uncertainty
    explore_x = 4.8
    ax.axvspan(explore_x - 0.6, explore_x + 0.6, alpha=0.15, color=COLORS["explore"])
    ax.annotate(
        "Exploration\n(reduce uncertainty)",
        xy=(explore_x, mu[np.argmin(np.abs(X_PLOT - explore_x))] + 1.5),
        fontsize=12,
        fontweight="bold",
        color=COLORS["explore"],
        ha="center",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.3", fc="white", ec=COLORS["explore"], alpha=0.9
        ),
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("Exploration vs Exploitation Trade-off")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/06_exploration_exploitation.png")
    plt.close(fig)
    print("  [6/8] 06_exploration_exploitation.png")


# ---------------------------------------------------------------------------
# Figure 7-10: Parameterized BO loop GIF + final result
# ---------------------------------------------------------------------------
def _bo_loop_gif(X_init, n_iters, gif_path, label_prefix, fig_num_label):
    X_obs = X_init.copy()
    Y_obs = true_function(X_obs)
    frames = []

    for it in range(n_iters + 1):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        mu, std = gp_posterior(X_obs, Y_obs, X_PLOT)
        y_best = Y_obs.min()
        ei = expected_improvement(mu, std, y_best)

        ax1.plot(
            X_PLOT,
            Y_TRUE,
            color=COLORS["true_fn"],
            lw=1.5,
            ls="--",
            alpha=0.35,
            label="True $f(x)$",
        )
        ax1.plot(X_PLOT, mu, color=COLORS["mean"], lw=2, label="GP mean")
        ax1.fill_between(
            X_PLOT,
            mu - 2 * std,
            mu + 2 * std,
            alpha=0.22,
            color=COLORS["ci"],
            label="95% CI",
        )
        ax1.scatter(
            X_obs,
            Y_obs,
            s=90,
            zorder=5,
            color=COLORS["obs"],
            edgecolors="k",
            linewidths=1,
        )

        best_idx = np.argmin(Y_obs)
        ax1.scatter(
            X_obs[best_idx],
            Y_obs[best_idx],
            s=160,
            zorder=6,
            marker="*",
            color=COLORS["next"],
            edgecolors="k",
            linewidths=1,
            label=f"Best so far: $f$={Y_obs[best_idx]:.2f}",
        )

        if it < n_iters:
            next_x_idx = np.argmax(ei)
            next_x = X_PLOT[next_x_idx]
            ax1.axvline(next_x, color=COLORS["next"], ls="--", lw=1.8, alpha=0.6)
            ax1.set_title(
                f"{label_prefix} — Iter {it + 1}/{n_iters} — "
                f"Querying $x = {next_x:.2f}$",
                fontsize=14,
            )
        else:
            ax1.set_title(
                f"{label_prefix} — Done! Best at $x = {X_obs[best_idx]:.2f}$,  "
                f"$f(x) = {Y_obs[best_idx]:.2f}$",
                fontsize=14,
            )

        ax1.set_ylabel("$f(x)$")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.set_xlim(0, 6)
        ax1.set_ylim(min(Y_TRUE) - 1.5, max(Y_TRUE) + 1.5)

        ax2.fill_between(X_PLOT, 0, ei, alpha=0.3, color=COLORS["acq"])
        ax2.plot(X_PLOT, ei, color=COLORS["acq"], lw=2)
        if it < n_iters:
            ax2.axvline(next_x, color=COLORS["next"], ls="--", lw=1.8, alpha=0.6)
            ax2.scatter(
                X_PLOT[next_x_idx],
                ei[next_x_idx],
                s=100,
                zorder=5,
                color=COLORS["next"],
                edgecolors="k",
                linewidths=1,
            )
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("EI$(x)$")
        ax2.set_xlim(0, 6)

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)

        if it < n_iters:
            next_x_val = X_PLOT[next_x_idx]
            X_obs = np.append(X_obs, next_x_val)
            Y_obs = np.append(Y_obs, true_function(next_x_val))

    frames += [frames[-1]] * 3
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=1500,
        loop=0,
    )
    print(f"  [{fig_num_label}] {os.path.basename(gif_path)}")
    return X_obs, Y_obs


def _bo_final_result(X_obs, Y_obs, png_path, title, fig_num_label):
    mu, std = gp_posterior(X_obs, Y_obs, X_PLOT)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(X_PLOT, Y_TRUE, color=COLORS["true_fn"], lw=2, label="True $f(x)$")
    ax.plot(X_PLOT, mu, color=COLORS["mean"], lw=2, ls="--", label="GP mean (final)")
    ax.fill_between(
        X_PLOT,
        mu - 2 * std,
        mu + 2 * std,
        alpha=0.2,
        color=COLORS["ci"],
        label="95% CI",
    )

    for i, (xi, yi) in enumerate(zip(X_obs, Y_obs)):
        ax.scatter(
            xi, yi, s=80, zorder=5, color=COLORS["obs"], edgecolors="k", linewidths=1
        )
        ax.annotate(
            f"{i + 1}",
            (xi, yi),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=9,
            fontweight="bold",
            color=COLORS["obs"],
        )

    best_idx = np.argmin(Y_obs)
    ax.scatter(
        X_obs[best_idx],
        Y_obs[best_idx],
        s=200,
        zorder=6,
        marker="*",
        color=COLORS["next"],
        edgecolors="k",
        linewidths=1.5,
        label=f"Best: $x$={X_obs[best_idx]:.2f}, $f$={Y_obs[best_idx]:.2f}",
    )

    true_min_idx = np.argmin(Y_TRUE)
    ax.scatter(
        X_PLOT[true_min_idx],
        Y_TRUE[true_min_idx],
        s=200,
        zorder=6,
        marker="D",
        color="white",
        edgecolors=COLORS["true_fn"],
        linewidths=2,
        label=f"True min: $x$={X_PLOT[true_min_idx]:.2f}, $f$={Y_TRUE[true_min_idx]:.2f}",
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(0, 6)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  [{fig_num_label}] {os.path.basename(png_path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures for Bayesian Optimization explainer...\n")
    fig01_blackbox()
    fig02_gp_prior()
    fig03_posterior_1()
    fig04_posterior_3()
    fig05_acquisition()
    fig06_exploration_exploitation()

    X_s, Y_s = _bo_loop_gif(
        X_init=np.array([2.0, 5.5]),
        n_iters=8,
        gif_path=f"{OUT}/07_bo_loop_success.gif",
        label_prefix="Success",
        fig_num_label="7/10",
    )
    _bo_final_result(
        X_s,
        Y_s,
        png_path=f"{OUT}/08_bo_success_result.png",
        title="Success — BO found the global minimum!",
        fig_num_label="8/10",
    )

    X_f, Y_f = _bo_loop_gif(
        X_init=np.array([0.5, 4.5]),
        n_iters=8,
        gif_path=f"{OUT}/09_bo_loop_local.gif",
        label_prefix="Local minimum",
        fig_num_label="9/10",
    )
    _bo_final_result(
        X_f,
        Y_f,
        png_path=f"{OUT}/10_bo_local_result.png",
        title="Stuck — BO converged to a local minimum",
        fig_num_label="10/10",
    )

    print(f"\nDone! All figures saved to '{OUT}/'")
