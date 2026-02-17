"""
Shared plotting utilities for Finance Tools.

Provides a consistent publication-quality style (SciencePlots + LaTeX),
a colorblind-safe palette (Tol Bright), semantic color/marker mappings,
standard figure sizes, and reusable plotting helpers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =====================================================================
# TOL BRIGHT PALETTE (colorblind-safe, 7 distinct hues)
# =====================================================================

PALETTE = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

# -- Semantic role mapping --------------------------------------------

COLORS = {
    # Annotation
    "true_value": "#000000",
    "ci_bounds": PALETTE["grey"],
    "target_line": PALETTE["grey"],
    # Bar chart slots
    "bar_1": PALETTE["blue"],
    "bar_2": PALETTE["yellow"],
    "bar_3": PALETTE["green"],
}

MARKERS = {}

# =====================================================================
# FIGURE SIZES
# =====================================================================

FIGSIZE = {
    "single": (3.5, 2.8),
    "single_square": (3.5, 3.5),
    "double": (7.0, 3.5),
    "double_tall": (7.0, 5.5),
    "double_square": (7.0, 7.0),
}


def figsize_grid(n_rows, n_cols, per_panel=(3.5, 2.8)):
    """Compute figure size for an n_rows x n_cols grid of panels."""
    return (per_panel[0] * n_cols, per_panel[1] * n_rows)


# =====================================================================
# STYLE SETUP
# =====================================================================

_STYLE_APPLIED = False


def setup_style():
    """
    Apply SciencePlots 'science' style with LaTeX rendering and Tol Bright
    color cycle.  Falls back gracefully if LaTeX is unavailable.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    try:
        plt.style.use(["science"])
    except Exception:
        pass

    cycle_colors = [
        PALETTE["blue"], PALETTE["red"], PALETTE["green"],
        PALETTE["yellow"], PALETTE["cyan"], PALETTE["purple"],
        PALETTE["grey"],
    ]

    rc_overrides = {
        "text.usetex": True,
        "font.family": "serif",
        "axes.prop_cycle": plt.cycler("color", cycle_colors),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
    }

    # Test LaTeX availability
    try:
        fig_test = plt.figure(figsize=(1, 1))
        fig_test.text(0.5, 0.5, r"$x$")
        fig_test.savefig(os.devnull, format="png")
        plt.close(fig_test)
    except Exception:
        rc_overrides["text.usetex"] = False

    plt.rcParams.update(rc_overrides)
    _STYLE_APPLIED = True


# =====================================================================
# PLOTTING HELPERS
# =====================================================================

def gaussian_pdf(x, mu, sigma, peak_normalize=True):
    """
    Evaluate a Gaussian PDF on array x.

    If peak_normalize is True, the peak is scaled to 1.0.
    """
    pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    if not peak_normalize:
        pdf /= sigma * np.sqrt(2 * np.pi)
    return pdf


def plot_kde_1d(samples, ax, color, label, peak_normalize=True,
                x_range=None, fill=False, lw=2, ls="-", alpha=1.0):
    """
    1D KDE via scipy.stats.gaussian_kde.

    Returns the Line2D artist.
    """
    kde = gaussian_kde(samples)
    if x_range is None:
        lo = samples.min() - 0.5 * samples.std()
        hi = samples.max() + 0.5 * samples.std()
        x_range = np.linspace(lo, hi, 300)
    y = kde(x_range)
    if peak_normalize and y.max() > 0:
        y = y / y.max()
    line, = ax.plot(x_range, y, color=color, lw=lw, ls=ls, alpha=alpha,
                    label=label)
    if fill:
        ax.fill_between(x_range, y, alpha=0.15, color=color)
    return line


def plot_histogram(data, ax, bins=25, color=None, peak_normalize=False,
                   edgecolor="white", alpha=0.8, label=None, range=None):
    """
    Styled histogram.

    Returns (counts, bin_edges, patches).
    """
    if color is None:
        color = PALETTE["blue"]
    n, edges, patches = ax.hist(
        data, bins=bins, color=color, edgecolor=edgecolor,
        linewidth=0.4, alpha=alpha, label=label, range=range,
    )
    if peak_normalize and n.max() > 0:
        for p, h in zip(patches, n):
            p.set_height(h / n.max())
        ax.set_ylim(0, 1.15)
    return n, edges, patches


def plot_contour_2d(samples, ax, color, label=None, levels_sigma=(1, 2),
                    filled=True, alpha=0.3, scatter=False, n_grid=100):
    """
    2D KDE contour plot at sigma levels.

    Parameters
    ----------
    samples : ndarray, shape (N, 2)
    levels_sigma : tuple of ints
        Sigma levels for contours (e.g. (1, 2) for 1-sigma and 2-sigma).
    filled : bool
        If True, draw filled contours.
    scatter : bool
        If True, overlay scatter points.

    Returns the contour set, or scatter artist if KDE fails.
    """
    x, y = samples[:, 0], samples[:, 1]

    # Fallback: if too few unique points, just scatter
    if len(np.unique(x)) < 5 or len(np.unique(y)) < 5:
        art = ax.scatter(x, y, s=2, alpha=0.3, color=color, label=label)
        return art

    try:
        kde = gaussian_kde(samples.T)
    except np.linalg.LinAlgError:
        art = ax.scatter(x, y, s=2, alpha=0.3, color=color, label=label)
        return art

    # Build evaluation grid
    x_margin = 0.15 * (x.max() - x.min())
    y_margin = 0.15 * (y.max() - y.min())
    xg = np.linspace(x.min() - x_margin, x.max() + x_margin, n_grid)
    yg = np.linspace(y.min() - y_margin, y.max() + y_margin, n_grid)
    Xg, Yg = np.meshgrid(xg, yg)
    positions = np.vstack([Xg.ravel(), Yg.ravel()])
    Z = kde(positions).reshape(Xg.shape)

    # Convert sigma levels to density thresholds
    # Evaluate KDE at sample points, sort to find density thresholds
    density_at_samples = kde(samples.T)
    sorted_density = np.sort(density_at_samples)

    # For Gaussian: fraction inside sigma ellipse
    # 1-sigma ~ 39.3%, 2-sigma ~ 86.5% (for 2D)
    from scipy.stats import chi2
    levels = []
    for s in sorted(levels_sigma, reverse=True):
        frac = chi2.cdf(s**2, df=2)  # fraction inside s-sigma ellipse
        idx = int((1 - frac) * len(sorted_density))
        idx = max(0, min(idx, len(sorted_density) - 1))
        levels.append(sorted_density[idx])

    levels = sorted(levels)

    if scatter:
        ax.scatter(x, y, s=1, alpha=0.08, color=color)

    cs = None
    if filled:
        cs = ax.contourf(Xg, Yg, Z, levels=levels + [Z.max()],
                         colors=[color], alpha=alpha)
    cs_lines = ax.contour(Xg, Yg, Z, levels=levels, colors=[color],
                          linewidths=1.0, alpha=0.8)

    # Add label via invisible proxy
    if label:
        ax.plot([], [], color=color, lw=2, label=label)

    return cs_lines


def plot_metric_vs_density(ax, x_densities, y_values, prior_type, label=None):
    """
    Line + marker plot for a metric vs data density.

    Uses semantic color/marker for the prior_type key.
    """
    color = COLORS.get(prior_type, PALETTE["blue"])
    marker = MARKERS.get(prior_type, "o")
    if label is None:
        label = prior_type
    ax.plot(x_densities, y_values, marker=marker, color=color,
            lw=2, markersize=5, label=label)


def mark_true_value(ax, value, direction="vertical",
                    color=None, lw=1.5, ls=":", alpha=0.8, label=None):
    """Draw a vertical or horizontal line marking a true value."""
    if color is None:
        color = COLORS["true_value"]
    if direction == "vertical":
        ax.axvline(value, color=color, lw=lw, ls=ls, alpha=alpha, label=label)
    else:
        ax.axhline(value, color=color, lw=lw, ls=ls, alpha=alpha, label=label)


def savefig(fig, path, dpi=300, close=True):
    """Save figure, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"Saved {path}")
