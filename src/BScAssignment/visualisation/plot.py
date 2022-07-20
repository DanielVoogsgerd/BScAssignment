import numpy as np
import numpy.linalg as LA
import sympy

def plot_band_structure(ax, t, band_structure, symmetry_points=None):
    ax.set_xlabel("Wavevector")
    ax.set_ylabel("Energy (eV)")

    if symmetry_points is not None:
        ax.set_xticks(range(len(symmetry_points)))
        ax.set_xticklabels([f"${symmetry_point}$" for symmetry_point in symmetry_points])

    ax.grid(which='major', axis='x', linewidth='0.5', color='lightgrey')

    ax.axhline(color='lightgrey', linewidth=0.5, linestyle="--")

    ax.set_xlim([t[0], t[-1]])

    for band in band_structure:
        ax.plot(t, band, color="black", linewidth=1)


# Plot band structure
def visualize_fit(fig, parameter, original_eigenvalues, fitted_eigenvalues, symmetry_points):
    ax1 = fig.add_subplot(221)
    ax1.set_title("Input bandstructure")
    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
    ax2.set_title("Fitted bandstructure")
    ax3 = fig.add_subplot(223)
    ax3.set_title("Error")
    ax4 = fig.add_subplot(224)
    ax4.set_title("Histogram Error")

    error = original_eigenvalues - fitted_eigenvalues
    plot_band_structure(ax1, parameter, original_eigenvalues, symmetry_points)
    plot_band_structure(ax2, parameter, fitted_eigenvalues, symmetry_points)
    plot_band_structure(ax3, parameter, error, symmetry_points)
    ax4.hist(error)
