import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# --- Atomic Units Conversion ---
angstrom_to_bohr = 1.889726

# --- Problem 4 Functions (unchanged) ---

def S_ab(k, R):
    """Calculates the overlap integral S_ab"""
    kR = k * R
    return np.exp(-kR) * (1 + kR + (kR**2) / 3.0)

def H_aa(k, R):
    """Calculates the Hamiltonian matrix element H_aa"""
    return (k**2 / 2.0) - k - (1.0 / R) + np.exp(-2 * k * R) * (k + 1.0 / R)

def H_ab(k, R):
    """Calculates the Hamiltonian matrix element H_ab"""
    s_ab = S_ab(k, R)
    kR = k * R
    term1 = -0.5 * k**2 * s_ab
    term2 = k * (k - 2) * (1 + kR) * np.exp(-kR)
    return term1 + term2

def calculate_energies(k, R):
    """Calculates the ground state (e1) and excited state (e2) energies."""
    s_ab = S_ab(k, R)
    h_aa = H_aa(k, R)
    h_ab = H_ab(k, R)
    
    epsilon_1 = (h_aa + h_ab) / (1 + s_ab)
    
    if (1 - s_ab) < 1e-6:
        epsilon_2 = np.nan
    else:
        epsilon_2 = (h_aa - h_ab) / (1 - s_ab)
        
    return epsilon_1, epsilon_2

# --- Problem 5: Generate data for 3D plot & Optimal Path ---

R_angstrom_vals = np.linspace(0.8, 2.0, 200)
k_vals = np.linspace(0.4, 2.0, 200)

R_grid_angstrom, K_grid = np.meshgrid(R_angstrom_vals, k_vals, indexing='ij')
R_grid_bohr = R_grid_angstrom * angstrom_to_bohr
E_grid = np.zeros(R_grid_angstrom.shape)

# Lists to store the optimal path
R_path_angstrom = []
k_path_optimal = []
E_path_min = []

# Calculate the energy for each (R, k) point
for i in range(len(R_angstrom_vals)):
    
    energies_for_this_R = [] # Store energies for one R
    
    for j in range(len(k_vals)):
        R_val = R_grid_bohr[i, j]
        k_val = K_grid[i, j]
        
        e1, _ = calculate_energies(k_val, R_val)
        E_grid[i, j] = e1
        energies_for_this_R.append(e1)
        
    # Find and store the minimum for this R
    min_energy = np.min(energies_for_this_R)
    min_k_index = np.argmin(energies_for_this_R)
    
    R_path_angstrom.append(R_angstrom_vals[i])
    k_path_optimal.append(k_vals[min_k_index])
    E_path_min.append(min_energy)

# --- Combined Plotting (3D Surface + 2D Contour) ---
fig = plt.figure(figsize=(16, 7))

# --- Plot 1: 3D Surface Plot ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

# Plot the 3D surface
surf = ax1.plot_surface(R_grid_angstrom, K_grid, E_grid, 
                        cmap=cm.viridis,
                        linewidth=0, 
                        antialiased=False,
                        alpha=0.8) # Slightly transparent

# Plot the optimal k-path line on the 3D surface
ax1.plot(R_path_angstrom, k_path_optimal, E_path_min, 
        'r-', 
        linewidth=1.5, 
        label='Optimal $k(R)$ Path', 
        zorder=10)

# Set labels and title
ax1.set_xlabel('R (Å)')
ax1.set_ylabel('k (parameter)')
ax1.set_zlabel('Energy (Hartrees)')
ax1.set_title('Energy Surface $E(R, k)$')
ax1.view_init(elev=20, azim=-1200) # Adjust view angle
ax1.legend()
fig.colorbar(surf, shrink=0.5, aspect=10, ax=ax1, label='Energy')


# --- Plot 2: 2D Contour Plot ---
ax2 = fig.add_subplot(1, 2, 2)

# Create contour plot
# We use E_grid.T because R_angstrom_vals is X and k_vals is Y
contour = ax2.contourf(R_angstrom_vals, k_vals, E_grid.T, 
                         levels=25, cmap=cm.viridis)
ax2.contour(R_angstrom_vals, k_vals, E_grid.T, 
              levels=25, colors='k', linewidths=0.5)

# Plot the path of optimal k
ax2.plot(R_path_angstrom, k_path_optimal, 'r--o', 
         label='Optimal $k(R)$ Path', markersize=3, linewidth=2)

ax2.set_xlabel('Internuclear Distance R (Å)')
ax2.set_ylabel('Variational Parameter k')
ax2.set_title('Optimal $k$ as a function of R')
ax2.legend()
plt.colorbar(contour, ax=ax2, label='Energy (Hartrees)')

# --- Finalize ---
plt.tight_layout()
plt.savefig('combined_3D_and_2D_plot.png')
plt.show()